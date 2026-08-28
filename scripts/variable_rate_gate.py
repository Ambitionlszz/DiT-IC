"""Real-rANS acceptance checks for the variable-rate LIC -> Plain DiT chain."""

import argparse
import hashlib
import json
import math
from pathlib import Path


QUALITY_NAMES = tuple(f"q{index}" for index in range(6))
LIC_SUMMARY_SCHEMA_VERSION = 1


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def validated_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 hex digest")
    value = value.lower()
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    return value


def direct_lic_metrics(summary_path):
    summary = load_json(summary_path)
    qualities = summary.get("qualities", {})
    metrics = {}
    for quality in QUALITY_NAMES:
        if quality not in qualities:
            raise ValueError(f"{summary_path} is missing {quality}")
        result = qualities[quality]
        metrics[quality] = {
            key: finite(result[key], f"{quality}.{key}")
            for key in ("bpp", "psnr", "lpips", "dists", "ms_ssim")
        }
    return summary, metrics


def validate_lic_evaluation_binding(
    checkpoint, summary_path, evaluation_dir, summary
):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"missing LIC checkpoint: {checkpoint}")
    checkpoint = checkpoint.resolve()
    summary_path = Path(summary_path).resolve()
    expected_summary = (Path(evaluation_dir) / "summary.json").resolve()
    if summary_path != expected_summary:
        raise ValueError(
            f"LIC summary {summary_path} is not the summary for {evaluation_dir}"
        )
    if int(summary.get("summary_schema_version", 0)) != LIC_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"LIC summary must use schema {LIC_SUMMARY_SCHEMA_VERSION}"
        )
    if summary.get("weights") != "ema":
        raise ValueError("Gate A requires EMA LIC evaluation weights")
    summary_checkpoint = Path(summary.get("checkpoint", "")).resolve()
    if summary_checkpoint != checkpoint:
        raise ValueError(
            f"LIC summary checkpoint {summary_checkpoint} does not match {checkpoint}"
        )
    expected_checkpoint_sha256 = validated_sha256(
        summary.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    actual_checkpoint_sha256 = sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"LIC checkpoint SHA256 does not match summary: {checkpoint}")
    image_manifest_sha256 = validated_sha256(
        summary.get("image_manifest_sha256"), "image_manifest_sha256"
    )
    num_images = int(summary.get("num_images", 0))
    if num_images < 1:
        raise ValueError("LIC summary num_images must be positive")
    return {
        "checkpoint_sha256": actual_checkpoint_sha256,
        "image_manifest_sha256": image_manifest_sha256,
        "num_images": num_images,
        "weights": "ema",
    }


def check_rate_ladder(
    metrics, min_bpp_span, q0_max_bpp=None, q5_min_bpp=None, q5_max_bpp=None
):
    bpp = [metrics[quality]["bpp"] for quality in QUALITY_NAMES]
    failures = []
    if any(left >= right for left, right in zip(bpp, bpp[1:])):
        failures.append("real BPP must be strictly increasing from q0 through q5")
    if bpp[-1] - bpp[0] < min_bpp_span:
        failures.append(
            f"real BPP span {bpp[-1] - bpp[0]:.6f} is below {min_bpp_span:.6f}"
        )
    if q0_max_bpp is not None and bpp[0] > q0_max_bpp:
        failures.append(f"q0 real BPP {bpp[0]:.6f} exceeds {q0_max_bpp:.6f}")
    if q5_min_bpp is not None and bpp[-1] < q5_min_bpp:
        failures.append(
            f"q5 real BPP {bpp[-1]:.6f} is below {q5_min_bpp:.6f}"
        )
    if q5_max_bpp is not None and bpp[-1] > q5_max_bpp:
        failures.append(f"q5 real BPP {bpp[-1]:.6f} exceeds {q5_max_bpp:.6f}")
    return failures


def check_quality_ladder(metrics):
    failures = []
    for key in ("psnr", "ms_ssim"):
        values = [metrics[quality][key] for quality in QUALITY_NAMES]
        if any(left > right for left, right in zip(values, values[1:])):
            failures.append(f"{key} must be non-decreasing from q0 through q5")
    for key in ("lpips", "dists"):
        values = [metrics[quality][key] for quality in QUALITY_NAMES]
        if any(left < right for left, right in zip(values, values[1:])):
            failures.append(f"{key} must be non-increasing from q0 through q5")
    return failures


def mean_metric(metrics, key):
    return sum(metrics[quality][key] for quality in QUALITY_NAMES) / len(QUALITY_NAMES)


def interpolate_metric(metrics, key, target_bpp):
    points = [(metrics[quality]["bpp"], metrics[quality][key]) for quality in QUALITY_NAMES]
    for (left_bpp, left_value), (right_bpp, right_value) in zip(points, points[1:]):
        if left_bpp <= target_bpp <= right_bpp:
            fraction = (target_bpp - left_bpp) / (right_bpp - left_bpp)
            return left_value + fraction * (right_value - left_value)
    raise ValueError(f"cannot interpolate {key} at BPP {target_bpp:.6f}")


def attach_matched_rd_scores(candidates):
    if len(candidates) == 1:
        candidates[0]["matched_mean_psnr"] = candidates[0]["mean_psnr"]
        candidates[0]["matched_bpp_range"] = [
            candidates[0]["qualities"]["q0"]["bpp"],
            candidates[0]["qualities"]["q5"]["bpp"],
        ]
        return
    lower = max(candidate["qualities"]["q0"]["bpp"] for candidate in candidates)
    upper = min(candidate["qualities"]["q5"]["bpp"] for candidate in candidates)
    if lower >= upper:
        raise ValueError("accepted LIC candidates have no common real-BPP interval")
    anchors = [lower + (upper - lower) * index / 4.0 for index in range(5)]
    for candidate in candidates:
        candidate["matched_mean_psnr"] = sum(
            interpolate_metric(candidate["qualities"], "psnr", anchor)
            for anchor in anchors
        ) / len(anchors)
        candidate["matched_bpp_range"] = [lower, upper]


def select_lic(args):
    candidates = []
    for name, checkpoint, summary_path, evaluation_dir in (
        ("last", args.last_checkpoint, args.last_summary, args.last_eval_dir),
        ("best", args.best_checkpoint, args.best_summary, args.best_eval_dir),
    ):
        if not (checkpoint or summary_path or evaluation_dir):
            continue
        if not all((checkpoint, summary_path, evaluation_dir)):
            raise ValueError(f"{name} candidate needs checkpoint, summary, and eval dir")
        summary, metrics = direct_lic_metrics(summary_path)
        binding = validate_lic_evaluation_binding(
            checkpoint, summary_path, evaluation_dir, summary
        )
        failures = check_rate_ladder(
            metrics,
            args.min_bpp_span,
            args.q0_max_bpp,
            args.q5_min_bpp,
            args.q5_max_bpp,
        )
        failures.extend(check_quality_ladder(metrics))
        candidates.append(
            {
                "name": name,
                "checkpoint": str(Path(checkpoint).resolve()),
                "summary": str(Path(summary_path).resolve()),
                "evaluation_dir": str(Path(evaluation_dir).resolve()),
                "summary_checkpoint": summary.get("checkpoint"),
                **binding,
                "accepted": not failures,
                "failures": failures,
                "qualities": metrics,
                "mean_bpp": mean_metric(metrics, "bpp"),
                "mean_psnr": mean_metric(metrics, "psnr"),
                "mean_lpips": mean_metric(metrics, "lpips"),
                "mean_dists": mean_metric(metrics, "dists"),
            }
        )

    evaluation_sets = {
        (candidate["image_manifest_sha256"], candidate["num_images"])
        for candidate in candidates
    }
    if len(evaluation_sets) != 1:
        raise ValueError("LIC candidates were not evaluated on the same image set")

    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    report = {
        "gate": "A_real_rans_lic",
        "criteria": {
            "strictly_increasing_bpp": True,
            "min_bpp_span": args.min_bpp_span,
            "monotonic_quality_metrics": ["psnr", "ms_ssim", "lpips", "dists"],
            "q0_max_bpp": args.q0_max_bpp,
            "q5_bpp_range": [args.q5_min_bpp, args.q5_max_bpp],
            "selection": "highest PSNR interpolated over the common real-BPP interval, then lower mean LPIPS, DISTS, BPP",
        },
        "candidates": candidates,
    }
    if accepted:
        attach_matched_rd_scores(accepted)
        selected = min(
            accepted,
            key=lambda candidate: (
                -candidate["matched_mean_psnr"],
                candidate["mean_lpips"],
                candidate["mean_dists"],
                candidate["mean_bpp"],
            ),
        )
        report["selected"] = {
            key: selected[key]
            for key in ("name", "checkpoint", "summary", "evaluation_dir")
        }
    write_json(args.output, report)
    if not accepted:
        raise SystemExit(f"Gate A rejected every LIC candidate; see {args.output}")
    print(report["selected"]["checkpoint"])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"missing bitstream directory: {root}")
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def plain_metrics(root, quality):
    matches = sorted((Path(root) / quality / "metrics").glob("*/summary_metrics.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Plain DiT summary for {quality}, found {len(matches)}"
        )
    return matches[0], load_json(matches[0])


def verify_p0(args):
    _, lic = direct_lic_metrics(args.lic_summary)
    p0_root = Path(args.p0_root)
    report = {"gate": "B_frozen_lic_bitstream", "qualities": {}}
    p0_bpp = []
    for quality in QUALITY_NAMES:
        summary_path, summary = plain_metrics(p0_root, quality)
        reference = collect_files(Path(args.lic_eval_dir) / quality / "bin")
        candidate = collect_files(p0_root / quality / "bin")
        if reference != candidate:
            raise SystemExit(f"Gate B failed: {quality} bitstreams are not byte-identical")
        metrics = {
            key: finite(summary[key], f"{quality}.{key}")
            for key in ("avg_bpp", "psnr", "lpips", "dists", "ms_ssim")
        }
        p0_bpp.append(metrics["avg_bpp"])
        report["qualities"][quality] = {
            "lic_bpp": lic[quality]["bpp"],
            "p0": metrics,
            "plain_summary": str(summary_path.resolve()),
            "byte_identical": True,
        }
    if any(left >= right for left, right in zip(p0_bpp, p0_bpp[1:])):
        raise SystemExit("Gate B failed: Plain DiT real BPP is not strictly increasing")
    report["accepted"] = True
    write_json(args.output, report)
    print(f"Gate B accepted: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-lic")
    select.add_argument("--last-checkpoint", required=True)
    select.add_argument("--last-summary", required=True)
    select.add_argument("--last-eval-dir", required=True)
    select.add_argument("--best-checkpoint")
    select.add_argument("--best-summary")
    select.add_argument("--best-eval-dir")
    select.add_argument("--output", required=True)
    select.add_argument("--min-bpp-span", type=float, default=0.02)
    select.add_argument("--q0-max-bpp", type=float)
    select.add_argument("--q5-min-bpp", type=float)
    select.add_argument("--q5-max-bpp", type=float)
    select.set_defaults(function=select_lic)

    verify = commands.add_parser("verify-p0")
    verify.add_argument("--lic-summary", required=True)
    verify.add_argument("--lic-eval-dir", required=True)
    verify.add_argument("--p0-root", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(function=verify_p0)

    args = parser.parse_args()
    try:
        args.function(args)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid gate input: {exc}") from exc


if __name__ == "__main__":
    main()
