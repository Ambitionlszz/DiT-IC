"""Real-rANS acceptance checks for the variable-rate LIC -> Plain DiT chain."""

import argparse
import hashlib
import json
import math
from pathlib import Path


QUALITY_NAMES = tuple(f"q{index}" for index in range(6))
LIC_SUMMARY_SCHEMA_VERSION = 2
LEGACY_METRIC_PROTOCOL = {
    "version": "legacy_pyiqa_y_float_unverified_v1",
    "pixel_domain": "float",
    "ms_ssim_alias": "ms_ssim_y",
    "provenance": "inferred from historical evaluators; metric implementation and package versions were not recorded",
}


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


def evaluation_contract(summary, lic=False):
    """Read protocol identity without importing training or metric packages.

    Historical LIC schema 1 and Plain summaries omitted protocol metadata.
    Preserve their original workflow with an explicit unverified legacy marker;
    they cannot be compared with newly measured, fully specified reports.
    """
    schema = int(summary.get("summary_schema_version", 0))
    if lic and schema not in (1, LIC_SUMMARY_SCHEMA_VERSION):
        raise ValueError("LIC summary must use schema 1 or 2")
    protocol = summary.get("metric_protocol")
    if protocol is None and schema != LIC_SUMMARY_SCHEMA_VERSION:
        return {"metric_protocol": dict(LEGACY_METRIC_PROTOCOL), "dataset_content_sha256": None}
    if not isinstance(protocol, dict):
        raise ValueError("schema 2 evaluation requires a structured metric_protocol")
    if protocol.get("version") != "image_quality_rgb_y_v2":
        raise ValueError("unsupported metric_protocol version")
    if protocol.get("pixel_domain") not in ("float", "uint8_roundtrip"):
        raise ValueError("metric_protocol pixel_domain must be float or uint8_roundtrip")
    if protocol.get("ms_ssim_alias") != "ms_ssim_rgb":
        raise ValueError("schema 2 metric_protocol must alias ms_ssim to ms_ssim_rgb")
    for name in ("ms_ssim_rgb", "ms_ssim_y"):
        if not isinstance(protocol.get(name), dict):
            raise ValueError(f"metric_protocol is missing the {name} definition")
    content = validated_sha256(summary.get("dataset_content_sha256"), "dataset_content_sha256")
    return {"metric_protocol": protocol, "dataset_content_sha256": content}


def validate_ms_ssim_alias(values, protocol, label):
    if protocol == LEGACY_METRIC_PROTOCOL:
        return
    alias = protocol["ms_ssim_alias"]
    # Both explicit channels must be present in new dual-channel reports.
    for name in ("ms_ssim_rgb", "ms_ssim_y"):
        if name not in values:
            raise ValueError(f"{label} is missing {name}")
        finite(values[name], f"{label}.{name}")
    actual = finite(values.get("ms_ssim"), f"{label}.ms_ssim")
    expected = finite(values[alias], f"{label}.{alias}")
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label}.ms_ssim disagrees with metric_protocol alias {alias}")


def require_matching_contracts(reference, candidate, label):
    if reference["metric_protocol"] != candidate["metric_protocol"]:
        raise ValueError(f"{label} metric protocols differ (channel, pixel domain, or implementation)")
    if reference["dataset_content_sha256"] != candidate["dataset_content_sha256"]:
        raise ValueError(f"{label} were not evaluated on the same image set (dataset_content_sha256 differs)")


def direct_lic_metrics(summary_path):
    summary = load_json(summary_path)
    contract = evaluation_contract(summary, lic=True)
    qualities = summary.get("qualities", {})
    metrics = {}
    for quality in QUALITY_NAMES:
        if quality not in qualities:
            raise ValueError(f"{summary_path} is missing {quality}")
        result = qualities[quality]
        validate_ms_ssim_alias(result, contract["metric_protocol"], quality)
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
    contract = evaluation_contract(summary, lic=True)
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
        **contract,
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

    for candidate in candidates[1:]:
        require_matching_contracts(candidates[0], candidate, "LIC candidates")
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
            "matching_metric_protocol_and_image_contents": True,
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


def collect_files(root, expected_count=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"missing bitstream directory: {root}")
    if expected_count is not None and (
        isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1
    ):
        raise ValueError("bitstream num_images must be a positive integer")
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name.endswith(".bin"):
            name = name[: -len(".bin")]
        if name in files:
            raise ValueError(
                f"duplicate bitstream identity after normalizing .bin suffix: {name}"
            )
        files[name] = (path.stat().st_size, sha256(path))
    if not files:
        raise ValueError(f"no bitstream files found in {root}")
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(
            f"bitstream file count in {root} is {len(files)}, but num_images is {expected_count}"
        )
    return files


def plain_metrics(root, quality):
    matches = sorted((Path(root) / quality / "metrics").glob("*/summary_metrics.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Plain DiT summary for {quality}, found {len(matches)}"
        )
    return matches[0], load_json(matches[0])


def verify_p0(args):
    lic_summary, lic = direct_lic_metrics(args.lic_summary)
    lic_contract = evaluation_contract(lic_summary, lic=True)
    p0_root = Path(args.p0_root)
    report = {"gate": "B_frozen_lic_bitstream", "qualities": {}, **lic_contract}
    p0_bpp = []
    for quality in QUALITY_NAMES:
        summary_path, summary = plain_metrics(p0_root, quality)
        plain_contract = evaluation_contract(summary)
        require_matching_contracts(lic_contract, plain_contract, f"LIC and Plain {quality}")
        validate_ms_ssim_alias(summary, plain_contract["metric_protocol"], f"Plain {quality}")
        # Older Plain reports may omit num_images. Use either available count
        # for both directories; equal, partially deleted sets must not pass.
        lic_count, plain_count = lic_summary.get("num_images"), summary.get("num_images")
        reference = collect_files(Path(args.lic_eval_dir) / quality / "bin",
                                  expected_count=lic_count if lic_count is not None else plain_count)
        candidate = collect_files(p0_root / quality / "bin",
                                  expected_count=plain_count if plain_count is not None else lic_count)
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
            "bitstream_files": len(reference),
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
