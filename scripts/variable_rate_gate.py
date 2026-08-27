"""Real-rANS acceptance checks for the V4 LIC -> Plain DiT chain."""

import argparse
import hashlib
import json
import math
from pathlib import Path


QUALITY_NAMES = tuple(f"q{index}" for index in range(6))


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


def check_rate_ladder(metrics, min_bpp_span, q0_max_bpp, q5_min_bpp, q5_max_bpp):
    bpp = [metrics[quality]["bpp"] for quality in QUALITY_NAMES]
    failures = []
    if any(left >= right for left, right in zip(bpp, bpp[1:])):
        failures.append("real BPP must be strictly increasing from q0 through q5")
    if bpp[-1] - bpp[0] < min_bpp_span:
        failures.append(
            f"real BPP span {bpp[-1] - bpp[0]:.6f} is below {min_bpp_span:.6f}"
        )
    if bpp[0] > q0_max_bpp:
        failures.append(f"q0 real BPP {bpp[0]:.6f} exceeds {q0_max_bpp:.6f}")
    if bpp[-1] < q5_min_bpp or bpp[-1] > q5_max_bpp:
        failures.append(
            f"q5 real BPP {bpp[-1]:.6f} is outside "
            f"[{q5_min_bpp:.6f}, {q5_max_bpp:.6f}]"
        )
    return failures


def mean_metric(metrics, key):
    return sum(metrics[quality][key] for quality in QUALITY_NAMES) / len(QUALITY_NAMES)


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
        failures = check_rate_ladder(
            metrics,
            args.min_bpp_span,
            args.q0_max_bpp,
            args.q5_min_bpp,
            args.q5_max_bpp,
        )
        candidates.append(
            {
                "name": name,
                "checkpoint": str(Path(checkpoint).resolve()),
                "summary": str(Path(summary_path).resolve()),
                "evaluation_dir": str(Path(evaluation_dir).resolve()),
                "summary_checkpoint": summary.get("checkpoint"),
                "accepted": not failures,
                "failures": failures,
                "qualities": metrics,
                "mean_bpp": mean_metric(metrics, "bpp"),
                "mean_psnr": mean_metric(metrics, "psnr"),
                "mean_lpips": mean_metric(metrics, "lpips"),
                "mean_dists": mean_metric(metrics, "dists"),
            }
        )

    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    report = {
        "gate": "A_real_rans_lic",
        "criteria": {
            "strictly_increasing_bpp": True,
            "min_bpp_span": args.min_bpp_span,
            "q0_max_bpp": args.q0_max_bpp,
            "q5_bpp_range": [args.q5_min_bpp, args.q5_max_bpp],
            "selection": "highest mean PSNR, then lower mean LPIPS, DISTS, BPP",
        },
        "candidates": candidates,
    }
    if accepted:
        selected = min(
            accepted,
            key=lambda candidate: (
                -candidate["mean_psnr"],
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
    select.add_argument("--q0-max-bpp", type=float, default=0.03)
    select.add_argument("--q5-min-bpp", type=float, default=0.045)
    select.add_argument("--q5-max-bpp", type=float, default=0.08)
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
