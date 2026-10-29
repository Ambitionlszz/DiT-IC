"""Compare C/D at common actual total bpp by log-rate interpolation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def compare(c, d, targets=None):
    for key in ("dataset_fingerprint", "images", "metric_protocol", "weights", "step", "phase"):
        if key not in c or c[key] != d.get(key):
            raise ValueError(f"C/D comparison requires identical {key}")
    if c["backend"] != "C" or d["backend"] != "D":
        raise ValueError("provide C followed by D")
    curves = {}
    for name, source in (("C", c), ("D", d)):
        rows = sorted(source["summary"].values(), key=lambda item: item["bpp"])
        rates = np.array([row["bpp"] for row in rows])
        if not np.all(np.isfinite(rates)) or np.any(rates <= 0) or len(np.unique(rates)) < 3:
            raise ValueError("each curve must have at least three distinct positive actual rates")
        # Average ties, retaining every measured quality point.
        unique = np.unique(rates)
        curves[name] = (unique, {metric: np.array([np.mean([row[metric] for row in rows if row["bpp"] == rate])
                                                  for rate in unique]) for metric in ("psnr", "lpips", "dists")})
    lo = max(values[0].min() for values in curves.values())
    hi = min(values[0].max() for values in curves.values())
    if not lo < hi:
        raise ValueError("the measured curves have no overlapping actual-rate interval")
    targets = np.array(targets) if targets is not None else np.geomspace(lo, hi, 3)
    if len(np.unique(targets)) < 3 or not np.all(np.isfinite(targets)) or np.any(targets < lo) or np.any(targets > hi):
        raise ValueError("supply at least three distinct common rates inside the overlap; extrapolation is forbidden")
    rows = []
    for target in sorted(targets):
        row = {"bpp": float(target)}
        for metric in ("psnr", "lpips", "dists"):
            for name, (rate, values) in curves.items():
                row[f"{name}_{metric}"] = float(np.interp(np.log(target), np.log(rate), values[metric]))
            row[f"D_minus_C_{metric}"] = row[f"D_{metric}"]-row[f"C_{metric}"]
        rows.append(row)
    return {"comparison": "linear interpolation in log(actual total bpp), including all packet overhead; no extrapolation",
            "interpretation": "positive PSNR delta and negative LPIPS/DISTS delta favor D; interpolated estimates require inspection of raw curves",
            "overlap_bpp": [float(lo), float(hi)], "weights": c["weights"], "phase": c["phase"], "step": c["step"],
            "C_checkpoint_sha256": c["checkpoint_sha256"], "D_checkpoint_sha256": d["checkpoint_sha256"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", required=True, help="C evaluation summary.json")
    parser.add_argument("--d", required=True, help="D evaluation summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bpp", type=float, nargs="+", help="at least three target rates, inside the measured overlap")
    args = parser.parse_args()
    result = compare(json.loads(Path(args.c).read_text()), json.loads(Path(args.d).read_text()), args.bpp)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    with (output/"comparison.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
