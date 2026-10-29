"""Verify completed packet evaluations and export the requested six-rate tables."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "bt_self_flow_v1"
METRICS = ("bpp", "psnr", "lpips", "ms_ssim", "dists")


def main():
    exported, identities = [], []
    parts = ["**Kodak24 六档真实码流指标**", "",
             "C 使用 36K、D 使用 38K deployment EMA；两组步数不同。所有指标来自同一份浮点重建，未经 PNG 再量化。",
             "", "PSNR / MS-SSIM 越高越好，LPIPS / DISTS 越低越好。bpp 包含全部压缩包开销。",
             "", "MS-SSIM 使用与旧 baseline 相同的 pyiqa 默认 Y 通道协议。", ""]
    protocols = {}
    for name, step in (("C", 36000), ("D", 38000)):
        path = ROOT / (f"D_{step}_gpu0" if name == "D" else f"{name}_{step}")
        manifest = json.loads((path / "summary.json").read_text())
        validation = json.loads((RUN / name / "joint" / f"validation_{step:07d}.json").read_text())
        assert manifest["step"] == step and manifest["weights"] == "ema"
        assert manifest["images"] == 24 and manifest["nfe"] == 1
        assert manifest["ms_ssim_protocol"]["same_float_reconstruction_as_other_metrics"]
        identities.append(manifest["dataset_fingerprint"])
        protocols[name] = {k: v for k, v in manifest.items() if k != "summary"}
        rows = [json.loads(line) for line in (path / "images.jsonl").read_text().splitlines()]
        assert len(rows) == 144
        parts.extend([f"**{name}：{step // 1000}K**", "",
                      "| 档位 λ | bpp ↓ | PSNR ↑ | LPIPS ↓ | MS-SSIM ↑ | DISTS ↓ |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for code, point in sorted(manifest["summary"].items(), key=lambda item: int(item[0])):
            quality = int(code) / 65535
            subset = [r for r in rows if r["lambda_code"] == int(code)]
            assert len(subset) == 24 and len({r["image"] for r in subset}) == 24
            assert all(r["nfe"] == 1 for r in subset)
            for row in subset:
                assert sum(row["bytes_by_component"].values()) == row["bytes"]
                assert math.isclose(row["bpp"], 8 * row["bytes"] / (row["height"] * row["width"]), abs_tol=1e-12)
                assert all(math.isfinite(row[key]) for key in METRICS)
                assert 0 <= row["ms_ssim"] <= 1
            v = validation[str(round(quality, 1))]
            assert abs(point["bpp"] - v["rate_actual"]) < 2e-8
            for key in ("psnr", "lpips", "dists"):
                assert abs(point[key] - v[key]) < 3e-5, (name, quality, key, point[key], v[key])
            record = dict(backend=name, step=step, quality=quality,
                          **{k: point[k] for k in METRICS})
            exported.append(record)
            parts.append(f"| {quality:.1f} | {point['bpp']:.5f} | {point['psnr']:.3f} | "
                         f"{point['lpips']:.5f} | {point['ms_ssim']:.5f} | {point['dists']:.5f} |")
        parts.append("")
    assert identities[0] == identities[1]
    (ROOT / "metrics.json").write_text(json.dumps(dict(protocols=protocols, rows=exported), indent=2) + "\n")
    with (ROOT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    text = "\n".join(parts)
    (ROOT / "metrics.zh.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
