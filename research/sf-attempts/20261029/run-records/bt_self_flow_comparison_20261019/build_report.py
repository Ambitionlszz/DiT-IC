"""Validate real-packet evaluations and compare only overlapping rate ranges."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
METRICS = ("psnr", "lpips", "dists")
LABELS = {"baseline_30k": "Baseline 512 / 30K", "C": "C / 34K", "D": "D / 34K",
          "historical_200k": "Baseline 512 / 200K (archived)"}
COLORS = {"baseline_30k": "#6c757d", "C": "#2563eb", "D": "#e85d04", "historical_200k": "#222222"}


def read_json(path):
    return json.loads(path.read_text())


def main():
    history = read_json(ROOT / "historical_reference.json")
    baseline = read_json(ROOT / "baseline/all_summaries.json")["global"]
    assert len(baseline) == 6 and all(x["num_images"] == 24 for x in baseline)
    curves = {"baseline_30k": baseline,
              "historical_200k": history["reference_curves"]["A：非窗口训练 / 全局"]}
    records = {}
    for group in ("C", "D"):
        manifest = read_json(ROOT / group / "summary.json")
        assert manifest["step"] == 34000 and manifest["images"] == 24
        assert manifest["weights"] == "ema" and manifest["nfe"] == 1
        rows = [json.loads(s) for s in (ROOT / group / "images.jsonl").read_text().splitlines()]
        assert len(rows) == 144
        curves[group] = []
        records[group] = {}
        for code, values in manifest["summary"].items():
            subset = [r for r in rows if str(r["lambda_code"]) == code]
            assert len(subset) == 24 and len({Path(r["image"]).name for r in subset}) == 24
            for r in subset:
                assert sum(r["bytes_by_component"].values()) == r["bytes"]
                assert r["nfe"] == 1 and all(math.isfinite(r[k]) for k in ("bpp", *METRICS))
                stem = Path(r["image"]).stem
                packet = next((ROOT / group / f"q{int(code):05d}").glob(f"*_{stem}.btic"))
                assert packet.stat().st_size == r["bytes"]
                assert math.isclose(r["bpp"], packet.stat().st_size * 8 / (r["height"] * r["width"]), abs_tol=1e-12)
                r["png"] = str(packet.with_suffix(".png"))
                records[group].setdefault(stem, []).append(r)
            curves[group].append(dict(values, avg_bpp=values["bpp"], quality_index=int(code)/65535, num_images=24))
    records["baseline_30k"] = {}
    for q in range(6):
        path = ROOT / f"baseline/global/q{q}/metrics/realrans/per_image_metrics.json"
        rows = read_json(path)
        assert len(rows) == 24
        for r in rows:
            packet = ROOT / f"baseline/packets/q{q}/bin" / r["image_name"]
            assert packet.stat().st_size == r["stream_bytes"]
            assert math.isclose(r["bpp"], packet.stat().st_size * 8 / (r["height"] * r["width"]), abs_tol=1e-12)
            r["png"] = str(ROOT / f"baseline/global/q{q}/rec" / f"{r['image_name']}.png")
            records["baseline_30k"].setdefault(r["image_name"], []).append(r)
    matched_visuals = {group: read_json(ROOT / "matched_visuals" / group / "summary.json") for group in ("C", "D")}
    maximum_visual_rate_error = max(v["relative_rate_error"] for rows in matched_visuals.values() for v in rows.values())
    for group, rows in matched_visuals.items():
        assert len(rows) == 3
        for stem, row in rows.items():
            reference = next(r for r in records["baseline_30k"][stem] if r["quality_index"] == 2)
            assert row["target_bpp"] == reference["bpp"] and row["nfe"] == 1
            packet = ROOT / "matched_visuals" / group / f"{stem}.btic"
            assert math.isclose(row["bpp"], packet.stat().st_size*8/(reference["height"]*reference["width"]), abs_tol=1e-12)
    for curve in curves.values():
        curve.sort(key=lambda r: r["avg_bpp"])
        assert np.all(np.diff([r["avg_bpp"] for r in curve]) > 0)
    with (ROOT / "rd_points.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "quality_index", "avg_bpp", *METRICS])
        writer.writeheader()
        for name, curve in curves.items():
            for row in curve:
                writer.writerow(dict(model=name, **{k: row[k] for k in writer.fieldnames[1:]}))
    spec = importlib.util.spec_from_file_location("old_comparison", "/data2/ls/code/deployment-bootstrap/compare_baseline512_history.py")
    comparison_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison_module)
    comparisons = {}
    for reference, candidate in (("C", "D"), ("baseline_30k", "C"), ("baseline_30k", "D"),
                                 ("historical_200k", "C"), ("historical_200k", "D")):
        comparisons[f"{candidate}_vs_{reference}"] = {
            metric: comparison_module.bd_rate(curves[reference], curves[candidate], metric) for metric in METRICS}
    shared = {}
    for target in (.04, .06, .08):
        shared[str(target)] = {}
        for name, curve in curves.items():
            if curve[0]["avg_bpp"] <= target <= curve[-1]["avg_bpp"]:
                shared[str(target)][name] = {
                    metric: float(PchipInterpolator(np.log([r["avg_bpp"] for r in curve]),
                                                     [r[metric] for r in curve])(np.log(target)))
                    for metric in METRICS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, METRICS):
        for name in ("baseline_30k", "C", "D", "historical_200k"):
            curve = curves[name]
            ax.plot([r["avg_bpp"] for r in curve], [r[metric] for r in curve],
                    marker="o", linestyle="--" if name == "historical_200k" else "-",
                    color=COLORS[name], label=LABELS[name], markersize=4)
        ax.set_xlabel("Real total bitrate (bpp)")
        ax.set_ylabel("PSNR (dB), higher is better" if metric == "psnr" else f"{metric.upper()}, lower is better")
        ax.grid(alpha=.22)
        ax.set_xlim(.015, .17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Kodak24, full images, FP32, EMA, global 1-NFE decoding; all packet bytes billed")
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(ROOT / "rd_curves.png", dpi=170)
    fig.savefig(ROOT / "rd_curves.pdf")
    plt.close(fig)
    previews = {}
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    names = ("kodim01", "kodim13", "kodim23")
    for detail in (False, True):
        cell_w, cell_h, label_h = (384, 288, 50) if not detail else (300, 300, 50)
        canvas = Image.new("RGB", (4*cell_w, len(names)*(cell_h+label_h)+40), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"Per-image rates matched to baseline q2; maximum mismatch {maximum_visual_rate_error*100:.2f}%", fill="black", font=font)
        for row_index, stem in enumerate(names):
            original = Image.open(Path("/data2/ls/data/Kodak") / f"{stem}.png").convert("RGB")
            previews.setdefault(stem, {})
            selected = []
            for group in ("baseline_30k", "C", "D"):
                r = (next(r for r in records[group][stem] if r["quality_index"] == 2)
                     if group == "baseline_30k" else matched_visuals[group][stem])
                previews[stem][group] = {k: r[k] for k in ("bpp", *METRICS, "png")}
                selected.append((Image.open(r["png"]).convert("RGB"), f"{LABELS[group]}  {r['bpp']:.4f} bpp", f"PSNR {r['psnr']:.2f}  LPIPS {r['lpips']:.3f}"))
            cells = [(original, f"Original / {stem}", "")] + selected
            for column, (img, title, subtitle) in enumerate(cells):
                if detail:
                    x, y = int(original.width*.32), int(original.height*.24)
                    img = img.crop((x, y, x+256, y+256))
                img = ImageOps.contain(img, (cell_w-8, cell_h-8))
                top = 40+row_index*(cell_h+label_h)
                canvas.paste(img, (column*cell_w+(cell_w-img.width)//2, top+(cell_h-img.height)//2))
                draw.text((column*cell_w+8, top+cell_h), title, fill="black", font=font)
                draw.text((column*cell_w+8, top+cell_h+23), subtitle, fill="black", font=font)
        canvas.save(ROOT / ("detail_crops.png" if detail else "reconstruction_preview.png"))
    report = {"curves": curves, "bd_rate": comparisons, "matched_rate_interpolation": shared,
              "interpolation": "PCHIP metric versus log real bpp, within observed range only; these are interpolated curve values, not new operating points",
              "previews": previews, "validated_reconstruction_count": 432,
              "additional_visual_decodes": 6, "maximum_visual_relative_rate_error": maximum_visual_rate_error,
              "historical_source_hash_matches": 24,
              "limitations": ["C/D 34K versus local baseline 30K; historical baseline 200K is archived results, its weights are unavailable",
                              "C/D train image loss 2/1/0.5; baseline image loss 4/1/1; frontends and initialization differ",
                              f"Preview rates are matched per image to baseline q2; maximum residual mismatch {maximum_visual_rate_error*100:.3f}%",
                              "Kodak is used for checkpoint selection in these runs; it is not a held-out test set"]}
    (ROOT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# C/D 与 512 baseline：Kodak24 真实码流比较", "",
             "C/D 均为 34K deployment EMA；本机 baseline 为 30K EMA；历史 200K baseline 使用原始归档曲线，未重新推理。",
             "原始整图、FP32、全局单步解码。bpp 包括码流头和全部侧信息，以原图像素计费。", "",
             "## 相同真实码率下的曲线插值", "",
             "以下数值由六档曲线在 log(bpp) 上做 PCHIP 插值，只使用共同覆盖范围；不是新增实测档位。", "",
             "| bpp | 模型 | PSNR ↑ | LPIPS ↓ | DISTS ↓ |", "|---:|---|---:|---:|---:|"]
    for rate, models in shared.items():
        for name in ("baseline_30k", "C", "D", "historical_200k"):
            if name in models:
                r = models[name]
                lines.append(f"| {rate} | {LABELS[name]} | {r['psnr']:.3f} | {r['lpips']:.5f} | {r['dists']:.5f} |")
    lines += ["", "## BD-rate", "", "负值表示相同质量所需比特更少；共同质量区间积分，不外推、不修复非单调曲线。", "",
              "| 比较 | PSNR | LPIPS | DISTS |", "|---|---:|---:|---:|"]
    for name, values in comparisons.items():
        rendered = [f"{values[m]['percent']:+.2f}%" if values[m]['status'] == 'computed' else 'N/A' for m in METRICS]
        lines.append("| " + " | ".join([name, *rendered]) + " |")
    cd = comparisons["D_vs_C"]
    if all(cd[m]["status"] == "computed" for m in METRICS):
        lines += ["", "## 当前判断", "",
                  f"相同 34K 下，D 相对 C 的 PSNR BD-rate 为 {cd['psnr']['percent']:+.2f}%，"
                  f"LPIPS 为 {cd['lpips']['percent']:+.2f}%，DISTS 为 {cd['dists']['percent']:+.2f}%。",
                  "当前表现是感知质量与像素保真的取舍，尚不能称为全范围优于 C。训练仍处于 200K 预算的早期。"]
    lines += ["", "## 解释边界", "", *[f"- {s}" for s in report["limitations"]], "",
              "全部 432 个六档推理重建通过有限值、实际文件字节计费和逐图数量核验；额外 6 次重建用于匹配逐图码率的视觉对照。C/D 每张仅一次 SANA 前向。", "",
              "![RD curves](rd_curves.png)", "", "![Reconstructions](reconstruction_preview.png)", "",
              "完整逐图记录、包、PNG、CSV、插值和 BD-rate 区间见本目录。"]
    (ROOT / "report.zh.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"bd_rate": comparisons, "matched_rate_interpolation": shared}, indent=2))


if __name__ == "__main__":
    main()
