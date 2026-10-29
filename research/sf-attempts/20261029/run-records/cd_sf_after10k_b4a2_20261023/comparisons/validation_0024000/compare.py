"""Compare pinned Kodak24 curves without using GPUs or changing training."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
OLD = Path("/data2/ls/checkpoints/bt_self_flow_stopped_eval_20261022/comparison.json")
CURRENT = RUN / "joint/validation_0024000.json"
PREFIX = RUN / "joint/validation_0010000.json"
METRICS = ("psnr", "lpips", "dists")
RATES = (.04, .06, .08)
NAMES = {
    "baseline": "512 baseline / 200K",
    "old_c": "Previous C / 95,542",
    "old_cd": "Previous C+D / 96,750",
    "current": "C+D+SF / 24K",
    "prefix": "C+D before SF / 10K",
}
ZH = {
    "baseline": "512 baseline（200K）",
    "old_c": "旧 C（95,542）",
    "old_cd": "旧 C+D（96,750）",
    "current": "当前 C+D+SF（24K）",
    "prefix": "本轮接入 SF 前（10K）",
}
COLORS = {
    "baseline": "#202020", "old_c": "#377eb8", "old_cd": "#e58920",
    "current": "#c52b39", "prefix": "#929292",
}


def read_validation(path):
    report = json.loads(path.read_text())
    identity = report["_validation_identity"]
    assert identity["images"] == 24 and identity["crop"] == 0
    assert identity["actual_rate"] is True
    points = [{"avg_bpp": row["rate_actual"], **{m: row[m] for m in METRICS}}
              for q, row in report.items() if not q.startswith("_")]
    assert len(points) == 6
    return points, identity


def interpolate(points, metric, rate):
    points = sorted(points, key=lambda row: row["avg_bpp"])
    rates = np.array([row["avg_bpp"] for row in points])
    assert rates[0] <= rate <= rates[-1], "Extrapolation is forbidden"
    return float(PchipInterpolator(np.log(rates), [r[metric] for r in points],
                                   extrapolate=False)(np.log(rate)))


def main():
    old = json.loads(OLD.read_text())
    current, identity = read_validation(CURRENT)
    prefix, prefix_identity = read_validation(PREFIX)
    assert identity == prefix_identity
    assert old["steps"] == {"C": 95542, "D": 96750}
    curves = {
        "baseline": old["curves"]["Baseline 512 / 200K"],
        "old_c": old["curves"]["C"],
        "old_cd": old["curves"]["D"],
        "current": current,
        "prefix": prefix,
    }
    for points in curves.values():
        rates = np.array([row["avg_bpp"] for row in points])
        assert len(points) == 6 and np.all(np.diff(rates) > 0)
        assert np.isfinite([[r["avg_bpp"], *[r[m] for m in METRICS]] for r in points]).all()

    # Use the same vetted integral and direction conventions as the archived report.
    bd_path = Path("/data2/ls/code/deployment-bootstrap/compare_baseline512_history.py")
    spec = importlib.util.spec_from_file_location("archived_bd", bd_path)
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    matched = {str(rate): {key: {m: interpolate(points, m, rate) for m in METRICS}
                          for key, points in curves.items()} for rate in RATES}
    comparisons = {f"current_vs_{key}": {m: bd.bd_rate(points, current, m) for m in METRICS}
                   for key, points in curves.items() if key != "current"}
    sources = [OLD, CURRENT, PREFIX, bd_path]
    report = {
        "labels": NAMES,
        "archive_mapping": {"C": "old_c", "D": "old_cd"},
        "steps": {"baseline": 200000, "old_c": 95542, "old_cd": 96750,
                  "current": 24000, "prefix": 10000},
        "sf_first_update": 10001,
        "current_validation_identity": identity,
        "interpolation": "PCHIP in log actual total packet bpp, no extrapolation",
        "bd_rate_sign": "Positive means current requires more bitrate at equal quality",
        "curves": curves, "matched_rate": matched, "bd_rate": comparisons,
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits": [
            "Training budgets, loss weights and effective batches differ; this is checkpoint performance, not a controlled SF ablation.",
            "Historical C/D use 2*MSE+LPIPS+0.5*DISTS plus 0.1 base and 0.1 repaired; current uses 4*MSE+LPIPS+DISTS plus 0.1 base, no repaired.",
            "Baseline is the archived non-window 512-trained 200K EMA evaluated with global decoding; it was not rerun.",
            "Historical C/D come from independent full-packet decoding; current validation encodes all 24 images at six qualities and checks decoded versus hard-forward equality on two images per quality.",
            "Current validation uses upstream LPIPS and MSE-derived RGB PSNR, historical packet evaluation uses torchmetrics LPIPS and pyiqa PSNR. Both use Alex LPIPS, RGB float reconstruction and pyiqa DISTS; this is not a newly unified evaluator run.",
            "Current versus prefix includes 14000 additional optimization steps and local mixed-quality training, as well as SF alignment.",
            "BD-rate comparisons integrate different common quality intervals; inspect intervals in JSON and use matched-rate rows to assess differences at a fixed bitrate.",
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows = [{"bpp": rate, "model": ZH[key], "step": report["steps"][key], **values}
            for rate in RATES for key, values in matched[str(rate)].items()]
    with (OUT / "matched_rate.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=("bpp", "model", "step", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Kodak24：512 baseline、旧 C、旧 C+D 与当前 Self-Flow", "",
             "当前使用 24K EMA：10K 纯 RD checkpoint 已保存，从 10,001 步开始 SF，12K 时完成 2K 权重渐增；24K 时已进行 14K 次 SF 更新。", "",
             "全部为 Kodak24 原尺寸、六档、全局解码。按真实总码率的 log bpp 做 PCHIP 同码率插值，不外推。表内精确 bpp 是插值，非新增实测点。", "",
             "| bpp | 模型 / 步数 | PSNR ↑ | LPIPS ↓ | DISTS ↓ |",
             "|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['bpp']:.2f} | {row['model']} | {row['psnr']:.3f} | {row['lpips']:.5f} | {row['dists']:.5f} |")
    lines += ["", "## 当前 24K 相对各参考的 BD-rate", "",
              "正值表示达到同质量需要更多码率，负值表示节省码率；各指标共同质量区间见 comparison.json。", "",
              "| 参考 | PSNR | LPIPS | DISTS |", "|---|---:|---:|---:|"]
    for key in ("baseline", "old_c", "old_cd", "prefix"):
        results = comparisons[f"current_vs_{key}"]
        values = [f"{results[m]['percent']:+.2f}%" if results[m]["status"] == "computed" else "N/A" for m in METRICS]
        lines.append("| " + " | ".join([ZH[key], *values]) + " |")
    lines += ["", "## 比较边界", "",
              "旧 C / 旧 C+D 分别对应归档 C / D；512 baseline 采用非窗口训练、全局解码的历史 200K EMA 曲线。", "",
              "旧 C/C+D 的训练损失是 2/1/0.5，包含 0.1 base、0.1 repaired；本轮是 4/1/1、0.1 base、不含 repaired。训练预算、有效 batch 也不同，不能把模型差异全部归因于 SF。", "",
              "当前使用训练内 FP32 EMA 验证：24 张 × 6 档实际编码计费，画质来自硬前向，每档另抽查两张实际包解码且误差为零。旧 C/C+D 来自独立完整包解码评测，baseline 为历史归档，没有新增推理。", "",
              "当前 PSNR/LPIPS 分别由 MSE / 原版 lpips 计算，历史独立评测使用 pyiqa / torchmetrics；均为 RGB 浮点、Alex LPIPS、pyiqa DISTS，末位差异不宜过度解读。", "",
              "10K → 24K 同时包含多训 14K 和混合精度采样的影响，不能当作相同预算的无 SF 对照。", ""]
    (OUT / "report.zh.md").write_text("\n".join(lines))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
    labels = {"psnr": "PSNR (dB), higher is better", "lpips": "LPIPS, lower is better",
              "dists": "DISTS, lower is better"}
    common_low = max(min(r["avg_bpp"] for r in points) for points in curves.values())
    common_high = min(max(r["avg_bpp"] for r in points) for points in curves.values())
    grid = np.linspace(common_low, common_high, 250)
    for ax, metric in zip(axes, METRICS):
        for key, points in curves.items():
            values = [interpolate(points, metric, rate) for rate in grid]
            ax.plot(grid, values, color=COLORS[key], label=NAMES[key],
                    lw=2.2 if key == "current" else 1.7,
                    ls="--" if key == "prefix" else "-")
            visible = [r for r in points if common_low <= r["avg_bpp"] <= common_high]
            ax.scatter([r["avg_bpp"] for r in visible], [r[metric] for r in visible],
                       color=COLORS[key], s=20, zorder=3)
        for rate in RATES:
            ax.axvline(rate, lw=.65, color="#d6d6d6", ls=":", zorder=0)
        ax.set(xlabel="Actual total packet bitrate (bpp)", ylabel=labels[metric],
               xlim=(common_low, common_high))
        ax.grid(alpha=.18)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Kodak24 full-image EMA | matched bitrate comparison", fontsize=13)
    fig.text(.5, .895, "Current checkpoint: 24K (SF since 10,001) | Different training budgets and objectives",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.06, right=.99, top=.82, bottom=.19, wspace=.3)
    fig.savefig(OUT / "rd_comparison.png", dpi=180)
    fig.savefig(OUT / "rd_comparison.svg")
    plt.close(fig)
    print("\n".join(lines[:23]))
    print("BD_RATE", json.dumps(comparisons, ensure_ascii=False))


if __name__ == "__main__":
    main()
