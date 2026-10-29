"""Plot measured rate coverage and paired backend outcomes; no model/GPU calls."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "bt_self_flow_v1"
OUT = ROOT / "rate_range_analysis"


def load(name, step):
    path = RUN / name / "joint" / f"validation_{step:07d}.json"
    values = json.loads(path.read_text())
    assert values["_validation_identity"]["images"] == 24
    assert values["_validation_identity"]["actual_rate"]
    assert values["_validation_identity"]["crop"] == 0
    points = [dict(quality=float(q), **m,
                   repair_gain=100 * (1 - m["latent"] / m["base"]))
              for q, m in values.items() if not q.startswith("_")]
    points.sort(key=lambda p: p["quality"])
    assert len(points) == 6
    return points, values["_validation_identity"]


def main():
    OUT.mkdir(exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path).get_name(),
        "font.size": 10.5, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 12,
    })
    specifications = [
        ("C", 34000, "#2563eb", "o", "-"),
        ("D", 34000, "#e85d04", "s", "-"),
        ("D", 36000, "#198754", "^", "--"),
    ]
    series, identities = {}, []
    for name, step, *_ in specifications:
        points, identity = load(name, step)
        series[f"{name}_{step}"] = points
        identities.append(identity)
    assert all(v == identities[0] for v in identities)
    comparison = json.loads((ROOT / "comparison.json").read_text())
    functions = {}
    for name in ("C", "D"):
        curve = comparison["curves"][name]
        rates = np.array([p["avg_bpp"] for p in curve])
        for metric in ("psnr", "lpips"):
            functions[name, metric] = PchipInterpolator(
                np.log(rates), [p[metric] for p in curve], extrapolate=False)
    lower = max(series["C_34000"][0]["rate_actual"], series["D_34000"][0]["rate_actual"])
    upper = min(series["C_34000"][-1]["rate_actual"], series["D_34000"][-1]["rate_actual"])
    rates = np.linspace(lower, upper, 300)

    def paired_values(rate):
        x = np.log(rate)
        return (functions["D", "psnr"](x) - functions["C", "psnr"](x),
                100 * (functions["C", "lpips"](x) - functions["D", "lpips"](x))
                / functions["C", "lpips"](x))

    psnr_delta, lpips_gain = paired_values(rates)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    fig.subplots_adjust(left=.08, right=.97, bottom=.13, top=.87, hspace=.55, wspace=.32)
    fig.suptitle("当前码率覆盖，以及 D 后端在哪些区间有效", fontsize=17, y=.975)
    fig.text(.5, .921, "C / D 34K 用于配对比较；D 36K 仅展示最新验证进展", ha="center", color="#555555")

    ax = axes[0, 0]
    ax.axhspan(.01, .1, color="#198754", alpha=.09)
    ax.axhline(.1, color="#198754", linestyle=":", linewidth=1.2)
    ax.axhline(.01, color="#198754", linestyle=":", linewidth=1.2)
    for name, step, color, marker, linestyle in specifications:
        pts = series[f"{name}_{step}"]
        ax.plot([p["quality"] for p in pts], [p["rate_actual"] for p in pts],
                marker=marker, color=color, linestyle=linestyle, label=f"{name} / {step//1000}K")
    ax.text(.025, .102, "目标上限 0.10", color="#166534", fontsize=9)
    ax.text(.025, .012, "目标下限 0.01", color="#166534", fontsize=9, va="bottom")
    ax.set(title="① 六档码率：当前整体偏高", xlabel="质量控制 λ（越大，量化越细）",
           ylabel="真实总 bpp", xlim=(-.02, 1.02), ylim=(0, .18))
    ax.set_yticks([0, .01, .03, .06, .10, .13, .16, .18])
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(.01, .96))

    ax = axes[0, 1]
    ax.axvspan(.01, .1, color="#198754", alpha=.09)
    ax.axvspan(.01, .0304, facecolor="#eeeeee", hatch="///", edgecolor="#bbbbbb", alpha=.45)
    ax.axhline(0, color="#555555", linewidth=1)
    for name, step, color, marker, linestyle in specifications:
        pts = series[f"{name}_{step}"]
        ax.plot([p["rate_actual"] for p in pts], [p["repair_gain"] for p in pts],
                marker=marker, color=color, linestyle=linestyle, label=f"{name} / {step//1000}K")
    ax.annotate("D 36K：+21.3%", (.03045, 21.286), xytext=(.055, 21),
                color="#166534", arrowprops={"arrowstyle": "-", "color": "#198754"}, fontsize=9)
    ax.text(.0202, -16.5, "未测", ha="center", color="#666666", fontsize=9)
    ax.set(title="② latent 修复：低码率已有正收益", xlabel="真实总 bpp",
           ylabel="修复前后 latent MSE 降幅（%）", xlim=(0, .175), ylim=(-18, 26))
    ax.legend(fontsize=9, loc="lower right")

    for ax in axes[1]:
        ax.axvspan(.01, .1, color="#198754", alpha=.09)
        ax.axvspan(.01, lower, facecolor="#eeeeee", hatch="///", edgecolor="#bbbbbb", alpha=.45)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xlim(0, .175)
        ax.set_xlabel("真实总 bpp（仅已测范围内插值）")
    axes[1, 0].plot(rates, psnr_delta, color="#e85d04", linewidth=2)
    axes[1, 0].set(title="③ 最终图像 PSNR：D 仍未稳定优于 C", ylabel="D − C 的 PSNR（dB；正值更好）")
    axes[1, 1].plot(rates, lpips_gain, color="#7c3aed", linewidth=2)
    axes[1, 1].set(title="④ 最终图像 LPIPS：低码率有感知收益", ylabel="D 相对 C 的 LPIPS 降幅（%；正值更好）")
    for rate in (.04, .06, .08):
        a, b = paired_values(rate)
        axes[1, 0].scatter(rate, a, color="#e85d04", s=25, zorder=3)
        axes[1, 1].scatter(rate, b, color="#7c3aed", s=25, zorder=3)
    axes[1, 0].annotate("0.06 bpp：−0.102 dB", (.06, float(paired_values(.06)[0])),
                        xytext=(.083, -.043), fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#e85d04"})
    axes[1, 1].annotate("0.04 bpp：+3.3%", (.04, float(paired_values(.04)[1])),
                        xytext=(.066, 3.2), fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#7c3aed"})
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.text(.08, .055, "绿色区域：目标 0.01–0.1 bpp；阴影斜线：当前缺少测量的低端。", fontsize=10, color="#444444")
    fig.text(.08, .028, "Kodak24 完整图、EMA、真实码流全部开销。下排为配对 34K 的 PCHIP 曲线；不向未测区域外推。", fontsize=9, color="#666666")
    fig.savefig(OUT / "rate_range_and_backend.png", dpi=180)
    fig.savefig(OUT / "rate_range_and_backend.pdf")
    plt.close(fig)

    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": identities[0], "target_bpp": [.01, .1], "series": series,
        "matched_rate_comparison": comparison["matched_rate_interpolation"],
        "suggested_quality_points_bpp": [.01, .02, .035, .05, .075, .1],
        "endpoint_estimates": {},
        "limitations": [
            "D 36K is not used as a paired comparison with C 34K.",
            "No measurements exist below approximately 0.03 bpp; no effectiveness extrapolation.",
            "Training lambda is uniform, but lambda=0.65 mapping to bpp=0.1 is a Kodak-average approximation, not a training-sample rate histogram.",
            "No experiments or training configurations were modified.",
        ],
    }
    packet_row = json.loads((ROOT / "D/images.jsonl").read_text().splitlines()[0])
    hw = packet_row["height"] * packet_row["width"]
    fixed = {k: packet_row["bytes_by_component"][k] for k in ("header", "allocation", "scales")}
    report["fixed_side_overhead_kodak"] = {
        "image_pixels": hw, "component_bytes": fixed,
        "bpp": {k: 8*v/hw for k, v in fixed.items()},
        "combined_bpp": 8*sum(fixed.values())/hw,
        "combined_budget_fraction_at_0.01_bpp": 8*sum(fixed.values())/hw/.01,
    }
    for label, pts in series.items():
        for a, b in zip(pts, pts[1:]):
            if a["rate_actual"] <= .1 <= b["rate_actual"]:
                q = a["quality"] + (.1-a["rate_actual"])/(b["rate_actual"]-a["rate_actual"]) * (b["quality"]-a["quality"])
                report["endpoint_estimates"][label] = {
                    "lambda_at_0.1_bpp_piecewise_linear": q,
                    "baseline_delta_at_0.1_bpp": 4*8**(-q),
                    "uniform_quality_fraction_above_threshold": 1-q,
                }
    (OUT / "rate_range_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(OUT / "rate_range_and_backend.png", flush=True)
    print(json.dumps(report["endpoint_estimates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
