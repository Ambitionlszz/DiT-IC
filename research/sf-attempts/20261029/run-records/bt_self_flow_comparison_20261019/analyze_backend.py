"""Analyze existing BT logs and packet evaluation; never load a GPU/model.

Run with python3 for JSON/CSV; add --plot in the experiment Python environment
for the standalone PNG/PDF. Sources are existing EMA Kodak evaluations.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "bt_self_flow_v1"
OUT = ROOT / "backend_diagnosis"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    calibration = read(RUN / "common/calibration.json")
    s2 = calibration["s_data"] ** 2
    data = {
        "analyzed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": "Kodak24 full images, deployment EMA, paired 34000 updates, 1 NFE",
        "sources": {
            "per_image_packet_evaluation": str(ROOT / "{C,D}/images.jsonl"),
            "final_latent_mse": str(RUN / "{C,D}/joint/validation_0034000.json"),
            "calibration": str(RUN / "common/calibration.json"),
        },
        "energy_identity": "F = B + U - 2 dot, r=mu-z0, d=mu-z_final, U=mean(d^2)",
        "energy_caveat": "cosine is an aggregate over all images/elements, not mean image cosine; optimal alpha minimizes frozen latent MSE only, not the image/perceptual objective",
        "calibration": calibration,
        "paired_points": [],
        "history": {},
        "latest_appended_train_log": {},
        "nonfinite_diagnostics": {},
    }
    identities = []
    for name in ("C", "D"):
        path = RUN / name / "joint"
        manifest = read(ROOT / name / "summary.json")
        assert manifest["step"] == 34000 and manifest["weights"] == "ema"
        assert manifest["images"] == 24 and manifest["nfe"] == 1
        rows = [json.loads(line) for line in (ROOT / name / "images.jsonl").read_text().splitlines()]
        validation = read(path / "validation_0034000.json")
        identities.append(validation["_validation_identity"])
        for quality in (0., .2, .4, .6, .8, 1.):
            subset = [r for r in rows if abs(r["lambda"] - quality) < 1e-4]
            assert len(subset) == 24 and len({r["image"] for r in subset}) == 24
            assert len({(r["height"] * r["width"]) for r in subset}) == 1
            assert all(sum(r["bytes_by_component"].values()) == r["bytes"] for r in subset)
            v = validation[str(quality)]
            base = sum(r["base_latent_mse"] for r in subset) / len(subset)
            correction = sum(r["repair_latent_rms"] ** 2 for r in subset) / len(subset)
            final = v["latent"] * s2
            assert abs(base - v["base"] * s2) < 2e-8
            for key in ("psnr", "lpips", "dists"):
                assert abs(sum(r[key] for r in subset) / len(subset) - v[key]) < 2e-5
            dot = (base + correction - final) / 2
            point = dict(
                backend=name, quality=quality, bpp=v["rate_actual"],
                psnr=v["psnr"], lpips=v["lpips"], dists=v["dists"],
                base_nmse=v["base"], final_nmse=v["latent"],
                latent_reduction_percent=100 * (1 - final / base),
                correction_rms=math.sqrt(correction),
                correction_relative_rms=math.sqrt(correction / base),
                correction_target_cosine=dot / math.sqrt(base * correction),
                optimal_latent_alpha=dot / correction,
                raw_base_energy=base, raw_final_energy=final,
                correction_energy=correction, target_correction_inner_product=dot,
                scale_saturation=v.get("scale_saturation", 0.),
            )
            data["paired_points"].append(point)
        history = []
        for f in sorted(path.glob("validation_*.json")):
            v = read(f)
            assert v["_validation_identity"] == identities[-1]
            history.append(dict(
                step=int(f.stem.split("_")[-1]),
                points=[dict(quality=float(q), **m,
                             latent_reduction_percent=100 * (1 - m["latent"] / m["base"]))
                        for q, m in v.items() if not q.startswith("_")],
            ))
        data["history"][name] = history
        # Reboot/resume can leave older, higher-step rows before resumed rows.
        # The current branch is identified by the final appended row, not max(step).
        logs = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
        data["latest_appended_train_log"][name] = logs[-1]
        bads = [read(f) for f in sorted(path.glob("nonfinite_*.json"))]
        ranks = [r for bad in bads for r in bad["ranks"]]
        data["nonfinite_diagnostics"][name] = dict(
            saved_files=len(bads), last_diagnostic_step=max((r["step"] for r in ranks), default=None),
            all_saved_model_states_finite=all(r.get("model_state_finite", False) for r in ranks),
            extra_optimizer_attempts=logs[-1]["attempts"] - logs[-1]["step"],
            caveat="Extra attempts include AMP skips; saved files are not a count of all skipped attempts.",
        )
    assert identities[0] == identities[1]
    data["validation_identity"] = identities[0]
    common = read(RUN / "common/validation_0050000.json")
    assert common["_validation_identity"] == identities[0]
    data["common_frontend_kodak_50k"] = common
    data["matched_rate_image_comparison"] = read(ROOT / "comparison.json")["matched_rate_interpolation"]
    (OUT / "diagnosis.json").write_text(json.dumps(data, indent=2) + "\n")
    with (OUT / "paired_34000.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["paired_points"][0]))
        writer.writeheader()
        writer.writerows(data["paired_points"])
    print(json.dumps(dict(output=str(OUT), paired_points=len(data["paired_points"]),
                          history_points={k: len(v) for k, v in data["history"].items()},
                          validation_identity=identities[0]), indent=2))
    if args.plot:
        plot(data)


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"C": "#2563eb", "D": "#e85d04"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in ("C", "D"):
        history = data["history"][name]
        for ax, q, title in ((axes[0, 0], 0., "Lowest quality setting"),
                             (axes[0, 1], 1., "Highest quality setting")):
            values = [next(p for p in h["points"] if p["quality"] == q) for h in history]
            ax.plot([h["step"] / 1000 for h in history],
                    [v["latent_reduction_percent"] for v in values],
                    marker=".", color=colors[name], label=name)
            ax.set(title=title, xlabel="Joint training updates (K)", ylabel="Latent MSE reduction (%)")
        points = [p for p in data["paired_points"] if p["backend"] == name]
        axes[1, 0].plot([p["quality"] for p in points], [p["correction_target_cosine"] for p in points],
                        marker="o", color=colors[name], label=name)
        axes[1, 1].plot([p["quality"] for p in points], [p["correction_rms"] for p in points],
                        marker="o", color=colors[name], label=name)
    axes[1, 0].set(title="Correction alignment at paired 34K", xlabel="Quality control (higher = finer)",
                    ylabel="Aggregate cosine with target residual")
    axes[1, 1].set(title="Correction amplitude at paired 34K", xlabel="Quality control (higher = finer)",
                    ylabel="RMS of latent correction")
    for ax in axes.flat:
        ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("BT backend diagnosis | Kodak24, EMA, full images, 1 NFE\n"
                 "History uses fixed quality controls; bpp changes during training", fontsize=13)
    fig.savefig(OUT / "backend_diagnosis.png", dpi=180)
    fig.savefig(OUT / "backend_diagnosis.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
