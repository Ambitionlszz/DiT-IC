"""Resolve the batch-8 C/SF comparison without launching or migrating training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.experiment import load_config
from scripts.launch_quantization_residual import check_effective_batch


CONFIGS = ROOT/"configs/bt_self_flow/c_sf_ablation_b8"


def paired_configs():
    configs = {name: load_config(CONFIGS/f"{name}.json") for name in ("pure_c", "c_sf")}
    normalized = []
    for config in configs.values():
        check_effective_batch(config, 2)
        normalized.append(deepcopy(config))
        normalized[-1].pop("self_flow")
        normalized[-1]["train"].pop("output")
    if normalized[0] != normalized[1]:
        raise ValueError("C and C+SF settings differ outside Self-Flow/output")
    return configs


def audit_source(path, target):
    """Check a supported checkpoint's metadata; this does not convert state."""
    path = Path(path).resolve()
    source = read_checkpoint(path)
    config = source.get("config", {})
    train = config.get("train", {})
    world = source.get("contract", {}).get("world_size")
    errors = []
    if source.get("phase") != "joint" or source.get("backend") != "C" or source.get("step") != 10000:
        errors.append("expected a pure-C joint checkpoint at optimizer step 10000")
    sf = config.get("self_flow", {})
    if sf.get("enabled") or sf.get("method") == "quantization_residual_v1" or "residual" in source.get("raw", {}):
        errors.append("source must be the original pure C, without a residual head/teacher")
    batch = train.get("batch_size", 0)*train.get("accumulation_steps", 0)*(world or 0)
    if batch != 8 or train.get("effective_batch") != 8:
        errors.append("source metadata must confirm effective batch 8")
    required = ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "scaler", "rng_by_rank",
                "micro_seen", "attempts", "assets", "contract")
    missing = [name for name in required if name not in source]
    if missing:
        errors.append("missing continuation state: "+", ".join(missing))
    for name in ("model", "data", "optimizer", "image_loss", "objective", "quality"):
        actual, expected = deepcopy(config.get(name, {})), deepcopy(target[name])
        # Files may move between hosts; their contents/fingerprints must be
        # verified during migration. AMP is intentionally changed on V100.
        for key in {"model": ("sana_path", "text_path"), "data": ("train", "valid", "workers")}.get(name, ()):
            actual.pop(key, None)
            expected.pop(key, None)
        if name == "quality":
            actual.setdefault("sampling", "independent")
        if actual != expected:
            errors.append(f"source {name} settings differ; reconcile with embedded C config before migration")
    for name in ("phase", "steps", "seed", "router_uniform_steps", "router_ramp_steps"):
        if train.get(name) != target["train"].get(name):
            errors.append(f"source train.{name} differs from the prepared C recipe")
    return dict(path=str(path), sha256=sha256_file(path), step=source.get("step"),
                world_size=world, batch_size=train.get("batch_size"),
                accumulation_steps=train.get("accumulation_steps"), effective_batch=batch,
                source_amp=train.get("amp_dtype", "auto (confirm source hardware/runtime)"),
                micro_seen=source.get("micro_seen"), attempts=source.get("attempts"),
                metadata_errors=errors, converted=False)


def prepare(directory, checkpoint=None):
    configs = paired_configs()
    source = audit_source(checkpoint, configs["pure_c"]) if checkpoint else None
    git = ["git", "-c", f"safe.directory={ROOT}"]
    plan = dict(
        status=("waiting_for_checkpoint" if source is None else
                "source_metadata_mismatch" if source["metadata_errors"] else "awaiting_state_migration"),
        launch_ready=False, repository=str(ROOT),
        commit=subprocess.check_output(git+["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(git+["status", "--porcelain"], cwd=ROOT, text=True).strip()),
        source=source, start_step=10000, total_step_target=200000,
        shared=dict(gpus=[0, 1], local_batch=4, accumulation=1, effective_batch=8, crop=512,
                    quality_sampling="independent; identical step-seeded draw on both ranks/branches",
                    amp="FP16 + GradScaler; common FP32 numerical fixes",
                    image_loss=configs["pure_c"]["image_loss"], objective=configs["pure_c"]["objective"],
                    optimizer=configs["pure_c"]["optimizer"],
                    periodic_save_every=10000, log_every=100, eval_every=2000),
        sf=dict(activation_after_saved_step=10000, first_sf_update=10001, full_weight_step=12000,
                settings=configs["c_sf"]["self_flow"],
                calibration="recompute on the uploaded 10K checkpoint; do not reuse the batch-16 alpha"),
        pending=[
            "Verify embedded C config, raw/EMA tensors, dataset fingerprint and pretrained-asset hashes.",
            "Preserve Adam/auxiliary moments and scheduler; map parameters by name and add only SF head state.",
            "Preserve global data counter; map one-rank RNG to a documented common two-rank policy.",
            "Create a fresh FP16 GradScaler consistently for both branches; source BF16 scaler is empty.",
            "Verify identical C outputs/loss at step 10000; calibrate SF and smoke-test the actual resumed update.",
        ],
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        (directory/f"{name}.resolved.json").write_text(json.dumps(config, indent=2)+"\n")
    (directory/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Preparation artifacts, not a training directory")
    parser.add_argument("--checkpoint", type=Path, help="Optional uploaded 10K pure-C checkpoint to inspect")
    args = parser.parse_args()
    plan = prepare(args.output, args.checkpoint)
    print(json.dumps(dict(status=plan["status"], launch_ready=False, output=str(args.output.resolve()))))
    if plan["source"] and plan["source"]["metadata_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
