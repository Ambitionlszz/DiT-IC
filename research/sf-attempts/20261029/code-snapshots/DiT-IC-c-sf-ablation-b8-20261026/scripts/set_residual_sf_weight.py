"""Explicitly continue an active C+SF checkpoint with a fixed SF coefficient."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic, sha256_file
from models.bt_self_flow.experiment import optimization_contract, training_contract
from models.bt_self_flow.residual_training import residual_weight


def with_fixed_weight(source, weight, output, provenance):
    config = deepcopy(source["config"])
    sf = config["self_flow"]
    if (sf.get("method") != "quantization_residual_v1" or not sf.get("enabled")
            or not source["raw"].get("residual", {}).get("active", False)
            or not source.get("teacher") or not source.get("residual_sf", {}).get("protocol")):
        raise ValueError("a complete active C residual-SF checkpoint is required")
    world = source["contract"]["world_size"]
    if len(source["rng_by_rank"]) != world or source.get("full_precision_retry"):
        raise ValueError("save a complete ordinary update before changing the SF weight")
    dataset = SimpleNamespace(fingerprint=source["contract"]["dataset_fingerprint"])
    if optimization_contract(source["contract"]) != training_contract(config, dataset, world):
        raise ValueError("source config and optimization contract differ")
    sf["weight"] = weight
    weight = residual_weight(config, source["rep_weight"])
    config["train"]["output"] = str(output)
    state = deepcopy(source["residual_sf"])
    state.setdefault("calibrated_alpha", state["alpha"])
    state.setdefault("weight_history", []).append(dict(
        after_step=source["step"], previous_weight=source["rep_weight"], weight=weight,
        normalization="whole-batch SF loss / steady participation probability 0.5",
        **provenance))
    state.update(alpha=weight, weight_policy="fixed")
    # Every tensor and optimizer/RNG state remains the original object. This is
    # an explicit objective change after the saved update, not initialization.
    return dict(source, config=config, contract=training_contract(config, dataset, world),
                rep_weight=weight, residual_sf=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    args = parser.parse_args()
    source_path, root = args.checkpoint.resolve(), args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("use an empty continuation root; preserve the old run")
    source = read_checkpoint(source_path)
    provenance = dict(source=str(source_path), source_sha256=sha256_file(source_path),
                      changed_at=datetime.now(timezone.utc).isoformat())
    updated = with_fixed_weight(source, args.weight, root/"joint", provenance)
    best = source.get("validation_best")
    if best and not (source_path.parent/best["checkpoint"]).is_file():
        raise ValueError("the source best checkpoint is unavailable")
    (root/"joint").mkdir(parents=True, exist_ok=True)
    (root/"configs").mkdir(exist_ok=True)
    (root/"logs").mkdir(exist_ok=True)
    if best:
        (root/"joint"/best["checkpoint"]).symlink_to(source_path.parent/best["checkpoint"])
    for path in source_path.parent.glob("validation_*.json"):
        if int(path.stem.split("_")[-1]) <= source["step"]:
            shutil.copyfile(path, root/"joint"/path.name)
    if (source_path.parent/"train.jsonl").exists():
        shutil.copyfile(source_path.parent/"train.jsonl", root/"joint/train.jsonl")
    (root/"configs/joint.json").write_text(json.dumps(updated["config"], indent=2)+"\n")
    (root/"joint/sf_transition.json").write_text(json.dumps(updated["residual_sf"], indent=2)+"\n")
    report = dict(**provenance, step=source["step"], first_fixed_weight_update=source["step"]+1,
                  previous_weight=source["rep_weight"], weight=updated["rep_weight"],
                  config=str(root/"configs/joint.json"), checkpoint=str(root/"joint/paused.pt"),
                  normalization="sf_weight multiplies whole-batch sf_loss / 0.5",
                  state_policy="preserve raw/EMA/teacher, main and auxiliary optimizer, scaler, scheduler, rank RNG, data counter and validation best")
    save_atomic(root/"joint/paused.pt", updated)
    (root/"weight_change.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
