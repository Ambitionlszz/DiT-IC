"""Explicitly continue a pre-SF joint checkpoint with batch 2 -> 4 per rank."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic, sha256_file
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, training_contract


def migrate_batch(state, config, contract):
    old = state["contract"]
    old_train, new_train = old["train"], contract["train"]
    if (state["phase"] != "joint" or state["backend"] != "D"
            or old["world_size"] != 2 or contract["world_size"] != 2
            or old_train["batch_size"] != 2 or new_train["batch_size"] != 4
            or old_train["accumulation_steps"] != 2 or new_train["accumulation_steps"] != 2
            or old_train["effective_batch"] != 8 or new_train["effective_batch"] != 16):
        raise ValueError("only the explicit two-rank 2/2 -> 4/2 migration is supported")
    if state["step"] >= config["self_flow"]["start_step"] or "teacher" in state:
        raise ValueError("this migration is restricted to the RD-only prefix")
    expected = deepcopy(old)
    expected["train"].update(batch_size=4, effective_batch=16)
    if expected != contract:
        raise ValueError("migration changes settings other than batch/effective batch")
    # CounterSampler's global position is micro_seen * world * local_batch.
    # Preserve it when regrouping images into larger rank-local batches.
    numerator = state["micro_seen"] * 2
    if numerator % 4:
        raise ValueError("data cursor is not aligned with the new batch size")
    migrated = dict(state)
    migrated.update(config=deepcopy(config), contract=deepcopy(contract), micro_seen=numerator // 4)
    report = dict(step=state["step"], world_size=2, old_batch_size=2, new_batch_size=4,
                  accumulation_steps=2, old_effective_batch=8, new_effective_batch=16,
                  old_micro_seen=state["micro_seen"], new_micro_seen=migrated["micro_seen"],
                  global_images_seen=state["micro_seen"]*2*2,
                  optimizer_scheduler_scaler_rng_weights_preserved=True,
                  self_flow_start_step=config["self_flow"]["start_step"])
    migrated["batch_migration"] = report
    return migrated, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root/"joint/paused.pt"
    if target.exists():
        raise ValueError("migration target already exists")
    config = load_config(root/"configs/joint.yaml")
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    contract = training_contract(config, dataset, 2)
    source = read_checkpoint(args.source)
    if source.get("validation_best") is not None:
        raise ValueError("migrate selected best checkpoint files explicitly before using this tool")
    migrated, report = migrate_batch(source, config, contract)
    report.update(source_checkpoint=str(args.source.resolve()), source_sha256=sha256_file(args.source),
                  migrated_checkpoint=str(target))
    save_atomic(target, migrated)
    (root/"batch_migration.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
