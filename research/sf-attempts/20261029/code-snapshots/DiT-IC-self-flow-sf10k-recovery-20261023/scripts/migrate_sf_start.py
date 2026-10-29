"""Move the delayed SF boundary to 10K without resetting training state."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import (read_checkpoint, save_atomic, sha256_file,
                                          publish_checkpoint_alias)
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, training_contract


def migrate_start(state, config, contract):
    old = state["contract"]
    if (state["phase"] != "joint" or state["backend"] != "D"
            or old["self_flow"]["start_step"] != 20000
            or contract["self_flow"]["start_step"] != 10000):
        raise ValueError("only the requested joint D 20K -> 10K SF boundary is supported")
    if state["step"] > 10000 or "teacher" in state or state.get("teacher_initialized_step") is not None:
        raise ValueError("a pure-RD checkpoint at or before 10K is required")
    expected = deepcopy(old)
    expected["self_flow"]["start_step"] = 10000
    if expected != contract:
        raise ValueError("settings other than the Self-Flow boundary differ")
    migrated = dict(state)
    migrated.update(config=deepcopy(config), contract=deepcopy(contract))
    report = dict(step=state["step"], old_start_step=20000, new_start_step=10000,
                  first_sf_update=10001, micro_seen=state["micro_seen"],
                  optimizer_scheduler_scaler_rng_weights_preserved=True)
    migrated["self_flow_start_migration"] = report
    return migrated, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/"joint"
    target = output/"paused.pt"
    if target.exists():
        raise ValueError("migration target already exists")
    config = load_config(root/"configs/joint.yaml")
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    source = read_checkpoint(args.source)
    migrated, report = migrate_start(source, config, training_contract(config, dataset, 2))
    report.update(source_checkpoint=str(args.source.resolve()), source_sha256=sha256_file(args.source),
                  migrated_checkpoint=str(target))
    output.mkdir(parents=True, exist_ok=True)
    best = source.get("validation_best")
    if best:
        previous = args.source.parent
        source_best = previous/best["checkpoint"]
        if not source_best.is_file():
            raise ValueError("selected best snapshot is missing")
        try:
            os.link(source_best, output/source_best.name)
        except OSError:
            shutil.copyfile(source_best, output/source_best.name)
        publish_checkpoint_alias(output/source_best.name, output/"best.pt")
        (output/"best.json").write_text(json.dumps(best, indent=2)+"\n")
        for path in previous.glob("validation_*.json"):
            if int(path.stem.split("_")[-1]) <= source["step"]:
                shutil.copyfile(path, output/path.name)
    save_atomic(target, migrated)
    (root/"sf_start_migration.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
