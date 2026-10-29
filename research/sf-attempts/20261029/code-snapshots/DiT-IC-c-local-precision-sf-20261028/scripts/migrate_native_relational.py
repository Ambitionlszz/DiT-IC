"""Resume the original pure C10K with unchanged parameters and Adam moments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, load_config, make_model
from models.bt_self_flow.native_relational import METHOD, MIGRATION
from scripts.migrate_c_sf_batch8 import parameter_layout, migrate_state, check_dataset
from scripts.prepare_c_sf_ablation import audit_source


def migrate(source, model, config, dataset, assets, digest):
    if config["self_flow"].get("method") != METHOD:
        raise ValueError("not a native relational SF configuration")
    if set(source["raw"]) != set(model.export_state()) or any(
            key in source for key in ("teacher", "native_relational", "dual_rate_task", "residual_sf")):
        raise ValueError("migration requires original pure C, without SF state")
    _, _, layout, aux = parameter_layout(model, config)
    result = migrate_state(source, layout, aux, model, config, dataset, assets, digest)
    result.update(rep_weight=1., native_relational=dict(protocol=METHOD, teacher_initialized=False))
    result["migration"].update(protocol=MIGRATION, shared_weights_exact=True,
        optimizer_moments_preserved=True, teacher_initialization="raw student C10K prefix, before first update")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--allow-path-relocation", action="store_true")
    args = p.parse_args()
    config = load_config(args.config)
    audit = audit_source(args.checkpoint, config)
    if audit["metadata_errors"]:
        raise ValueError(audit["metadata_errors"])
    source = read_checkpoint(args.checkpoint)
    assets = asset_identity(config)
    if assets != source["assets"]:
        raise ValueError("source C pretrained assets differ")
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    projected = check_dataset(source, config, dataset, args.allow_path_relocation)
    model = make_model(config, torch.device("cpu"))
    result = migrate(source, model, config, dataset, assets, audit["sha256"])
    result["migration"]["dataset"] = dict(local_images=len(dataset), local_fingerprint=dataset.fingerprint,
        relocated_fingerprint=projected, source_fingerprint=source["contract"]["dataset_fingerprint"],
        user_confirmed_directory_layout_only=args.allow_path_relocation)
    path = Path(config["train"]["output"])/"0010000.pt"
    if path.exists():
        raise ValueError("migration output already exists")
    save_atomic(path, result)
    report = dict(output=str(path), source=audit, migration=result["migration"])
    (path.parent.parent/"migration.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
