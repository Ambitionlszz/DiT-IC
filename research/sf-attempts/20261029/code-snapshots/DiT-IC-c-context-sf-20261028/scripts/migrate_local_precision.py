"""Preserve original pure C10K state, adding only a zero precision adapter."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, load_config, make_model
from models.bt_self_flow.local_precision import METHOD, MIGRATION
from scripts.migrate_c_sf_batch8 import parameter_layout, migrate_state, check_dataset
from scripts.prepare_c_sf_ablation import audit_source


def migrate(source, model, config, dataset, assets, digest):
    if config['self_flow'].get('method') != METHOD or source['config']['self_flow'].get('enabled'):
        raise ValueError('requires original pure C10K and local precision target')
    if set(source['raw']) | {'precision_adapter'} != set(model.export_state()):
        raise ValueError('unexpected source/target state components')
    if any(torch.count_nonzero(v) for v in model.precision_adapter.state_dict().values()):
        raise ValueError('new precision adapter must be exactly zero initialized')
    _, _, layout, aux = parameter_layout(model, config)
    source_layout = [(label, [(n, s) for n, s in params if not n.startswith('precision_adapter.')])
                     for label, params in layout]
    result = migrate_state(source, source_layout, aux, model, config, dataset, assets, digest)
    result.update(rep_weight=1., local_precision=dict(protocol=METHOD, teacher_initialized=False))
    result['migration'].update(protocol=MIGRATION, shared_weights_exact=True,
        optimizer_moments_preserved=True, new_parameters='precision_adapter zero projection; fresh Adam moments',
        teacher_initialization='raw C10K DiT prefix20 and text, plus zero precision adapter')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--allow-path-relocation', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    old_recipe = deepcopy(config)
    old_recipe['model']['codec'].pop('local_precision_protocol')
    audit = audit_source(args.checkpoint, old_recipe)
    if audit['metadata_errors']:
        raise ValueError(audit['metadata_errors'])
    source = read_checkpoint(args.checkpoint)
    assets = asset_identity(config)
    if assets != source['assets']:
        raise ValueError('pretrained assets differ')
    dataset = ImageDataset(config['data']['train'], config['data']['crop'])
    mapped = check_dataset(source, config, dataset, args.allow_path_relocation)
    model = make_model(config, torch.device('cpu'))
    result = migrate(source, model, config, dataset, assets, audit['sha256'])
    result['migration']['dataset'] = dict(local_images=len(dataset), local_fingerprint=dataset.fingerprint,
        relocated_fingerprint=mapped, source_fingerprint=source['contract']['dataset_fingerprint'],
        user_confirmed_directory_layout_only=args.allow_path_relocation)
    path = Path(config['train']['output'])/'0010000.pt'
    if path.exists():
        raise ValueError('migration output already exists')
    save_atomic(path, result)
    report = dict(output=str(path), source=audit, migration=result['migration'])
    (path.parent.parent/'migration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
