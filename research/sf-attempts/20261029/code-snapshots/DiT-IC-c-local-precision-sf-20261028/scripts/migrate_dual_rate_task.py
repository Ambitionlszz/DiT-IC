"""Preserve pure C10K weights, moments and sample progress; add task SF only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic, sha256_file, clone_state
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model, training_contract, asset_identity
from scripts.migrate_c_sf_batch8 import (parameter_layout, remap_optimizer, remap_scheduler,
                                         migrate_progress, check_dataset)
from scripts.prepare_c_sf_ablation import audit_source


def migrate(source, model, config, dataset, assets, digest):
    groups, aux_parameters, target_layout, aux_layout = parameter_layout(model, config)
    # All original C parameters retain their registration order and names;
    # require exact optimizer partitions and shapes in remap_optimizer.
    source_layout = [(label, entries) for label, entries in target_layout if label != 'task_sf']
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    auxiliary = torch.optim.AdamW(aux_parameters, lr=config['optimizer']['aux_lr'], weight_decay=0)
    main = remap_optimizer(source['optimizer'], source_layout, target_layout,
                           optimizer.state_dict(), allow_task=True)
    aux = remap_optimizer(source['aux_optimizer'], [('aux', aux_layout)], [('aux', aux_layout)], auxiliary.state_dict())
    extra = {k: v for k, v in clone_state(model.export_state(), cpu=True).items() if k in ('conditioner', 'projector')}
    weights = {}
    for kind in ('raw', 'ema'):
        weights[kind] = dict(source[kind], **extra)
        model.load_bt_state(weights[kind])
        for part, values in source[kind].items():
            actual = model.export_state()[part]
            if actual.keys() != values.keys() or any(not torch.equal(actual[k], v) for k, v in values.items()):
                raise ValueError(f'{kind} C state changed: {part}')
            if any(not torch.isfinite(v).all() for v in values.values() if v.is_floating_point()):
                raise ValueError(f'nonfinite {kind} state: {part}')
    micro, samples, rng, layouts = migrate_progress(source, config)
    result = dict(source, **weights, optimizer=main, aux_optimizer=aux,
        scheduler=remap_scheduler(source['scheduler'], source_layout, target_layout, main),
        scaler=torch.amp.GradScaler('cpu', init_scale=64.).state_dict(),
        config=config, contract=training_contract(config, dataset, 2), assets=assets,
        source_sha256=digest, validation_best=None, rep_weight=1., full_precision_retry=False,
        micro_seen=micro, rng_by_rank=rng, dual_rate_task=dict(prepared=False))
    result.pop('teacher', None)
    result.pop('residual_sf', None)
    result['migration'] = dict(protocol='pure_c10k_dual_rate_task_v1', source_sha256=digest,
        source_layout=layouts[0], target_layout=layouts[1], preserved_global_samples=samples,
        source_micro_seen=source['micro_seen'], target_micro_seen=micro,
        shared_weights_exact=True, existing_optimizer_moments_preserved=True, fresh_fp16_scale=64.)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--allow-path-relocation', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    audit = audit_source(args.checkpoint, config)
    if audit['metadata_errors']:
        raise ValueError(audit['metadata_errors'])
    source = read_checkpoint(args.checkpoint)
    assets = asset_identity(config)
    if assets != source['assets']:
        raise ValueError('pretrained assets differ from source C')
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
