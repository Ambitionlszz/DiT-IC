"""Migrate effective-batch8 pure C into matched two-rank C/SF continuations."""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, make_model, optimizer_groups, training_contract
from scripts.prepare_c_sf_ablation import audit_source, paired_configs


def parameter_layout(model, config):
    groups, auxiliary = optimizer_groups(model, config)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    main = [(group["name"], [(names[id(p)], tuple(p.shape)) for p in group["params"]]) for group in groups]
    aux = [(names[id(p)], tuple(p.shape)) for p in auxiliary]
    return groups, auxiliary, main, aux


def remap_optimizer(stored, source_layout, target_layout, template, allow_residual=False, allow_task=False, allow_local=False, allow_context=False):
    """Map moments by names, requiring an exact source parameter partition."""
    if len(stored["param_groups"]) != len(source_layout):
        raise ValueError("source optimizer group count differs from original C")
    if len(template["param_groups"]) != len(target_layout):
        raise ValueError("target optimizer group count differs")
    source_ids = {identity for group in stored["param_groups"] for identity in group["params"]}
    if not set(stored["state"]).issubset(source_ids):
        raise ValueError("source optimizer has orphaned parameter state")
    source_by_name = {}
    source_groups = {}
    for group, (label, parameters) in zip(stored["param_groups"], source_layout):
        if group.get("name", label) != label or len(group["params"]) != len(parameters):
            raise ValueError(f"source optimizer parameter order/count differs in {label}")
        source_groups[label] = group
        for identity, (name, shape) in zip(group["params"], parameters):
            if name in source_by_name:
                raise ValueError("duplicate source optimizer parameter")
            state = stored["state"].get(identity, {})
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step" and tuple(value.shape) != shape:
                    raise ValueError(f"optimizer state shape mismatch: {name}/{key}")
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite source optimizer state: {name}/{key}")
            source_by_name[name] = (shape, state)
    consumed = set()
    result = dict(state={}, param_groups=[])
    for default, (label, parameters) in zip(template["param_groups"], target_layout):
        if len(default["params"]) != len(parameters):
            raise ValueError("target optimizer layout differs")
        group = deepcopy(source_groups.get(label, default))
        group["params"] = default["params"]
        for identity, (name, shape) in zip(group["params"], parameters):
            if name in source_by_name:
                source_shape, state = source_by_name[name]
                if shape != source_shape:
                    raise ValueError(f"shared parameter shape changed: {name}")
                if state:
                    result["state"][identity] = state
                consumed.add(name)
            elif not ((allow_residual and label == "residual" and name.startswith("residual_head."))
                      or (allow_task and label == "task_sf" and name.startswith(("quality_conditioner.", "representation_projector.")))
                      or (allow_local and label == 'backend' and name.startswith('precision_adapter.'))
                      or (allow_context and label == 'backend' and name.startswith(('context_generator.', 'context_readout.')))):
                raise ValueError(f"unexpected new trainable parameter: {name}")
        result["param_groups"].append(group)
    if consumed != set(source_by_name):
        raise ValueError("migration dropped source optimizer parameters")
    return result


def remap_scheduler(stored, source_layout, target_layout, optimizer):
    labels = [label for label, _ in source_layout]
    result = deepcopy(stored)
    for key in ("base_lrs", "_last_lr", "lr_lambdas"):
        values = stored.get(key)
        if values is None or len(values) != len(labels):
            raise ValueError(f"source scheduler is missing group state: {key}")
        by_name = dict(zip(labels, values))
        result[key] = [by_name[label] if label in by_name else
                       None if key == "lr_lambdas" else group.get("initial_lr", group["lr"])
                       for (label, _), group in zip(target_layout, optimizer["param_groups"])]
    return result


def new_rank_rng(seed):
    numpy = np.random.RandomState(seed).get_state()
    return dict(python=random.Random(seed).getstate(), numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
                torch=torch.Generator().manual_seed(seed).get_state(), cuda=None)


def migrate_progress(source, config, target_world=2):
    """Preserve global sample counters when local batch/accumulation changes."""
    source_world = source["contract"]["world_size"]
    previous, current = source["config"]["train"], config["train"]
    layouts = [dict(world_size=world, batch_size=train["batch_size"],
                    accumulation_steps=train["accumulation_steps"])
               for world, train in ((source_world, previous), (target_world, current))]
    for layout, train in zip(layouts, (previous, current)):
        if any(type(value) is not int or value <= 0 for value in layout.values()):
            raise ValueError("invalid batch layout")
        if (layout["world_size"]*layout["batch_size"]*layout["accumulation_steps"] != 8
                or train["effective_batch"] != 8):
            raise ValueError("migration requires effective batch 8 on both sides")
    if source_world not in (1, 2) or len(source["rng_by_rank"]) != source_world:
        raise ValueError("source RNG rank count differs or rank layout is unsupported")
    micro = source["micro_seen"]
    if type(micro) is not int or micro < 0 or micro % previous["accumulation_steps"]:
        raise ValueError("source counter is not at an optimizer update boundary")
    samples = micro*source_world*previous["batch_size"]
    target_stride = target_world*current["batch_size"]
    if samples % target_stride:
        raise ValueError("global sample counter cannot be represented by target layout")
    target_micro = samples//target_stride
    if target_micro % current["accumulation_steps"]:
        raise ValueError("target counter is not at an optimizer update boundary")
    rng = list(source["rng_by_rank"])
    while len(rng) < target_world:
        rng.append(new_rank_rng(current["seed"]+len(rng)))
    return target_micro, samples, rng, layouts


def check_dataset(source, target_config, dataset, allow_changed=False):
    original = source["config"]["data"]["train"]
    current = target_config["data"]["train"]
    if len(original) != len(current):
        raise ValueError("source dataset roots differ")
    projected = []
    for path in dataset.paths:
        matches = [(Path(old)/Path(path).relative_to(Path(new))) for old, new in zip(original, current)
                   if Path(path).is_relative_to(Path(new))]
        if len(matches) != 1:
            raise ValueError(f"ambiguous dataset root mapping: {path}")
        projected.append(str(matches[0]))
    fingerprint = hashlib.sha256("\n".join(sorted(set(projected))).encode()).hexdigest()
    if fingerprint != source["contract"]["dataset_fingerprint"] and not allow_changed:
        raise ValueError("source training file list differs after path relocation")
    return fingerprint


def migrate_state(source, source_layout, source_aux, model, config, dataset, assets, source_hash):
    micro_seen, samples, rng, layouts = migrate_progress(source, config)
    residual = config["self_flow"].get("method") == "quantization_residual_v1"
    local = config['self_flow'].get('method') == 'local_precision_v1'
    extra = {"residual": model.residual_head.state_dict()} if residual else {}
    if local:
        extra['precision_adapter'] = deepcopy(model.precision_adapter.state_dict())
    context = config['self_flow'].get('method') == 'context_sf_v1'
    if context:
        extra.update(context_generator=deepcopy(model.context_generator.state_dict()),
                     context_readout=deepcopy(model.context_readout.state_dict()))
    if residual and model.residual_head.active.item():
        raise ValueError("new residual head must be dormant at the 10K boundary")
    for kind in ("raw", "ema"):
        model.load_bt_state(dict(source[kind], **extra))
        for component, state in source[kind].items():
            actual = model.export_state()[component]
            if state.keys() != actual.keys() or any(not torch.equal(value, actual[key]) for key, value in state.items()):
                raise ValueError(f"{kind} shared weights changed during load: {component}")
            if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
                raise ValueError(f"nonfinite source {kind}: {component}")
    groups, auxiliary, target_layout, target_aux = parameter_layout(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    auxiliary_optimizer = torch.optim.AdamW(auxiliary, lr=config["optimizer"]["aux_lr"], weight_decay=0)
    main = remap_optimizer(source["optimizer"], source_layout, target_layout, optimizer.state_dict(), residual,
                           allow_local=local, allow_context=context)
    aux = remap_optimizer(source["aux_optimizer"], [("aux", source_aux)], [("aux", target_aux)],
                          auxiliary_optimizer.state_dict())
    # Both branches use exactly the same fresh FP16 scaling policy. The CPU
    # scaler serializes the same format without allocating a CUDA context.
    scaler = torch.amp.GradScaler("cpu", init_scale=64.).state_dict()
    result = dict(source)
    result.update(raw=dict(source["raw"], **extra), ema=dict(source["ema"], **extra),
                  optimizer=main, aux_optimizer=aux,
                  scheduler=remap_scheduler(source["scheduler"], source_layout, target_layout, main),
                  scaler=scaler, config=config, contract=training_contract(config, dataset, 2), assets=assets,
                  source_sha256=source_hash, validation_best=None, rep_weight=0., full_precision_retry=False,
                  micro_seen=micro_seen, rng_by_rank=rng)
    result.pop("teacher", None)
    result.pop("residual_sf", None)
    if residual:
        result["residual_sf"] = {}
    result["migration"] = dict(protocol="pure_c_batch8_layout_v2", source_sha256=source_hash,
                               parent_source_sha256=source.get("source_sha256"),
                               source_layout=layouts[0], target_layout=layouts[1],
                               source_micro_seen=source["micro_seen"], target_micro_seen=micro_seen,
                               rng_policy="preserve existing rank RNGs; seed only newly added ranks with train.seed+rank",
                               fresh_fp16_scale=64., preserved_global_samples=samples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New comparison root; contains pure_c and c_sf")
    parser.add_argument("--dataset-policy", choices=("require-source", "shared-local"), default="require-source",
                        help="Require the source file list, or explicitly start both continuations on the same local list")
    args = parser.parse_args()
    configs = paired_configs()
    for name, config in configs.items():
        config["train"]["output"] = str(args.output.resolve()/name/"joint")
        if Path(config["train"]["output"]).exists():
            raise ValueError("migration requires fresh output directories")
    audit = audit_source(args.checkpoint, configs["pure_c"])
    if audit["metadata_errors"]:
        raise ValueError("; ".join(audit["metadata_errors"]))
    source = read_checkpoint(args.checkpoint)
    migrate_progress(source, configs["pure_c"])
    assets = asset_identity(configs["pure_c"])
    if assets != source["assets"]:
        raise ValueError("source pretrained assets differ")
    dataset = ImageDataset(configs["pure_c"]["data"]["train"], configs["pure_c"]["data"]["crop"])
    mapped_fingerprint = check_dataset(source, configs["pure_c"], dataset,
                                       allow_changed=args.dataset_policy == "shared-local")
    dataset_record = dict(policy=args.dataset_policy,
                          source_path_fingerprint=source["contract"]["dataset_fingerprint"],
                          relocated_local_path_fingerprint=mapped_fingerprint,
                          local_path_fingerprint=dataset.fingerprint, local_images=len(dataset),
                          source_file_list_matches=mapped_fingerprint == source["contract"]["dataset_fingerprint"])
    print(json.dumps(dict(dataset=dataset_record)), flush=True)
    source_layout = source_aux = None
    for name, config in configs.items():
        print(json.dumps(dict(migrating=name, step=source["step"])), flush=True)
        model = make_model(config, torch.device("cpu"))
        if source_layout is None:
            _, _, source_layout, source_aux = parameter_layout(model, config)
        checkpoint = migrate_state(source, source_layout, source_aux, model, config, dataset, assets, audit["sha256"])
        checkpoint["migration"]["dataset"] = dataset_record
        destination = Path(config["train"]["output"])/"0010000.pt"
        save_atomic(destination, checkpoint)
        record = dict(source=audit, output=str(destination), config=config,
                      migration=checkpoint["migration"], shared_weights_exact=True,
                      optimizer_states=len(checkpoint["optimizer"]["state"]),
                      auxiliary_states=len(checkpoint["aux_optimizer"]["state"]),
                      source_layout_sha256=hashlib.sha256(json.dumps(source_layout).encode()).hexdigest())
        (destination.parent.parent/"migration.json").write_text(json.dumps(record, indent=2)+"\n")
        del model, checkpoint
        gc.collect()
    print(json.dumps(dict(migrated=True, source_sha256=audit["sha256"], output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
