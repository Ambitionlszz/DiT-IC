"""Configuration and trainable parameter policies shared by train/eval."""

from copy import deepcopy
from functools import lru_cache
import math
import os
from pathlib import Path
import re

import torch
import yaml

from .checkpoint import sha256_file
from .model import BTCodec, BTCodecStage


def load_config(path):
    path = Path(path).resolve()
    with path.open() as source:
        config = yaml.safe_load(source)
    if "base" in config:
        parent = load_config(path.parent/config.pop("base"))
        config = merge(parent, config)
    config.setdefault("self_flow", {"enabled": False, "mixed_sampling": False})
    phase = config["train"]["phase"]
    if phase not in ("codec", "backend", "joint", "self_flow"):
        raise ValueError("unknown BT training phase")
    if config["model"]["backend"] not in ("C", "D"):
        raise ValueError("unknown BT backend")
    sf = config["self_flow"]
    joint_sf = phase == "joint" and sf.get("enabled") and "start_step" in sf
    if sf.get("enabled") and (not (phase == "self_flow" or joint_sf) or config["model"]["backend"] != "D"):
        raise ValueError("BT Self-Flow requires D and either its own phase or an explicit joint start_step")
    if joint_sf:
        if not 0 <= int(sf["start_step"]) < int(config["train"]["steps"]):
            raise ValueError("self_flow.start_step must fall within joint training")
        if int(sf.get("mixed_every", 2)) < 1 or float(sf.get("ramp", 2000)) <= 0:
            raise ValueError("Self-Flow mixed_every and ramp must be positive")
        if not 0 < float(sf.get("high_fraction", .25)) < 1:
            raise ValueError("Self-Flow needs both low- and high-quality student regions")
    if phase != "self_flow" and not joint_sf and sf.get("mixed_sampling"):
        raise ValueError("initial C/D experiments must not add binary mixtures")
    for key in ("batch_size", "accumulation_steps", "steps", "eval_every", "save_every"):
        if int(config["train"][key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    config = expand_environment(config)
    if config["model"].get("codec", {}).get("precision_conditioning"):
        if phase not in ("codec", "joint") or sf.get("enabled") or sf.get("mixed_sampling"):
            raise ValueError("precision v2 has exactly codec warmup and joint C/D training")
        if config["train"].get("router_uniform_steps", 0) or config["train"].get("router_ramp_steps", 0):
            raise ValueError("precision v2 must use the same full Router throughout both stages")
        values = config.get("quality", {}).get("values", [])
        weights = config.get("quality", {}).get("rate_coefficients", [])
        if values != [0., .2, .4, .6, .8, 1.] or len(weights) != 6:
            raise ValueError("precision v2 requires six shared quality/rate pairs")
        if any(not math.isfinite(w) or w <= 0 for w in weights) or any(a <= b for a, b in zip(weights, weights[1:])):
            raise ValueError("rate coefficients must be positive, finite and descending")
        if config.get("objective", {}).get("repaired_weight", 0) != 0:
            raise ValueError("precision v2 reports repaired NMSE but does not optimize it")
    return config


def expand_environment(value):
    """Portable asset/data paths, including ${NAME:-default}, without a shell."""
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, str):
        def replace(match):
            name, default = match.group(1), match.group(2)
            result = os.environ.get(name) or default
            if result is None:
                raise ValueError(f"set environment variable {name} for this config")
            return result
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}", replace, value)
    return value


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def make_model(config, device):
    model = config["model"]
    phase = config["train"]["phase"]
    torch.manual_seed(config["train"]["seed"])
    if phase == "codec":
        result = BTCodecStage(model["sana_path"], model.get("codec"))
    else:
        result = BTCodec(model["sana_path"], model["text_path"], backend=model["backend"],
                         codec_params=model.get("codec"), dit_rank=model.get("dit_rank", 128),
                         vae_rank=model.get("vae_rank", 32), checkpoint_blocks=model.get("checkpoint_blocks", True),
                         representation=config["self_flow"].get("enabled", False),
                         student_layer=config["self_flow"].get("student_layer", 8),
                         teacher_layer=config["self_flow"].get("teacher_layer", 20), device=str(device))
        result.set_codec_frozen(phase == "backend")
    return result.to(device)


def optimizer_groups(model, config):
    opt = config["optimizer"]
    phase = config["train"]["phase"]
    groups, auxiliary, seen = [], [], set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".quantiles"):
            auxiliary.append(parameter)
            continue
        if name.startswith("codec."):
            label, lr = "codec", opt["codec_lr"] if phase == "codec" else opt["joint_codec_lr"]
        elif name.startswith("vae."):
            if "lora_" not in name or "decoder." not in name:
                raise ValueError(f"unexpected trainable DC-AE parameter: {name}")
            label, lr = "vae", opt["vae_lr"]
        else:
            label, lr = "backend", opt["backend_lr"]
        if id(parameter) in seen:
            raise ValueError("duplicate trainable parameter")
        seen.add(id(parameter))
        matching = next((group for group in groups if group["name"] == label), None)
        if matching is None:
            matching = {"name": label, "params": [], "lr": float(lr)}
            groups.append(matching)
        matching["params"].append(parameter)
    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    aux_ids = {id(p) for p in auxiliary}
    if seen & aux_ids or seen | aux_ids != all_trainable:
        raise ValueError("optimizer groups do not exactly partition trainable parameters")
    return groups, auxiliary


def lr_factor(step, total, warmup=500):
    warmup = min(warmup, total)
    if step < warmup:
        return (step+1)/max(warmup, 1)
    progress = min(1., (step-warmup)/max(total-warmup, 1))
    return .1+.9*.5*(1+math.cos(math.pi*progress))


def schedule_factor(step, total, optimizer):
    schedule = optimizer.get("schedule", "cosine")
    if schedule == "constant":
        return 1.
    if schedule == "milestone":
        milestones = optimizer.get("milestones", [100000, 160000, 190000])
        factors = optimizer.get("factors", [.5, .2, .1])
        if len(milestones) != len(factors) or milestones != sorted(set(milestones)):
            raise ValueError("LR milestones and multiplicative factors do not match")
        return math.prod(factor for boundary, factor in zip(milestones, factors) if step >= boundary)
    if schedule != "cosine":
        raise ValueError("unknown LR schedule")
    return lr_factor(step, total, optimizer.get("warmup", 500))


def training_contract(config, dataset, world_size):
    value = deepcopy(config)
    # Storage locations and log cadence do not change the experiment.
    for key in ("output", "log_every", "eval_every", "save_every", "keep_last_checkpoints"):
        value["train"].pop(key, None)
    value["dataset_fingerprint"] = dataset.fingerprint
    value["world_size"] = world_size
    value["format"] = "bt_packet_v1_q16_a3_s4"
    return optimization_contract(value)


def optimization_contract(value):
    """Validation selection can change explicitly without resetting training."""
    value = deepcopy(value)
    for key in ("valid", "valid_limit", "valid_crop"):
        value["data"].pop(key, None)
    value["train"].pop("validation_actual_bpp", None)
    return value


def asset_identity(config):
    root = Path(config["model"]["sana_path"])
    result = {}
    for part in ("vae", "transformer", "scheduler"):
        filename = "scheduler_config.json" if part == "scheduler" else "config.json"
        file = root/part/filename
        result[f"{part}_config_sha256"] = sha256_file(file)
        weights = root/part/"diffusion_pytorch_model.safetensors"
        if weights.exists():
            result[f"{part}_weights_bytes"] = weights.stat().st_size
            result[f"{part}_weights_sha256"] = _asset_hash(str(weights.resolve()), weights.stat().st_size,
                                                           weights.stat().st_mtime_ns)
    result["text_sha256"] = sha256_file(config["model"]["text_path"])
    return result


@lru_cache(maxsize=8)
def _asset_hash(path, size, modified_ns):
    return sha256_file(path)


def check_stage_model(config, source, assets_verified=False):
    """Stage changes may add D conditioning/projectors, not change the codec."""
    keys = ("codec", "dit_rank", "vae_rank") if assets_verified else ("sana_path", "text_path", "codec", "dit_rank", "vae_rank")
    for key in keys:
        if config["model"].get(key) != source["model"].get(key):
            raise ValueError(f"stage handoff model.{key} differs; start a separate experiment")
    if config["model"].get("codec", {}).get("precision_conditioning") and config.get("quality") != source.get("quality"):
        raise ValueError("precision v2 quality sampling and rate coefficients must match across stages")
