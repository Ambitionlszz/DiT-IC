"""Configuration and trainable parameter policies shared by train/eval."""

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

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
    if sf.get("enabled") and (phase != "self_flow" or config["model"]["backend"] != "D"):
        raise ValueError("BT Self-Flow is a separate D-backend phase")
    if phase != "self_flow" and sf.get("mixed_sampling"):
        raise ValueError("initial C/D experiments must not add binary mixtures")
    for key in ("batch_size", "accumulation_steps", "steps", "eval_every", "save_every"):
        if int(config["train"][key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    rate_weight = config["train"].get("rate_weight", 1.)
    if not math.isfinite(rate_weight) or rate_weight <= 0:
        raise ValueError("train.rate_weight must be finite and positive")
    return config


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


def check_stage_model(config, source):
    """Stage changes may add D conditioning/projectors, not change the codec."""
    for key in ("sana_path", "text_path", "codec", "dit_rank", "vae_rank"):
        if config["model"].get(key) != source["model"].get(key):
            raise ValueError(f"stage handoff model.{key} differs; start a separate experiment")
