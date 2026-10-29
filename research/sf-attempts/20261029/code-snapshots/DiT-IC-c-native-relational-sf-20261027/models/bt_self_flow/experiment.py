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
    if config.get("quality", {}).get("sampling", "independent") not in ("independent", "balanced"):
        raise ValueError("quality.sampling must be independent or balanced")
    if config["train"].get("amp_dtype", "auto") not in ("auto", "fp16", "bf16"):
        raise ValueError("train.amp_dtype must be auto, fp16 or bf16")
    task_sf = sf.get("method") == "dual_rate_task_v1"
    residual_sf = sf.get("method") == "quantization_residual_v1"
    relational_sf = sf.get("method") == "native_relational_v1"
    if residual_sf and (phase != "joint" or config["model"]["backend"] != "C" or not config["model"].get("codec", {}).get("precision_conditioning")):
        raise ValueError("quantization residual SF requires precision-v2 joint C")
    if residual_sf:
        if not 0 <= sf.get("warmup", -1) < config["train"]["steps"] or sf.get("ramp", 0) <= 0:
            raise ValueError("residual SF requires a valid RD prefix and positive ramp")
        if not 0 < sf.get("cell_fraction", 0) < 1 or not 0 < sf.get("teacher_decay", .999) < 1:
            raise ValueError("invalid residual SF erasure fraction or EMA decay")
        if "weight" in sf and (isinstance(sf["weight"], bool) or
                                not isinstance(sf["weight"], (int, float)) or
                                not math.isfinite(sf["weight"]) or sf["weight"] <= 0):
            raise ValueError("self_flow.weight must be finite and positive")
    if task_sf:
        if phase != "joint" or config["model"]["backend"] != "C" or not sf.get("enabled"):
            raise ValueError("dual-rate task SF requires joint C with SF enabled")
        if not config["model"].get("codec", {}).get("precision_conditioning"):
            raise ValueError("dual-rate task SF requires precision conditioning")
        if sf.get("weight") != 1. or sf.get("ramp", 0) <= 0 or not 0 < sf.get("cell_fraction", 0) < 1:
            raise ValueError("task SF requires explicit unit weight and a valid mask/ramp")
    if relational_sf:
        if (phase != "joint" or config["model"]["backend"] != "C" or not sf.get("enabled")
                or not config["model"].get("codec", {}).get("precision_conditioning")):
            raise ValueError("native relational SF requires enabled precision-v2 joint C")
        if (sf.get("weight") != 1. or sf.get("ramp", 0) <= 0
                or not 0 <= sf.get("warmup", -1) < config["train"]["steps"]
                or not 0 < sf.get("cell_fraction", 0) < 1
                or not 0 < sf.get("teacher_decay", 0) < 1
                or type(sf.get("teacher_batch_size", 4)) is not int or sf.get("teacher_batch_size", 4) < 1):
            raise ValueError("invalid relational SF weight, ramp, mask or EMA settings")
        if sf.get("student_layer") != 8 or sf.get("teacher_layer") != 20:
            raise ValueError("this relational SF protocol requires student 8 and teacher 20")
        if any(key in sf for key in ("prepare_steps", "prepare_lr", "gradient_target", "calibrate_every")):
            raise ValueError("relational SF has no predictor preparation or weight calibration")
    if sf.get("enabled") and not (residual_sf or task_sf or relational_sf) and (phase != "self_flow" or config["model"]["backend"] != "D"):
        raise ValueError("BT Self-Flow is a separate D-backend phase")
    if phase != "self_flow" and sf.get("mixed_sampling"):
        raise ValueError("initial C/D experiments must not add binary mixtures")
    for key in ("batch_size", "accumulation_steps", "steps", "eval_every", "save_every"):
        if int(config["train"][key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    config = expand_environment(config)
    if config["model"].get("codec", {}).get("precision_conditioning"):
        if phase not in ("codec", "joint") or (sf.get("enabled") and not (residual_sf or task_sf or relational_sf)) or sf.get("mixed_sampling"):
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
        residual_sf = config["self_flow"].get("method") == "quantization_residual_v1"
        if residual_sf:
            from .quantization_model import CQuantizationResidualCodec
        task_sf = config["self_flow"].get("method") == "dual_rate_task_v1"
        if task_sf:
            from .dual_rate_task import CDualRateTaskCodec
        relational_sf = config["self_flow"].get("method") == "native_relational_v1"
        if relational_sf:
            from .native_relational import CNativeRelationalCodec
        factory = (CNativeRelationalCodec if relational_sf else CDualRateTaskCodec if task_sf
                   else CQuantizationResidualCodec if residual_sf else BTCodec)
        result = factory(model["sana_path"], model["text_path"], backend=model["backend"],
                         codec_params=model.get("codec"), dit_rank=model.get("dit_rank", 128),
                         vae_rank=model.get("vae_rank", 32), checkpoint_blocks=model.get("checkpoint_blocks", True),
                         representation=config["self_flow"].get("enabled", False) and not (residual_sf or task_sf or relational_sf),
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
        elif name.startswith("residual_head."):
            label, lr = "residual", opt["backend_lr"]
        elif config["self_flow"].get("method") == "dual_rate_task_v1" and name.startswith(("quality_conditioner.", "representation_projector.")):
            label, lr = "task_sf", opt["backend_lr"]
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


def check_stage_model(config, source, asset_verified=False):
    """Stage changes may add D conditioning/projectors, not change the codec."""
    for key in ("sana_path", "text_path", "codec", "dit_rank", "vae_rank"):
        if asset_verified and key in ("sana_path", "text_path"):
            continue
        if config["model"].get(key) != source["model"].get(key):
            raise ValueError(f"stage handoff model.{key} differs; start a separate experiment")
    if config["model"].get("codec", {}).get("precision_conditioning") and config.get("quality") != source.get("quality"):
        raise ValueError("precision v2 quality sampling and rate coefficients must match across stages")
