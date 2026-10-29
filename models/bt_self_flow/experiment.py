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


PIXEL_METHOD = "sana_conditioned_pixel_adapter_v1"
PRECISION_PIXEL_METHOD = "precision_consistent_pixel_adapter_v1"


def validate_decoder(config):
    decoder = config.get("decoder", {})
    method = decoder.get("method", "native")
    if method == "native":
        if decoder and set(decoder) != {"method"}:
            raise ValueError("native decoder does not accept PiT options")
        return
    if method not in (PIXEL_METHOD, PRECISION_PIXEL_METHOD):
        raise ValueError("unknown decoder method")
    if set(decoder) - {"method", "pixel_adapter"}:
        raise ValueError("unknown PiT decoder settings")
    if (config["model"]["backend"] != "C" or config["train"]["phase"] not in ("codec", "joint")
            or config["self_flow"].get("enabled") or config["self_flow"].get("mixed_sampling")):
        raise ValueError("PiT requires pure C codec/joint stages with Self-Flow disabled")
    codec = config["model"].get("codec", {})
    if not codec.get("precision_conditioning"):
        raise ValueError("PiT requires precision_conditioning")
    if bool(codec.get("precision_consistent")) != (method == PRECISION_PIXEL_METHOD):
        raise ValueError("PiT decoder method must match the precision-v2/v3 frontend")
    settings = decoder.get("pixel_adapter", {})
    allowed = {"depth", "pixel_dim", "patch_size", "context_dim", "attention_dim", "heads",
               "use_quantization", "delta_ref", "checkpoint_blocks", "seed"}
    if set(settings) - allowed:
        raise ValueError("unknown pixel_adapter option")
    for name, default in (("depth", 2), ("pixel_dim", 16), ("patch_size", 16),
                          ("context_dim", 256), ("attention_dim", 256), ("heads", 4)):
        value = settings.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"pixel_adapter.{name} must be a positive integer")
    if settings.get("pixel_dim", 16) % 4 or settings.get("context_dim", 256) % 2:
        raise ValueError("PiT sinusoidal widths must have compatible dimensions")
    width, heads = settings.get("attention_dim", 256), settings.get("heads", 4)
    if width % heads or (width // heads) % 4:
        raise ValueError("PiT attention head dimension must be divisible by four")
    reference = settings.get("delta_ref", 1.)
    if not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("pixel_adapter.delta_ref must be finite and positive")
    for name in ("use_quantization", "checkpoint_blocks"):
        if name in settings and not isinstance(settings[name], bool):
            raise ValueError(f"pixel_adapter.{name} must be boolean")
    if not isinstance(settings.get("seed", 903), int):
        raise ValueError("pixel_adapter.seed must be an integer")


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
    config = expand_environment(config)
    validate_decoder(config)
    storage = config.get("execution", {}).get("saved_tensors", "device")
    if storage not in ("device", "stride_preserving_cpu"):
        raise ValueError("unknown saved tensor storage policy")
    codec = config["model"].get("codec", {})
    if codec.get("precision_consistent"):
        if not codec.get("precision_conditioning") or config["model"]["backend"] != "C":
            raise ValueError("precision-consistent v3 requires pure C with precision_conditioning")
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
    validate_decoder(config)
    model = config["model"]
    phase = config["train"]["phase"]
    torch.manual_seed(config["train"]["seed"])
    if phase == "codec":
        result = BTCodecStage(model["sana_path"], model.get("codec"))
        size = int(config.get("execution", {}).get("frozen_encoder_batch_size", 0))
        if size < 0:
            raise ValueError("frozen encoder batch size must be nonnegative")
        result.encoder_batch_size = size
    else:
        constructor, decoder_options = BTCodec, {}
        method = config.get("decoder", {}).get("method", "native")
        if method in (PIXEL_METHOD, PRECISION_PIXEL_METHOD):
            if method == PRECISION_PIXEL_METHOD:
                from .precision_pixel_model import PrecisionCPixelAdapterCodec
                constructor = PrecisionCPixelAdapterCodec
            else:
                from .pixel_model import CPixelAdapterCodec
                constructor = CPixelAdapterCodec
            decoder_options["pixel_config"] = config["decoder"].get("pixel_adapter", {})
        result = constructor(model["sana_path"], model["text_path"], backend=model["backend"],
                         codec_params=model.get("codec"), dit_rank=model.get("dit_rank", 128),
                         vae_rank=model.get("vae_rank", 32), checkpoint_blocks=model.get("checkpoint_blocks", True),
                         representation=config["self_flow"].get("enabled", False),
                         student_layer=config["self_flow"].get("student_layer", 8),
                         teacher_layer=config["self_flow"].get("teacher_layer", 20), device=str(device),
                         **decoder_options)
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
        elif name.startswith("pixel_adapter."):
            label, lr = "pixel_adapter", opt["backend_lr"]
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
    value["format"] = ("bt_packet_v3_q16_a3_precision_consistent"
                       if config["model"].get("codec", {}).get("precision_consistent")
                       else "bt_packet_v1_q16_a3_s4")
    return optimization_contract(value)


def optimization_contract(value):
    """Exclude validation selection and tensor placement from optimizer identity.

    CPU/GPU saved-tensor placement changes storage, not the model, objective,
    sample stream or optimizer. Keep all other execution/model settings strict.
    """
    value = deepcopy(value)
    for key in ("valid", "valid_limit", "valid_crop"):
        value["data"].pop(key, None)
    value["train"].pop("validation_actual_bpp", None)
    execution = value.get("execution", {})
    execution.pop("saved_tensors", None)
    if not execution:
        value.pop("execution", None)
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


def check_stage_model(config, source, source_assets=None):
    """Stage changes may add D conditioning/projectors, not change the codec."""
    relocated = any(config["model"].get(key) != source["model"].get(key)
                    for key in ("sana_path", "text_path"))
    verified_relocation = relocated and source_assets is not None and asset_identity(config) == source_assets
    for key in ("sana_path", "text_path", "codec", "dit_rank", "vae_rank"):
        if key in ("sana_path", "text_path") and verified_relocation:
            continue
        if config["model"].get(key) != source["model"].get(key):
            raise ValueError(f"stage handoff model.{key} differs; start a separate experiment")
    if config["model"].get("codec", {}).get("precision_conditioning") and config.get("quality") != source.get("quality"):
        raise ValueError("precision v2 quality sampling and rate coefficients must match across stages")
