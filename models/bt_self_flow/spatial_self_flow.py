"""Spatial-response supervision; normalization is used only by the loss."""

from copy import deepcopy
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from .data import counter_seed
from .packet import RateControl


METHOD = "cross_rate_spatial_response_self_flow_v1"


class SpatialProjector(nn.Module):
    """Identity-initialized, main-path convolution on the native token grid."""

    def __init__(self, channels):
        super().__init__()
        # Conv2d's constructor initializes randomly before Dirac initialization.
        # Preserve the student's RNG even when a baseline omits this module.
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
            nn.init.dirac_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, tokens, height, width, active=None):
        batch, count, channels = tokens.shape
        if count != height * width:
            raise ValueError("spatial projector requires the native two-dimensional grid")
        if active is not None:
            if active.shape != (batch,) or active.dtype != torch.bool:
                raise ValueError("projector activity must be one boolean per sample")
            if not bool(active.any()):
                return tokens
        grid = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        result = self.conv(grid).flatten(2).transpose(1, 2)
        return result if active is None else torch.where(active[:, None, None], result, tokens)


def spatial_normalize(feature, epsilon=1e-6, valid=None):
    """Per-image, per-channel population statistics over valid spatial tokens."""
    if feature.ndim != 3 or feature.shape[1] == 0 or epsilon <= 0:
        raise ValueError("expected nonempty [B,N,C] features and positive epsilon")
    with torch.autocast(device_type=feature.device.type, enabled=False):
        value = feature.float()
        if valid is None:
            mean = value.mean(dim=1, keepdim=True)
            variance = (value - mean).square().mean(dim=1, keepdim=True)
            return (value - mean) * torch.rsqrt(variance + epsilon)
        if valid.shape != value.shape[:2] or valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
            raise ValueError("each sample needs at least one valid spatial token")
        weight = valid[..., None].float()
        count = weight.sum(dim=1, keepdim=True)
        mean = (value * weight).sum(dim=1, keepdim=True) / count
        centered = value - mean
        variance = (centered.square() * weight).sum(dim=1, keepdim=True) / count
        return centered * torch.rsqrt(variance + epsilon) * weight


def spatial_response_error(student, teacher, epsilon=1e-6, valid=None):
    """Return a sum and element count, permitting a global distributed mean."""
    if student.shape != teacher.shape:
        raise ValueError("teacher and student spatial feature grids must match")
    target = spatial_normalize(teacher.detach(), epsilon, valid)
    error = (spatial_normalize(student, epsilon, valid) - target).abs()
    if valid is None:
        return error.sum(), error.numel()
    return (error * valid[..., None]).sum(), int(valid.sum()) * student.shape[-1]


def active_samples(settings, batch, seed, step, rank, micro, device):
    generator = torch.Generator(device="cpu").manual_seed(
        counter_seed(seed, step, settings.get("bypass_stream", 4903) + rank * 1009 + micro * 100003))
    probability = settings.get("bypass_probability", .1)
    return (torch.rand(batch, generator=generator) >= probability).to(device)


def higher_quality(config, low, step, rank, micro=0):
    settings = config["spatial_self_flow"]
    candidates = [q for q in config["quality"]["values"] if q > low.value + 1e-5]
    if not candidates:
        return low
    rng = random.Random(counter_seed(config["train"]["seed"], step,
                                    settings.get("teacher_stream", 3903) + rank * 1009 + micro * 100003))
    return RateControl(rng.choice(candidates))


class SpatialEMATeacher(nn.Module):
    """Truncated teacher derived from deployment EMA, without a second history."""

    def __init__(self, model):
        super().__init__()
        self.codec = deepcopy(model.codec)
        self.DiT = deepcopy(model.DiT)
        sana = unwrap_sana_transformer(self.DiT)
        self.target_layer = model.teacher_layer
        sana.transformer_blocks = nn.ModuleList(list(sana.transformer_blocks[:self.target_layer + 1]))
        self.text_proj = deepcopy(model.text_proj)
        self.quality_conditioner = deepcopy(model.quality_conditioner)
        self.spatial_projector = deepcopy(model.spatial_projector)
        self.projector_layer = model.student_layer
        self.register_buffer("fixed_text_embed", model.fixed_text_embed.detach().clone())
        self.register_buffer("timestep", model.self_flow_timestep(1, model.fixed_text_embed.device).detach().clone())
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def load_deployment_ema(self, state):
        self.codec.load_state_dict(state["codec"], strict=True)
        expected = {name for name in self.DiT.state_dict() if "lora_" in name}
        if not expected.issubset(state["dit"]):
            raise ValueError("EMA is missing truncated teacher SANA parameters")
        self.DiT.load_state_dict({name: state["dit"][name] for name in expected}, strict=False)
        self.text_proj.load_state_dict(state["text"], strict=True)
        self.quality_conditioner.load_state_dict(state["conditioner"], strict=True)
        self.spatial_projector.load_state_dict(state["spatial_projector"], strict=True)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, clean, control):
        # Analysis is quality-conditioned: perform a native high-rate forward,
        # rather than requantizing the student's low-rate source analysis.
        view = self.codec(clean.detach(), control)
        mean, delta = view["mean"], view["delta"]
        context = self.text_proj(self.fixed_text_embed).expand(mean.shape[0], -1, -1)
        log_step = F.interpolate(delta.float().log(), size=mean.shape[-2:], mode="nearest").squeeze(1)
        _, target = quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), mean, context, self.timestep.expand(mean.shape[0]),
            log_step, self.quality_conditioner, capture_layer=self.target_layer,
            stop_layer=self.target_layer, continuous_conditioning=True,
            precision_embedding=self.codec.precision_embedding,
            spatial_projector=self.spatial_projector, projector_layer=self.projector_layer)
        return target


def sf_parameters(model):
    """Only P and trainable ancestors of M; C/shared precision stays main-only."""
    sana = unwrap_sana_transformer(model.DiT)
    modules = [model.spatial_projector, model.text_proj, model.quality_conditioner,
               sana.patch_embed, sana.time_embed, sana.caption_projection, sana.caption_norm,
               *list(sana.transformer_blocks[:model.student_layer + 1])]
    codec_ids = {id(parameter) for parameter in model.codec.parameters()}
    seen, parameters = set(), []
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen and id(parameter) not in codec_ids:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters
