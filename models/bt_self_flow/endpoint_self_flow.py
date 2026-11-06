"""Native EMA endpoints and decoder-feature supervision for endpoint SF.

The teacher predicts its own endpoint from a native C view.  It never lends
high-rate inputs or hidden states to the student's reconstruction path.
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from .packet import RateControl, quality_code


METHOD = "native_endpoint_decoder_feature_self_flow_v1"
TARGETS = frozenset({"adjacent_ema", "same_rate_ema", "clean_latent"})
NORMALIZATIONS = frozenset({"shared_student", "independent_spatial"})


def endpoint_sf_coefficient(settings):
    value = settings.get("coefficient", 1.)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("endpoint SF coefficient must be positive and finite")
    return float(value)


def endpoint_sf_normalization(settings):
    value = settings.get("normalization", "shared_student")
    if value not in NORMALIZATIONS:
        raise ValueError(f"unknown endpoint SF normalization: {value}")
    return value


def adjacent_quality(config, control):
    """Return the next configured packet quality, or None at the highest one.

    Quality is sampled once per global update by the existing trainer.  This
    function is deterministic and consumes no random numbers on either rank.
    """
    if control.mixed:
        raise ValueError("endpoint SF requires native, whole-image quality")
    values = config.get("quality", {}).get("values", ())
    codes = [quality_code(value) for value in values]
    if len(codes) < 2 or codes != sorted(set(codes)):
        raise ValueError("endpoint SF requires distinct, increasing quality levels")
    if control.low_code not in codes:
        raise ValueError("student quality must be one of the configured quality levels")
    index = codes.index(control.low_code)
    return None if index == len(codes) - 1 else RateControl(values[index + 1])


def _validate_spatial_features(student, teacher, epsilon):
    if (student.ndim != 4 or student.shape != teacher.shape
            or any(size == 0 for size in student.shape)):
        raise ValueError("endpoint features must have matching, nonempty [B,C,H,W] shapes")
    if student.device != teacher.device:
        raise ValueError("endpoint features must be on the same device")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("spatial normalization epsilon must be positive and finite")


def _shared_spatial_error(student, teacher, epsilon):
    _validate_spatial_features(student, teacher, epsilon)
    with torch.autocast(device_type=student.device.type, enabled=False):
        value, target = student.float(), teacher.detach().float()
        # Both features use the SAME student statistics. The shared mean
        # cancels, so their L1 distance is exactly abs(value-target)/std.
        # Detach the statistics: reducing the loss must not differentiate
        # through a larger normalization denominator in this update.
        reference = value.detach()
        mean = reference.mean(dim=(-2, -1), keepdim=True)
        variance = (reference - mean).square().mean(dim=(-2, -1), keepdim=True)
        std = torch.sqrt(variance + epsilon)
        return (value - target).abs() / std, std


def _independent_spatial_error(student, teacher, epsilon):
    _validate_spatial_features(student, teacher, epsilon)
    with torch.autocast(device_type=student.device.type, enabled=False):
        value, target = student.float(), teacher.detach().float()
        centered = value - value.mean(dim=(-2, -1), keepdim=True)
        target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
        # Differentiating the student's own statistics is part of this loss.
        std = (centered.square().mean(dim=(-2, -1), keepdim=True) + epsilon).sqrt()
        target_std = (target_centered.square().mean(dim=(-2, -1), keepdim=True) + epsilon).sqrt()
        return (centered / std - target_centered / target_std).abs(), std, target_std


def independent_spatial_l1(student, teacher, epsilon=1e-6):
    """Compare per-channel spatial patterns with differentiable student moments."""
    error, _, _ = _independent_spatial_error(student, teacher, epsilon)
    return error.sum(), error.numel()


def endpoint_spatial_l1(student, teacher, settings):
    function = (independent_spatial_l1 if endpoint_sf_normalization(settings) == "independent_spatial"
                else shared_spatial_l1)
    return function(student, teacher, settings.get("epsilon", 1e-6))


def shared_spatial_l1(student, teacher, epsilon=1e-6):
    """Return FP32 error sum/count for shared-statistics spatial L1.

    This retains differences in mean and scale between the features, unlike
    independently standardizing both sides.  Only student values get gradients.
    """
    error, _ = _shared_spatial_error(student, teacher, epsilon)
    return error.sum(), error.numel()


@torch.no_grad()
def shared_spatial_diagnostics(student, teacher, epsilon=1e-6):
    """Report normalization conditioning; these values never weight the loss."""
    error, std = _shared_spatial_error(student, teacher, epsilon)
    # A fixed multiple of the numerical floor is a diagnostic only, not a
    # learned gate or a rule for choosing teacher supervision.
    low = std <= 10 * math.sqrt(epsilon)
    channel_error = error.sum(dim=(-2, -1), keepdim=True)
    return {
        "sf_std_min": std.min(),
        "sf_std_median": std.flatten().median(),
        "sf_low_variance_fraction": low.float().mean(),
        "sf_low_variance_contribution": (channel_error * low).sum()
        / channel_error.sum().clamp_min(torch.finfo(torch.float32).tiny),
    }


@torch.no_grad()
def endpoint_spatial_diagnostics(student, teacher, settings):
    epsilon = settings.get("epsilon", 1e-6)
    if endpoint_sf_normalization(settings) == "shared_student":
        return shared_spatial_diagnostics(student, teacher, epsilon)
    error, std, target_std = _independent_spatial_error(student, teacher, epsilon)
    low = std <= 10 * math.sqrt(epsilon)
    channel_error = error.sum(dim=(-2, -1), keepdim=True)
    return {
        "sf_std_min": std.min(),
        "sf_std_median": std.flatten().median(),
        "sf_low_variance_fraction": low.float().mean(),
        "sf_low_variance_contribution": (channel_error * low).sum()
        / channel_error.sum().clamp_min(torch.finfo(torch.float32).tiny),
        "sf_teacher_std_min": target_std.min(),
        "sf_teacher_low_variance_fraction": (target_std <= 10 * math.sqrt(epsilon)).float().mean(),
    }


def sf_parameters(model):
    """All existing trainable SANA parameters, including its real output head.

    Shared aliases remain excluded if a future wrapper also registers codec,
    condition/text, AE, or D parameters underneath the SANA module.
    """
    excluded = set()
    for name in ("codec", "text_proj", "quality_conditioner", "vae", "pixel_adapter"):
        module = getattr(model, name, None)
        if module is not None:
            excluded.update(id(parameter) for parameter in module.parameters())
    result, seen = [], set()
    for parameter in model.DiT.parameters():
        identity = id(parameter)
        if parameter.requires_grad and identity not in excluded and identity not in seen:
            result.append(parameter)
            seen.add(identity)
    return result


class EndpointEMATeacher(nn.Module):
    """Full C/SANA teacher reconstructed exclusively from deployment EMA.

    Frozen pretrained base weights are copied once. Every exported SANA LoRA
    tensor, including output-head adapters, is subsequently loaded from the
    single deployment EMA. No AE, D, or independent teacher history is kept.
    """

    def __init__(self, model):
        super().__init__()
        self.codec = deepcopy(model.codec)
        self.DiT = deepcopy(model.DiT)
        self.text_proj = deepcopy(model.text_proj)
        self.quality_conditioner = deepcopy(model.quality_conditioner)
        self.sched = deepcopy(model.sched)
        self.register_buffer("fixed_text_embed", model.fixed_text_embed.detach().clone())
        self.register_buffer("timestep", model.self_flow_timestep(
            1, model.fixed_text_embed.device).detach().clone())
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        # A teacher cannot enter stochastic training/quantization mode when a
        # parent container calls train(). Input gradients are never needed.
        super().train(False)
        return self

    @torch.no_grad()
    def load_deployment_ema(self, state):
        required = {"codec", "dit", "text", "conditioner"}
        if not required.issubset(state):
            raise ValueError(f"EMA is missing endpoint teacher components: {sorted(required - set(state))}")
        expected = {name for name in self.DiT.state_dict() if "lora_" in name}
        actual = set(state["dit"])
        if actual != expected:
            raise ValueError("EMA must contain the complete, exact SANA LoRA state; "
                             f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        # Validate all key sets before mutating the teacher. Shape mismatches
        # still raise through the corresponding strict module loader.
        for name, key in (("codec", "codec"), ("text_proj", "text"),
                          ("quality_conditioner", "conditioner")):
            module = getattr(self, name)
            if module is None or set(state[key]) != set(module.state_dict()):
                raise ValueError(f"EMA has incompatible endpoint teacher {key} state")
        self.codec.load_state_dict(state["codec"], strict=True)
        self.DiT.load_state_dict(state["dit"], strict=False)
        self.text_proj.load_state_dict(state["text"], strict=True)
        self.quality_conditioner.load_state_dict(state["conditioner"], strict=True)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, clean, control):
        if control.mixed:
            raise ValueError("endpoint EMA teacher requires a native C view")
        # Run the teacher's own analysis AND quantization at its own quality.
        # Reusing low-rate source analysis would remove some privileged bits.
        view = self.codec(clean.detach(), control)
        mean, delta = view["mean"], view["delta"]
        context = self.text_proj(self.fixed_text_embed).expand(mean.shape[0], -1, -1)
        log_step = F.interpolate(delta.float().log(), size=mean.shape[-2:],
                                 mode="nearest").squeeze(1)
        prediction, _ = quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), mean, context,
            self.timestep.expand(mean.shape[0]), log_step, self.quality_conditioner,
            continuous_conditioning=True, precision_embedding=self.codec.precision_embedding)
        return self.sched.step(prediction, mean)
