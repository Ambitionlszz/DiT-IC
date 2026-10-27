"""Dual-quality Self-Flow on the native C path, without a train-only head.

Native H8 retains the codec graph. The extra mixed prefix has detached inputs;
both predict the same spatial relation from an EMA DiT H20. Only the native
packet has an RD objective, and deployment is exactly the original C decoder.
"""
from dataclasses import dataclass
import random

import torch
from torch.nn import functional as F

from .data import counter_seed
from .model import BTCodec
from .packet import RateControl
from .quantization_residual import c_sana_forward
from models.sana_self_flow import unwrap_sana_transformer


METHOD = "native_relational_v1"
MIGRATION = "pure_c10k_native_relational_v1"


def spatial_relation(feature):
    """Off-diagonal, spatially centered Gram entries, always computed in FP32."""
    if feature.ndim != 3 or feature.shape[1] < 2:
        raise ValueError("relations require B,N,D features with N >= 2")
    with torch.autocast(device_type=feature.device.type, enabled=False):
        value = feature.float()
        value = value-value.mean(dim=1, keepdim=True)
        gram = value @ value.transpose(1, 2)
        off_diagonal = ~torch.eye(gram.shape[-1], dtype=torch.bool, device=gram.device)
        return gram[:, off_diagonal]


def relation_distance(relation, target):
    if relation.shape != target.shape:
        raise ValueError("student/teacher spatial grids differ")
    with torch.autocast(device_type=relation.device.type, enabled=False):
        return 1-F.cosine_similarity(relation.float(), target.detach().float(), dim=-1, eps=1e-12)


def relation_ramp(config, step):
    sf = config["self_flow"]
    return max(0., min(1., (step-sf["warmup"]+1)/sf["ramp"]))


@dataclass
class RelationalViews:
    target: torch.Tensor
    mixed: torch.Tensor | None
    quality_b: float
    cell_fraction: float
    teacher_spatial_std: torch.Tensor

    @property
    def native_weight(self):
        return .5 if self.mixed is not None else 1.


class CNativeRelationalCodec(BTCodec):
    def __init__(self, *args, **kwargs):
        kwargs.update(backend="C", representation=False)
        super().__init__(*args, **kwargs)
        if not self.codec.precision_conditioning:
            raise ValueError("native relational SF requires precision-v2 C")

    def predict(self, coarse, log_rho=None, transition=1., capture=False):
        if not capture:
            return super().predict(coarse, log_rho, transition, capture=False)
        self.prediction_calls += 1
        prediction, feature, _, _ = c_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device),
            self.self_flow_timestep(coarse.shape[0], coarse.device),
            capture_layer=self.student_layer, checkpoint_blocks=self.checkpoint_blocks)
        return prediction, feature

    @torch.no_grad()
    def relational_views(self, result, control, teacher, config, step, micro, rank, *, quality_b=None):
        if control.mixed:
            raise ValueError("native relational SF requires a uniform native packet")
        qa = control.value
        # Counter-based, separate streams: never advance the native crop/q RNG.
        qb = (random.Random(counter_seed(config["train"]["seed"], step, 2700)).choice(
            config["quality"]["values"]) if quality_b is None else quality_b)
        if qb not in config["quality"]["values"]:
            raise ValueError("auxiliary quality must be one of the native six points")
        a = result.coarse.detach()
        mixed = None
        if abs(qb-qa) < 2/65535:
            high = a
        else:
            # q conditions g_a itself: re-analyze z, not A's analysis state.
            b_control = RateControl(qb)
            # analyze() samples unused hyper-rate noise in train mode. Preserve
            # the native RNG even though the decoded latent is deterministic.
            devices = [a.device.index] if a.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                source = self.codec.analyze(result.clean.detach(), b_control)
                b = self.codec.quantize_analysis(source, b_control, compute_likelihood=False)["mean"]
            batch, _, height, width = a.shape
            generator = torch.Generator(device=a.device).manual_seed(counter_seed(
                config["train"]["seed"], step, 2800+micro*31+rank))
            count = max(1, round(height*width*config["self_flow"]["cell_fraction"]))
            order = torch.rand(batch, height*width, device=a.device, generator=generator).argsort(-1)
            mask = torch.zeros(batch, height*width, dtype=torch.bool, device=a.device)
            mask.scatter_(1, order[:, :count], True)
            mixed = torch.where(mask.reshape(batch, 1, height, width), b, a)
            high = b if qb > qa else a
        chunk = config["self_flow"].get("teacher_batch_size", a.shape[0])
        features = [teacher(value, self.fixed_text_embed,
                           self.self_flow_timestep(value.shape[0], value.device))
                    for value in high.split(chunk)]
        target = torch.cat(features)
        return RelationalViews(spatial_relation(target), mixed, qb,
            config["self_flow"]["cell_fraction"] if mixed is not None else 0.,
            target.float().std(dim=1).mean())

    def mixed_relation(self, views):
        if views.mixed is None:
            raise ValueError("equal-quality views need only native supervision")
        mixed = views.mixed.detach()
        _, feature, _, _ = c_sana_forward(
            unwrap_sana_transformer(self.DiT), mixed,
            self._fixed_text_condition(mixed.shape[0], mixed.device),
            self.self_flow_timestep(mixed.shape[0], mixed.device),
            capture_layer=self.student_layer, stop_layer=self.student_layer,
            checkpoint_blocks=self.checkpoint_blocks)
        return relation_distance(spatial_relation(feature), views.target).mean()
