"""C-only quantization residual Self-Flow; no alternate decoder or packet."""

from copy import deepcopy
from types import SimpleNamespace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.sana_self_flow import unwrap_sana_transformer


def token_rms(value):
    return value.float().square().mean(-1, keepdim=True).add(1e-6).sqrt()


def normalized_residual(fine, received):
    return (fine.float()/token_rms(fine) - received.float()/token_rms(received)).detach()


def gaussian_residual_loss(mean, variance, target, valid):
    if valid.dtype != torch.bool or valid.shape != (mean.shape[0],):
        raise ValueError("valid must be a boolean flag per sample")
    nll = .5*((target.float()-mean.float()).square()/variance.float() + variance.float().log())
    # Normalize by the *whole microbatch*, then divide by the configured
    # participation probability in the trainer. This is stable across b8/a1
    # and b4/a2 and does not amplify rare nondegenerate examples.
    per_sample = nll.mean((1, 2))
    return (per_sample*valid.float()).mean()


def erasure_mask(codes, participation, cell_fraction, generator):
    batch, _, height, width = codes.shape
    if not 0 <= participation <= 1 or not 0 < cell_fraction < 1:
        raise ValueError("invalid erasure proportions")
    # Exactly half the samples are eligible at the steady-state p=.5;
    # thin that half during the ramp without changing cell fraction.
    active = torch.zeros(batch, dtype=torch.bool, device=codes.device)
    eligible = batch if participation > .5 else max(1, batch//2)
    order = torch.randperm(batch, device=codes.device, generator=generator)[:eligible]
    active[order] = torch.rand(eligible, device=codes.device, generator=generator) < min(1., participation*batch/eligible)
    count = max(1, round(height*width*cell_fraction))
    priorities = torch.rand(batch, height*width, device=codes.device, generator=generator)
    indices = priorities.argsort(-1)[:, :count]
    mask = torch.zeros(batch, height*width, dtype=torch.bool, device=codes.device)
    mask.scatter_(1, indices, True)
    return mask.reshape_as(codes) & active[:, None, None, None]


class QuantizationResidualHead(nn.Module):
    def __init__(self, hidden, initial_std=.5, minimum_std=.05):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden+1, 2*hidden), nn.SiLU())
        self.mean = nn.Linear(2*hidden, hidden)
        self.variance = nn.Linear(2*hidden, hidden)
        self.minimum_variance = float(minimum_std)**2
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.variance.weight)
        nn.init.constant_(self.variance.bias, math.log(math.expm1(initial_std**2-self.minimum_variance)))

    def forward(self, feature, log_step, token_shape):
        # The caller detaches log_step for the SF-only replay, but leaves it
        # connected for the actual RD reconstruction.
        condition = F.interpolate(log_step.float(), size=token_shape, mode="nearest").flatten(2).transpose(1, 2)
        hidden = self.trunk(torch.cat([self.norm(feature), condition.to(feature.dtype)], -1))
        mean = self.mean(hidden).float()
        variance = self.minimum_variance + F.softplus(self.variance(hidden).float())
        return mean, variance


def c_sana_forward(sana, latent, condition, timestep, *, capture_layer=None,
                   stop_layer=None, checkpoint_blocks=False, head=None,
                   log_step=None, student_layer=7, injection_layer=19,
                   residual_override=None, shuffle_mean=False):
    """Native SANA blocks with an explicit 8→20 reconstruction connection.

    Layer indices are zero based. With no head this follows the library C
    forward exactly; teacher/prefix execution returns before output norm.
    """
    batch, _, height, width = latent.shape
    patch = int(sana.config.patch_size)
    height, width = height//patch, width//patch
    hidden = sana.patch_embed(latent)
    modulation, embedded = sana.time_embed(timestep, batch_size=batch, hidden_dtype=hidden.dtype)
    condition = sana.caption_projection(condition).view(batch, -1, hidden.shape[-1])
    condition = sana.caption_norm(condition)
    captured = mean = variance = None
    for index, block in enumerate(sana.transformer_blocks):
        args = (hidden, None, condition, None, modulation, height, width)
        hidden = checkpoint(block, *args, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else block(*args)
        if index == capture_layer:
            captured = hidden
        if head is not None and index == student_layer:
            mean, variance = head(hidden, log_step, (height, width))
            if shuffle_mean:
                order = torch.randperm(mean.shape[1], generator=torch.Generator().manual_seed(417)).to(mean.device)
                mean = mean[:, order]
        if index == injection_layer and (mean is not None or residual_override is not None):
            correction = mean if residual_override is None else residual_override
            hidden = hidden + (token_rms(hidden).detach()*correction.float()).to(hidden.dtype)
        if index == stop_layer:
            return None, captured, mean, variance
    hidden = sana.norm_out(hidden, embedded, sana.scale_shift_table)
    hidden = sana.proj_out(hidden).reshape(batch, height, width, patch, patch, -1)
    output = hidden.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, height*patch, width*patch)
    return output, captured, mean, variance


class CResidualTeacher(nn.Module):
    """One frozen EMA prefix, reused sequentially for received/fine views."""
    def __init__(self, student_dit, text_projection, target_layer=19):
        super().__init__()
        sana = unwrap_sana_transformer(student_dit)
        self.target_layer = int(target_layer)
        self.config = SimpleNamespace(patch_size=int(sana.config.patch_size))
        for name in ("patch_embed", "time_embed", "caption_projection", "caption_norm"):
            setattr(self, name, deepcopy(getattr(sana, name)))
        self.transformer_blocks = nn.ModuleList(deepcopy(list(sana.transformer_blocks[:target_layer+1])))
        self.text_projection = deepcopy(text_projection)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, latent, text, timestep):
        condition = self.text_projection(text).expand(latent.shape[0], -1, -1)
        return c_sana_forward(self, latent, condition, timestep,
                              capture_layer=self.target_layer, stop_layer=self.target_layer)[1]

    def _adapted_state(self):
        return {key: value for key, value in self.state_dict().items()
                if "lora_" in key or key.startswith("text_projection.")}

    @torch.no_grad()
    def update_from_student(self, student_dit, text_projection, unused=None, decay=.999):
        sana = unwrap_sana_transformer(student_dit)
        source = {key: value for key, value in sana.state_dict().items() if "lora_" in key}
        source.update({"text_projection."+key: value for key, value in text_projection.state_dict().items()})
        for key, target in self._adapted_state().items():
            target.lerp_(source[key].detach().to(target), 1-float(decay))

    def get_ema_state_dict(self):
        return self._adapted_state()

    def load_ema_state_dict(self, state):
        if self._adapted_state().keys() != state.keys():
            raise ValueError("C teacher checkpoint keys do not match")
        self.load_state_dict(state, strict=False)
