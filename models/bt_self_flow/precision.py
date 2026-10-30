"""FP32 attention islands under AMP, without changing weights or topology.

diffusers 0.35.2's SANA processors cast inputs to float, but outer autocast
still casts their matmuls back to FP16. The 1e-15 normalization denominator
can then underflow to zero. Keep self-attention projections under AMP and
isolate their contractions/normalization. Cross-attention also needs FP32
projections: late SANA residuals are finite in FP32, but their query projection
can exceed FP16's range before scaled-dot-product attention. Used exclusively
by the BT models.
"""

import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import (
    SanaLinearAttnProcessor2_0, SanaMultiscaleAttnProcessor2_0, SanaMultiscaleLinearAttention,
)
from diffusers.models.transformers.sana_transformer import SanaAttnProcessor2_0
from diffusers.models.embeddings import PixArtAlphaTextProjection


# ReLU-gated linear attention can produce an exactly-zero normalization
# denominator for a token. The usual "+1e-15" keeps the forward output at zero
# but lets the backward divide by 1e-15, amplifying gradients by up to 1e15.
# Treat such degenerate tokens as contributing nothing instead: the forward is
# zero and the gradient through the division is masked.
_LINEAR_ATTENTION_DENOMINATOR_FLOOR = 1e-4

_ATTENTION_STATS = {}


def reset_attention_stats():
    _ATTENTION_STATS.clear()


def attention_stats():
    return {name: float(value) for name, value in _ATTENTION_STATS.items()}


def _record_stat(name, value, mode):
    value = value.detach().float()
    if name not in _ATTENTION_STATS:
        _ATTENTION_STATS[name] = value
    elif mode == "min":
        torch.minimum(_ATTENTION_STATS[name], value, out=_ATTENTION_STATS[name])
    else:
        torch.maximum(_ATTENTION_STATS[name], value, out=_ATTENTION_STATS[name])


def _attention_enabled():
    return os.environ.get("BT_ATTN_STATS") == "1"


class StableSanaCaptionProjection(PixArtAlphaTextProjection):
    """Reuse the original children and keys, retaining FP32 until caption_norm.

    Training text_proj can increase the unnormalized caption magnitude. Its
    second pretrained projection can exceed FP16 before the following RMSNorm
    brings the condition back to normal scale. Do not clamp or cast it down.
    """

    def __init__(self, projection):
        nn.Module.__init__(self)
        self.linear_1 = projection.linear_1
        self.act_1 = projection.act_1
        self.linear_2 = projection.linear_2
        self.train(projection.training)

    def forward(self, caption):
        with torch.autocast(device_type=caption.device.type, enabled=False):
            return super().forward(caption.float())


class StableSanaLinearAttention(SanaLinearAttnProcessor2_0):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        dtype = hidden_states.dtype
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query, key, value = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query = F.relu(query.transpose(1, 2).unflatten(1, (attn.heads, -1)))
        key = F.relu(key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3))
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            augmented = F.pad(value.float(), (0, 0, 0, 1), value=1.)
            product = (augmented @ key.float()) @ query.float()
            denominator = product[:, :, -1:]
            normalized = torch.where(
                denominator > _LINEAR_ATTENTION_DENOMINATOR_FLOOR,
                product[:, :, :-1] / (denominator + 1e-15),
                torch.zeros_like(product[:, :, :-1]),
            )
            if _attention_enabled():
                _record_stat("sana_denominator_absmin", denominator.abs().amin(), "min")
                _record_stat("sana_denominator_min", denominator.amin(), "min")
                _record_stat("sana_denominator_absmax", denominator.abs().amax(), "max")
                _record_stat("sana_ratio_absmax", normalized.abs().amax(), "max")
        output = normalized.flatten(1, 2).transpose(1, 2).to(dtype)
        output = attn.to_out[1](attn.to_out[0](output))
        return output.clamp(-65504, 65504) if dtype == torch.float16 else output


class StableSanaCrossAttention(SanaAttnProcessor2_0):
    """Preserve large finite queries rather than overflowing or clipping them.

    Call the native processor with autocast disabled, including the PEFT
    projections. Casting only an already overflowing query cannot repair it.
    Leave the result in FP32 for the following residual addition.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Memory-efficient CUDA backward reconstructs softmax from logsumexp;
        # large finite SANA queries can lose the subtraction's precision.
        # The math backend retains normalized probabilities for backward.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            context = None if encoder_hidden_states is None else encoder_hidden_states.float()
            return super().__call__(attn, hidden_states.float(), context, attention_mask)


class _FloatContractions:
    """A transient view of a DC-AE attention module, retaining its projections."""
    def __init__(self, attention):
        self.attention = attention

    def __getattr__(self, name):
        return getattr(self.attention, name)

    def apply_linear_attention(self, query, key, value):
        with torch.autocast(device_type=query.device.type, enabled=False):
            query, key, value = query.float(), key.float(), value.float()
            augmented = F.pad(value, (0, 0, 0, 1), value=1.)
            scores = torch.matmul(augmented, key.transpose(-1, -2))
            hidden = torch.matmul(scores, query)
            if _attention_enabled():
                denominator = hidden[:, :, -1:]
                _record_stat("vae_denominator_absmin", denominator.abs().amin(), "min")
                _record_stat("vae_denominator_min", denominator.amin(), "min")
                _record_stat("vae_denominator_absmax", denominator.abs().amax(), "max")
            denominator = hidden[:, :, -1:]
            numerator = hidden[:, :, :-1]
            hidden = torch.where(
                denominator > _LINEAR_ATTENTION_DENOMINATOR_FLOOR,
                numerator / (denominator + self.attention.eps),
                torch.zeros_like(numerator),
            )
            if _attention_enabled():
                _record_stat("vae_ratio_absmax", hidden.abs().amax(), "max")
            return hidden

    def apply_quadratic_attention(self, query, key, value):
        with torch.autocast(device_type=query.device.type, enabled=False):
            return self.attention.apply_quadratic_attention(query.float(), key.float(), value.float())


class StableDCMultiscaleAttention(SanaMultiscaleAttnProcessor2_0):
    def __call__(self, attn, hidden_states):
        return super().__call__(_FloatContractions(attn), hidden_states)


def install_stable_attention(sana=None, vae=None):
    if sana is not None:
        if not isinstance(sana.caption_projection, PixArtAlphaTextProjection):
            raise TypeError("BT expects the native SANA caption projection")
        if not isinstance(sana.caption_projection, StableSanaCaptionProjection):
            sana.caption_projection = StableSanaCaptionProjection(sana.caption_projection)
        for block in sana.transformer_blocks:
            if not isinstance(block.attn1.processor, SanaLinearAttnProcessor2_0):
                raise TypeError("BT expects the native SANA linear self-attention processor")
            block.attn1.set_processor(StableSanaLinearAttention())
            if block.attn2 is not None:
                if not isinstance(block.attn2.processor, SanaAttnProcessor2_0):
                    raise TypeError("BT expects the native SANA cross-attention processor")
                block.attn2.set_processor(StableSanaCrossAttention())
    if vae is not None:
        for module in vae.modules():
            if isinstance(module, SanaMultiscaleLinearAttention):
                if not isinstance(module.processor, SanaMultiscaleAttnProcessor2_0):
                    raise TypeError("BT expects the native DC-AE multiscale attention processor")
                module.processor = StableDCMultiscaleAttention()
