"""SANA/quantization-conditioned PiT updates before the native AE RGB head.

PiT, RMSNorm, rotary attention and sinusoidal conventions are adapted from
NVlabs/PixelDiT, commit f7d7c35f71893d81c6e71d2576ea80601a1e8529.
The optional four-way post-modulation follows upstream PR #8, merge commit
f5ed1f6014c93def68bba38866b8da4077e6f678.
See third_party/PixelDiT/LICENSE for the upstream research/evaluation license.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value):
        dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype)


def pixel_position(height, width, channels, device, dtype):
    """Upstream x-sin/x-cos/y-sin/y-cos convention, including rectangular grids."""
    frequency = 10000. ** (-torch.arange(channels // 4, device=device, dtype=torch.float64)
                            / (channels // 4))
    x = torch.arange(width, device=device, dtype=torch.float64)[:, None] * frequency
    y = torch.arange(height, device=device, dtype=torch.float64)[:, None] * frequency
    x = torch.cat((x.sin(), x.cos()), -1).to(dtype)
    y = torch.cat((y.sin(), y.cos()), -1).to(dtype)
    return torch.cat((x[None].expand(height, -1, -1), y[:, None].expand(-1, width, -1)), -1)


def rotary_position(head_dim, height, width, device):
    # Match PiT's normalized 0..16 coordinates and interleaved x/y complex pairs.
    frequency = 1. / (10000. ** (torch.arange(0, head_dim, 4, device=device).float() / head_dim))
    y, x = torch.meshgrid(torch.linspace(0, 16, height, device=device),
                          torch.linspace(0, 16, width, device=device), indexing="ij")
    x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.stack((torch.polar(torch.ones_like(x), x), torch.polar(torch.ones_like(y), y)), -1).flatten(1)


def apply_rotary(query, key, position):
    with torch.autocast(device_type=query.device.type, enabled=False):
        def rotate(value):
            paired = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
            return torch.view_as_real(paired * position[None, :, None]).flatten(-2).to(value.dtype)
        return rotate(query), rotate(key)


class RotaryAttention(nn.Module):
    def __init__(self, width=256, heads=4):
        super().__init__()
        if width % heads or (width // heads) % 4:
            raise ValueError("PiT attention requires a head dimension divisible by four")
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width)

    def forward(self, tokens, position):
        batch, count, width = tokens.shape
        query, key, value = self.qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dim).unbind(2)
        query, key = apply_rotary(self.q_norm(query), self.k_norm(key), position)
        attended = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
                                                  value.transpose(1, 2), dropout_p=0.)
        return self.proj(attended.transpose(1, 2).reshape(batch, count, width))


class PiTBlock(nn.Module):
    def __init__(self, pixel_dim=16, context_dim=256, patch_size=16, attention_dim=256, heads=4,
                 post_modulation=False):
        super().__init__()
        self.pixel_dim, self.patch_size = pixel_dim, patch_size
        self.norm1, self.norm2 = RMSNorm(pixel_dim), RMSNorm(pixel_dim)
        self.compress_to_attn = nn.Linear(patch_size ** 2 * pixel_dim, attention_dim)
        self.attn = RotaryAttention(attention_dim, heads)
        self.expand_from_attn = nn.Linear(attention_dim, patch_size ** 2 * pixel_dim)
        self.mlp = nn.Sequential(nn.Linear(pixel_dim, 4 * pixel_dim), nn.GELU(),
                                 nn.Linear(4 * pixel_dim, pixel_dim))
        self.post_modulation = bool(post_modulation)
        groups = 4 if self.post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(context_dim, patch_size ** 2 * groups * pixel_dim))

    def forward(self, pixels, condition, position):
        batch, count, area, width = pixels.shape
        if self.post_modulation:
            # Preserve upstream [region, pixel, parameter group, channel] layout.
            scale_a, shift_a, scale_m, shift_m = self.adaLN_modulation(condition).reshape(
                batch, count, area, 4 * width).chunk(4, -1)
            compact = self.compress_to_attn(self.norm1(pixels).flatten(-2))
            expanded = self.expand_from_attn(self.attn(compact, position)).reshape_as(pixels)
            pixels = pixels + (1 + scale_a) * expanded + shift_a
            return pixels + (1 + scale_m) * self.mlp(self.norm2(pixels)) + shift_m
        # Upstream ordering is [patch, pixel, six groups, channel].
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(condition).reshape(
            batch, count, area, 6 * width).chunk(6, -1)
        normalized = self.norm1(pixels) * (1 + scale_a) + shift_a
        compact = self.compress_to_attn(normalized.flatten(-2))
        attended = self.attn(compact, position)
        expanded = self.expand_from_attn(attended).reshape_as(pixels)
        pixels = pixels + gate_a * expanded
        return pixels + gate_m * self.mlp(self.norm2(pixels) * (1 + scale_m) + shift_m)


class ScaleEmbedding(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, log2_scale):
        with torch.autocast(device_type=log2_scale.device.type, enabled=False):
            frequency = torch.exp(-math.log(10) * torch.arange(self.width // 2, device=log2_scale.device).float()
                                  / (self.width // 2))
            angles = log2_scale.float()[..., None] * frequency
            embedding = torch.cat((angles.cos(), angles.sin()), -1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class PixelDecoderAdapter(nn.Module):
    def __init__(self, feature_channels, sana_dim, depth=2, pixel_dim=16, patch_size=16,
                 context_dim=256, attention_dim=256, heads=4, use_quantization=True,
                 delta_ref=1., checkpoint_blocks=True):
        super().__init__()
        sizes = (feature_channels, sana_dim, depth, pixel_dim, patch_size, context_dim, attention_dim, heads)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in sizes):
            raise ValueError("PiT dimensions and depth must be positive integers")
        if pixel_dim % 4 or context_dim % 2 or not math.isfinite(delta_ref) or delta_ref <= 0:
            raise ValueError("invalid PiT position width or reference step")
        self.patch_size, self.pixel_dim = patch_size, pixel_dim
        self.use_quantization = bool(use_quantization)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.register_buffer("delta_ref", torch.tensor(float(delta_ref)))
        self.content_norm = RMSNorm(sana_dim)
        self.content_projection = nn.Linear(sana_dim, context_dim)
        self.scale_embedding = ScaleEmbedding(context_dim)
        self.pixel_input = nn.Linear(feature_channels, pixel_dim)
        self.blocks = nn.ModuleList(PiTBlock(pixel_dim, context_dim, patch_size, attention_dim, heads)
                                    for _ in range(depth))
        self.pixel_output = nn.Linear(pixel_dim, feature_channels)
        nn.init.zeros_(self.pixel_output.weight)
        nn.init.zeros_(self.pixel_output.bias)

    def condition(self, sana_tokens, token_hw, delta, region_hw):
        batch, count, _ = sana_tokens.shape
        if len(token_hw) != 2 or count != token_hw[0] * token_hw[1]:
            raise ValueError("SANA tokens must retain their actual two-dimensional grid")
        if delta.ndim != 4 or delta.shape[:2] != (batch, 1):
            raise ValueError("effective delta must have shape [B,1,Hq,Wq]")
        if not torch.isfinite(delta).all() or not (delta > 0).all():
            raise ValueError("effective quantization steps must be finite and positive")
        rh, rw = region_hw
        dh, dw = delta.shape[-2:]
        if rh % dh or rw % dw:
            raise ValueError("each quantization cell must cover whole PiT regions")
        content = self.content_projection(self.content_norm(sana_tokens))
        content = content.transpose(1, 2).reshape(batch, -1, *token_hw)
        content = F.interpolate(content, size=region_hw, mode="bilinear", align_corners=False)
        content = content.flatten(2).transpose(1, 2)
        log2_scale = torch.log2(delta.float() / self.delta_ref)
        if not self.use_quantization:
            log2_scale = torch.zeros_like(log2_scale)
        log2_scale = log2_scale.repeat_interleave(rh // dh, -2).repeat_interleave(rw // dw, -1)
        scale = self.scale_embedding(log2_scale.flatten(1))
        return F.silu(content + scale)

    def forward(self, features, sana_tokens, token_hw, delta, return_features=False, checkpoint_blocks=None):
        batch, channels, height, width = features.shape
        patch = self.patch_size
        if height % patch or width % patch:
            raise ValueError("AE output grid must be divisible by the PiT patch size")
        rh, rw = height // patch, width // patch
        condition = self.condition(sana_tokens, token_hw, delta, (rh, rw))
        return self.apply_condition(features, condition, return_features, checkpoint_blocks)

    def apply_condition(self, features, condition, return_features=False, checkpoint_blocks=None):
        batch, _, height, width = features.shape
        patch = self.patch_size
        rh, rw = height // patch, width // patch
        pixels = self.pixel_input(features.movedim(1, -1))
        pixels = pixels + pixel_position(height, width, self.pixel_dim, pixels.device, features.dtype)
        pixels = pixels.reshape(batch, rh, patch, rw, patch, self.pixel_dim).permute(0, 1, 3, 2, 4, 5)
        pixels = pixels.reshape(batch, rh * rw, patch * patch, self.pixel_dim)
        position = rotary_position(self.blocks[0].attn.head_dim, rh, rw, features.device)
        recompute = self.checkpoint_blocks if checkpoint_blocks is None else checkpoint_blocks
        for block in self.blocks:
            if recompute and torch.is_grad_enabled():
                pixels = checkpoint(block, pixels, condition, position, use_reentrant=False)
            else:
                pixels = block(pixels, condition, position)
        grid = pixels.reshape(batch, rh, rw, patch, patch, self.pixel_dim).permute(0, 1, 3, 2, 4, 5)
        grid = grid.reshape(batch, height, width, self.pixel_dim)
        if hasattr(self, "output_norm"):
            grid = self.output_norm(grid)
        updated = features + self.pixel_output(grid).movedim(-1, 1)
        if return_features:
            return updated, {"pixel_tokens": pixels, "token_hw": (rh, rw)}
        return updated


class ReceivedUpdatePixelAdapter(PixelDecoderAdapter):
    """Separate received content/update channels, precision-gated update, post-PiT.

    The shared precision embedding and fixed s_data are supplied by C on every
    call, so this adapter owns neither a second precision coordinate nor a copy
    of the calibrated scale. No source latent or raw SANA hidden is consumed.
    """

    def __init__(self, feature_channels, latent_channels, precision_dim=128, depth=2,
                 pixel_dim=16, patch_size=16, context_dim=256, attention_dim=256,
                 heads=4, checkpoint_blocks=True):
        nn.Module.__init__(self)
        sizes = (feature_channels, latent_channels, precision_dim, depth, pixel_dim,
                 patch_size, context_dim, attention_dim, heads)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in sizes):
            raise ValueError("PiT dimensions and depth must be positive integers")
        if pixel_dim % 4 or context_dim % 2:
            raise ValueError("invalid PiT pixel or condition width")
        self.patch_size, self.pixel_dim = patch_size, pixel_dim
        self.checkpoint_blocks = bool(checkpoint_blocks)
        branch_dim = context_dim // 2
        self.mu_projection = nn.Conv2d(latent_channels, branch_dim, 1, bias=False)
        self.r_projection = nn.Conv2d(latent_channels, branch_dim, 1, bias=False)
        self.gate_mlp = nn.Sequential(nn.Conv2d(precision_dim, precision_dim, 1), nn.SiLU(),
                                      nn.Conv2d(precision_dim, branch_dim, 1))
        self.pixel_input = nn.Linear(feature_channels, pixel_dim)
        self.blocks = nn.ModuleList(
            PiTBlock(pixel_dim, context_dim, patch_size, attention_dim, heads, post_modulation=True)
            for _ in range(depth))
        self.output_norm = RMSNorm(pixel_dim)
        self.pixel_output = nn.Linear(pixel_dim, feature_channels)
        nn.init.xavier_uniform_(self.mu_projection.weight)
        nn.init.xavier_uniform_(self.r_projection.weight)
        for layer in (self.gate_mlp[-1], self.pixel_output,
                      *(block.adaLN_modulation[-1] for block in self.blocks)):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def condition_parts(self, mu, update, precision_features, s_data, region_hw):
        if mu.ndim != 4 or mu.shape != update.shape or mu.shape[1] != self.mu_projection.in_channels:
            raise ValueError("mu and actual scheduler update must share the canonical latent grid")
        batch = mu.shape[0]
        if (precision_features.ndim != 4 or precision_features.shape[0] != batch
                or precision_features.shape[1] != self.gate_mlp[0].in_channels):
            raise ValueError("PiT requires C's shared spatial precision embedding")
        if s_data.numel() != 1 or not torch.isfinite(s_data).all() or not (s_data > 0).all():
            raise ValueError("PiT requires C's positive fixed s_data")
        rh, rw = region_hw
        dh, dw = precision_features.shape[-2:]
        if rh < 1 or rw < 1 or rh % dh or rw % dw:
            raise ValueError("each quantization cell must cover whole PiT regions")
        # Fixed normalization retains cross-image/position update amplitude.
        base = self.mu_projection(mu / s_data)
        residual = self.r_projection(update / s_data)
        base = F.interpolate(base, size=region_hw, mode="bilinear", align_corners=False)
        residual = F.interpolate(residual, size=region_hw, mode="bilinear", align_corners=False)
        gate = 1 + .5 * torch.tanh(self.gate_mlp(precision_features))
        gate = F.interpolate(gate, size=region_hw, mode="nearest")
        return base, residual, gate

    def condition(self, mu, update, precision_features, s_data, region_hw):
        base, residual, gate = self.condition_parts(mu, update, precision_features, s_data, region_hw)
        return F.silu(torch.cat((base, gate * residual), dim=1)).flatten(2).transpose(1, 2)

    def forward(self, features, mu, update, precision_features, s_data,
                return_features=False, checkpoint_blocks=None):
        if features.ndim != 4 or features.shape[0] != mu.shape[0]:
            raise ValueError("PiT requires batch-aligned AE features and receiver conditions")
        height, width = features.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("AE output grid must be divisible by the PiT patch size")
        condition = self.condition(mu, update, precision_features, s_data,
                                   (height // self.patch_size, width // self.patch_size))
        return self.apply_condition(features, condition, return_features, checkpoint_blocks)
