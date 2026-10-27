import math

import torch
from torch import nn
from torch.nn import functional as F


def ste_round(value):
    # Exactly integral forward values, including at large magnitudes.
    return value.round().detach() + (value - value.detach())


def allocation_from_codes(codes):
    values = codes.float() * (2.0 / 7) - 1
    return values - values.mean(dim=(-2, -1), keepdim=True)


def position_encoding(height, width, channels, device):
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    frequency = torch.exp(torch.arange(channels // 4, device=device)
                          * (-math.log(10000) / max(channels // 4 - 1, 1)))
    angles = [axis.flatten()[:, None] * frequency for axis in (y, x)]
    return torch.cat([f(a) for a in angles for f in (torch.sin, torch.cos)], dim=-1)


class SourcePrecisionPlanner(nn.Module):
    """Choose a hard precision field before the conditional analysis transform."""

    def __init__(self, source_channels=32, width=128):
        super().__init__()
        self.width = width
        self.source = nn.Conv2d(4 * source_channels, width, 1)
        self.source_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.precision_projections = nn.ModuleList(nn.Linear(width, 4 * width) for _ in range(2))
        for projection in self.precision_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.constant_(self.head[-1].bias, math.atanh(1 / 7))

    def forward(self, clean, precision):
        b, _, height, width = clean.shape
        if height % 2 or width % 2 or precision.shape != (b, self.width, 1, 1):
            raise ValueError("source planner requires an even source grid and global precision embedding")
        source = self.source(F.pixel_unshuffle(clean, 2))
        h, w = source.shape[-2:]
        hidden = self.source_norm(source.flatten(2).transpose(1, 2))
        hidden = hidden + position_encoding(h, w, self.width, clean.device)
        condition = precision.flatten(2).transpose(1, 2)
        for layer, projection in zip(self.transformer.layers, self.precision_projections):
            gamma_a, beta_a, gamma_f, beta_f = projection(condition).chunk(4, -1)
            normalized = (1 + gamma_a) * layer.norm1(hidden) + beta_a
            attended = layer.self_attn(normalized, normalized, normalized, need_weights=False)[0]
            hidden = hidden + layer.dropout1(attended)
            normalized = (1 + gamma_f) * layer.norm2(hidden) + beta_f
            hidden = hidden + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized)))))
        continuous = (self.head(hidden).transpose(1, 2).reshape(b, 1, h, w).tanh() + 1) * 3.5
        codes = ste_round(continuous)
        return allocation_from_codes(codes), codes.detach().to(torch.uint8)


class RateRouter(nn.Module):
    def __init__(self, source_channels=32, feature_channels=320, width=128, precision_conditioning=False):
        super().__init__()
        self.width = width
        self.precision_conditioning = bool(precision_conditioning)
        self.source = nn.Conv2d(4 * source_channels, width, 1)
        self.feature = nn.Conv2d(feature_channels, width, 1)
        layer = nn.TransformerEncoderLayer(width, 4, 4*width, dropout=0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if self.precision_conditioning:
            self.source_norm, self.feature_norm = nn.LayerNorm(width), nn.LayerNorm(width)
            self.fusion = nn.Sequential(nn.Linear(2*width, 2*width), nn.SiLU(), nn.Linear(2*width, width))
            self.precision_projections = nn.ModuleList(nn.Linear(width, 4*width) for _ in range(2))
            for projection in self.precision_projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            self.head = nn.Sequential(nn.Linear(width+1, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.normal_(self.head[-1].weight, std=.001)
            # s=4 is a bin centre. s=3.5 would start on the 3/4 boundary.
            nn.init.constant_(self.head[-1].bias, math.atanh(1/7))
        else:
            self.cost = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, clean, feature, reference_cost, precision=None):
        b, _, h, w = feature.shape
        source = F.pixel_unshuffle(clean, 2)
        if source.shape[-2:] != (h, w):
            raise ValueError("Router source and feature grids disagree")
        cost = torch.log1p(reference_cost.detach().clamp_min(0)).flatten(2).transpose(1, 2)
        source = self.source(source).flatten(2).transpose(1, 2)
        feature = self.feature(feature).flatten(2).transpose(1, 2)
        if self.precision_conditioning:
            if precision is None or precision.shape != (b, self.width, 1, 1):
                raise ValueError("precision-conditioned Router requires a global precision embedding")
            source, feature = self.source_norm(source), self.feature_norm(feature)
            hidden = feature + self.fusion(torch.cat([feature, source], -1))
            hidden = hidden + position_encoding(h, w, self.width, feature.device)
            condition = precision.flatten(2).transpose(1, 2)
            for layer, projection in zip(self.transformer.layers, self.precision_projections):
                gamma_a, beta_a, gamma_f, beta_f = projection(condition).chunk(4, -1)
                normalized = (1+gamma_a)*layer.norm1(hidden)+beta_a
                attended = layer.self_attn(normalized, normalized, normalized, need_weights=False)[0]
                hidden = hidden + layer.dropout1(attended)
                normalized = (1+gamma_f)*layer.norm2(hidden)+beta_f
                hidden = hidden + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized)))))
            raw = self.head(torch.cat([hidden, cost], -1))
        else:
            hidden = source + feature + self.cost(cost) + position_encoding(h, w, self.width, feature.device)
            raw = self.head(self.transformer(hidden))
        raw = raw.transpose(1, 2).reshape(b, 1, h, w).tanh()
        continuous_codes = (raw + 1) * 3.5
        codes = ste_round(continuous_codes)
        allocation = codes * (2.0 / 7) - 1
        allocation = allocation - allocation.mean(dim=(-2, -1), keepdim=True)
        return allocation, codes.detach().to(torch.uint8)
