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


class RateRouter(nn.Module):
    def __init__(self, source_channels=32, feature_channels=320, width=128):
        super().__init__()
        self.width = width
        self.source = nn.Conv2d(4 * source_channels, width, 1)
        self.feature = nn.Conv2d(feature_channels, width, 1)
        self.cost = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, 4*width, dropout=0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, clean, feature, reference_cost):
        b, _, h, w = feature.shape
        source = F.pixel_unshuffle(clean, 2)
        if source.shape[-2:] != (h, w):
            raise ValueError("Router source and feature grids disagree")
        hidden = (self.source(source) + self.feature(feature)).flatten(2).transpose(1, 2)
        cost = torch.log1p(reference_cost.detach().clamp_min(0)).flatten(2).transpose(1, 2)
        hidden = hidden + self.cost(cost) + position_encoding(h, w, self.width, feature.device)
        raw = self.head(self.transformer(hidden)).transpose(1, 2).reshape(b, 1, h, w).tanh()
        continuous_codes = (raw + 1) * 3.5
        codes = ste_round(continuous_codes)
        allocation = codes * (2.0 / 7) - 1
        allocation = allocation - allocation.mean(dim=(-2, -1), keepdim=True)
        return allocation, codes.detach().to(torch.uint8)
