from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from compressai.layers import AttentionBlock


def _conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


class _ResidualBottleneck(nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                channels // 2,
                channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.branch(x)


class ELICAnalysisEncoder(nn.Module):
    """Frozen ELIC g_a used by the historical AuxE path."""

    def __init__(self, channels=192, latent_channels=320):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            _conv(3, channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            _conv(channels, channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            AttentionBlock(channels),
            _conv(channels, channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            _ResidualBottleneck(channels),
            _conv(channels, latent_channels),
            AttentionBlock(latent_channels),
        )

    def forward(self, image):
        return self.analysis_transform(image)


def load_elic_analysis_encoder(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"ELIC checkpoint does not exist: {path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"ELIC checkpoint must contain a state dict: {path}")
    if isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]

    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("g_a."):
            normalized[key[len("g_a."):]] = value

    if not normalized:
        normalized = dict(state)

    encoder = ELICAnalysisEncoder()
    encoder.load_state_dict(normalized, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder
