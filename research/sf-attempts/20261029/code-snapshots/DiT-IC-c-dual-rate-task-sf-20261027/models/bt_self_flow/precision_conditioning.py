"""One precision coordinate for analysis, allocation and synthesis.

These conditions never change the quantizer's step or add packet payloads.
The decoder reconstructs the local log-step from the existing q16/a3 fields.
"""

from torch import nn


class PrecisionEmbedding(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())

    def forward(self, log_step):
        # N1HW -> N128HW; the same MLP handles a global scalar or a local map.
        return self.network(log_step.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class PrecisionAffine(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        self.projection = nn.Conv2d(width, 2 * channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature, embedding):
        gamma, beta = self.projection(embedding).chunk(2, 1)
        return feature + gamma * feature + beta
