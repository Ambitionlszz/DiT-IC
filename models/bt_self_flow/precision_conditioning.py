"""One precision coordinate for analysis, allocation and synthesis.

These conditions never change the quantizer's step or add packet payloads.
The decoder reconstructs the local log-step from the existing q16/a3 fields.
"""

import torch
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


class PrecisionGeneratorConditioner(nn.Module):
    """Project the codec's shared precision embedding into native SANA units."""

    def __init__(self, hidden_dim, num_layers, embedding_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_projection = nn.Linear(embedding_dim, 6 * hidden_dim)
        self.output_projection = nn.Linear(embedding_dim, 2 * hidden_dim)
        self.layer_gain = nn.Parameter(torch.ones(num_layers, 6))
        for projection in (self.block_projection, self.output_projection):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, embedding_tokens):
        b, n, _ = embedding_tokens.shape
        block = self.block_projection(embedding_tokens).reshape(b, n, 6, self.hidden_dim)
        output = self.output_projection(embedding_tokens).reshape(b, n, 2, self.hidden_dim)
        return block, output

    def block_residual(self, modulation, layer):
        return modulation * self.layer_gain[layer].view(1, 1, 6, 1)
