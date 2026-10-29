import math

import torch
from torch import nn


class ContinuousDistortionConditioner(nn.Module):
    def __init__(self, hidden_dim, num_layers, embedding_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, -math.log(10000), embedding_dim//2)))
        self.in_projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.block_projection = nn.Linear(hidden_dim, 6*hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 2*hidden_dim)
        self.layer_gain = nn.Parameter(torch.ones(num_layers, 6))
        for projection in (self.block_projection, self.output_projection):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, log_rho_tokens):
        if log_rho_tokens.ndim != 2 or not log_rho_tokens.is_floating_point():
            raise ValueError("continuous conditioning requires floating [batch, tokens]")
        angle = log_rho_tokens[..., None] * self.frequencies
        embedding = torch.cat([angle.sin(), angle.cos()], -1)
        hidden = self.activation(self.in_projection(embedding))
        b, n, _ = hidden.shape
        return (self.block_projection(hidden).reshape(b, n, 6, self.hidden_dim),
                self.output_projection(hidden).reshape(b, n, 2, self.hidden_dim))

    def block_residual(self, modulation, layer):
        return modulation * self.layer_gain[layer].view(1, 1, 6, 1)


def residual_update(coarse, prediction, scale, reference_scale, transition):
    transition = float(transition)
    if not 0 <= transition <= 1:
        raise ValueError("repair transition must be in [0, 1]")
    coefficient = (1-transition)*reference_scale + transition*scale.detach()
    return coarse - coefficient * prediction
