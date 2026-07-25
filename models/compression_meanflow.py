"""AdaLN-conditioned Compression MeanFlow utilities.

The MeanFlow bridge connects a clean VAE latent to its entropy-decoded
compression latent. The endpoints are detached inside ``compute_loss`` so the
auxiliary identity trains the reconstruction operator without directly
changing the codec through the bridge.
"""

import math
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


MEANFLOW_MODE = "meanflow"
ADALN_CONDITIONING = "adaln"
MEANFLOW_MODE_ID = 0
ADALN_CONDITIONING_ID = 1


def _math_sdpa_context():
    """Force the SDPA implementation that supports forward-mode AD."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda,
            "sdp_kernel",
        ):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        return nullcontext()


class MeanFlowTimestepEmbedder(nn.Module):
    """Reference-size sinusoidal scalar embedding followed by a two-layer MLP."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or frequency_embedding_size < 2:
            raise ValueError(
                "MeanFlow hidden size must be positive and frequency "
                "embedding size must be at least 2"
            )

        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError("MeanFlow timesteps must be one-dimensional")
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32,
            )
            / half
        )
        angles = timestep[:, None].to(torch.float32) * frequencies[None]
        embedding = torch.cat(
            (torch.cos(angles), torch.sin(angles)),
            dim=-1,
        )
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])),
                dim=-1,
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(
            timestep,
            self.frequency_embedding_size,
        )
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class AdaLNIntervalAdapter(nn.Module):
    """Embed MeanFlow time ``t`` and interval length ``h=t-r`` independently."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.t_embedder = MeanFlowTimestepEmbedder(
            hidden_size,
            frequency_embedding_size,
        )
        self.h_embedder = MeanFlowTimestepEmbedder(
            hidden_size,
            frequency_embedding_size,
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1 or h.ndim != 1 or t.shape != h.shape:
            raise ValueError(
                "t and h must be one-dimensional tensors with matching shapes"
            )
        return self.t_embedder(t) + self.h_embedder(h)


class SanaAdaLNConditioningEmbedding(nn.Module):
    """Add a precomputed MeanFlow condition to SANA's native time embedding."""

    def __init__(self, base_embedding: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("SANA AdaLN embedding dimension must be positive")
        self.base_embedding = base_embedding
        self.embedding_dim = int(embedding_dim)

    def forward(
        self,
        packed_timestep: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        expected_width = 1 + self.embedding_dim
        if (
            packed_timestep.ndim != 2
            or packed_timestep.shape[1] != expected_width
        ):
            raise ValueError(
                f"Packed SANA timestep must have shape [batch, {expected_width}]"
            )
        base_timestep = packed_timestep[:, 0]
        interval_embedding = packed_timestep[:, 1:]
        base_embedding = self.base_embedding(
            base_timestep,
            *args,
            **kwargs,
        )
        return base_embedding + interval_embedding.to(
            device=base_embedding.device,
            dtype=base_embedding.dtype,
        )


class CompressionMeanFlow(nn.Module):
    """Compression MeanFlow 5A with AdaLN-only interval conditioning."""

    def __init__(
        self,
        mode: str = MEANFLOW_MODE,
        conditioning_mode: str = ADALN_CONDITIONING,
        adaln_embedding_dim: int = 1152,
        frequency_embedding_size: int = 256,
        equal_time_ratio: float = 0.5,
        time_distribution: str = "logit_normal",
        logit_mean: float = -0.4,
        logit_std: float = 1.0,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
    ) -> None:
        super().__init__()
        if mode != MEANFLOW_MODE:
            raise ValueError(
                "Pure SANA supports only MeanFlow 5A mode='meanflow'"
            )
        if conditioning_mode != ADALN_CONDITIONING:
            raise ValueError(
                "Pure SANA MeanFlow supports only conditioning_mode='adaln'"
            )
        if not 0.0 <= equal_time_ratio <= 1.0:
            raise ValueError("equal_time_ratio must be in [0, 1]")
        if time_distribution not in ("logit_normal", "uniform"):
            raise ValueError(
                "time_distribution must be 'logit_normal' or 'uniform'"
            )
        if logit_std <= 0:
            raise ValueError("logit_std must be positive")
        if norm_p < 0 or norm_eps <= 0:
            raise ValueError(
                "norm_p must be non-negative and norm_eps must be positive"
            )

        self.mode = mode
        self.conditioning_mode = conditioning_mode
        self.equal_time_ratio = float(equal_time_ratio)
        self.time_distribution = time_distribution
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.norm_p = float(norm_p)
        self.norm_eps = float(norm_eps)
        self.register_buffer(
            "training_mode_id",
            torch.tensor(MEANFLOW_MODE_ID, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "conditioning_mode_id",
            torch.tensor(ADALN_CONDITIONING_ID, dtype=torch.int64),
            persistent=True,
        )
        self.adaln_interval_adapter = AdaLNIntervalAdapter(
            adaln_embedding_dim,
            frequency_embedding_size,
        )

    def condition_adaln(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Return the global ``(t, t-r)`` condition for SANA's AdaLN path."""
        return self.adaln_interval_adapter(t, t - r)

    def _sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.time_distribution == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        samples = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.sigmoid(
            samples * self.logit_std + self.logit_mean
        )

    def sample_tr(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ordered pairs ``r <= t`` and force ``r=t`` by Bernoulli mask."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        first = self._sample_time(batch_size, device, dtype)
        second = self._sample_time(batch_size, device, dtype)
        t = torch.maximum(first, second)
        r = torch.minimum(first, second)

        if self.equal_time_ratio > 0:
            equal_mask = (
                torch.rand(batch_size, device=device)
                < self.equal_time_ratio
            )
            r = torch.where(equal_mask, t, r)
        return t, r

    @staticmethod
    def _expand_batch(
        values: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return values.reshape(
            values.shape[0],
            *([1] * (reference.ndim - 1)),
        )

    def build_target(
        self,
        velocity: torch.Tensor,
        du_dt: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        correction = self._expand_batch(t - r, velocity) * du_dt
        target = velocity - correction
        return target.detach(), correction.detach()

    def compute_loss(
        self,
        velocity_fn: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ],
        clean_latent: torch.Tensor,
        compressed_latent: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate the MeanFlow identity on a detached compression bridge.

        ``velocity_fn`` receives latent, ``t``, ``r`` and indices into the
        original batch. It returns ``sigma_0 * SANA(...)``.
        """
        if clean_latent.shape != compressed_latent.shape:
            raise ValueError(
                "clean_latent and compressed_latent must have identical shapes"
            )
        if clean_latent.ndim < 2:
            raise ValueError(
                "latents must include a batch and at least one feature dimension"
            )
        if not hasattr(torch, "func") or not hasattr(torch.func, "jvp"):
            raise RuntimeError(
                "Compression MeanFlow requires torch.func.jvp "
                "(PyTorch >= 2.0)"
            )

        clean = clean_latent.detach()
        compressed = compressed_latent.detach()
        if clean.dtype != torch.float32:
            raise RuntimeError(
                "Compression MeanFlow requires float32 latents for stable JVPs"
            )
        velocity = compressed - clean
        batch_size = clean.shape[0]

        if t is None or r is None:
            if t is not None or r is not None:
                raise ValueError(
                    "t and r must both be provided or both be omitted"
                )
            t, r = self.sample_tr(
                batch_size,
                clean.device,
                clean.dtype,
            )
        else:
            if t.shape != (batch_size,) or r.shape != (batch_size,):
                raise ValueError("t and r must have shape [batch]")
            t = t.to(device=clean.device, dtype=clean.dtype)
            r = r.to(device=clean.device, dtype=clean.dtype)
            if torch.any(r > t):
                raise ValueError("MeanFlow requires r <= t")

        bridge_shape = (batch_size,) + (1,) * (clean.ndim - 1)
        t_view = t.reshape(bridge_shape)
        z_t = (1.0 - t_view) * clean + t_view * compressed

        nonzero_indices = torch.nonzero(
            t > r,
            as_tuple=False,
        ).flatten()
        equal_indices = torch.nonzero(
            t == r,
            as_tuple=False,
        ).flatten()
        prediction = torch.zeros_like(z_t)
        du_dt = torch.zeros_like(z_t)
        device_type = z_t.device.type

        if nonzero_indices.numel() > 0:
            z_nonzero = z_t.index_select(0, nonzero_indices)
            t_nonzero = t.index_select(0, nonzero_indices)
            r_nonzero = r.index_select(0, nonzero_indices)
            velocity_nonzero = velocity.index_select(
                0,
                nonzero_indices,
            )

            def nonzero_velocity_fn(z_value, t_value, r_value):
                return velocity_fn(
                    z_value,
                    t_value,
                    r_value,
                    nonzero_indices,
                )

            with _math_sdpa_context():
                with torch.autocast(
                    device_type=device_type,
                    enabled=False,
                ):
                    prediction_nonzero, du_dt_nonzero = torch.func.jvp(
                        nonzero_velocity_fn,
                        (z_nonzero, t_nonzero, r_nonzero),
                        (
                            velocity_nonzero,
                            torch.ones_like(t_nonzero),
                            torch.zeros_like(r_nonzero),
                        ),
                    )
            prediction = prediction.index_copy(
                0,
                nonzero_indices,
                prediction_nonzero,
            )
            du_dt = du_dt.index_copy(
                0,
                nonzero_indices,
                du_dt_nonzero,
            )

        if equal_indices.numel() > 0:
            with torch.autocast(
                device_type=device_type,
                enabled=False,
            ):
                prediction_equal = velocity_fn(
                    z_t.index_select(0, equal_indices),
                    t.index_select(0, equal_indices),
                    r.index_select(0, equal_indices),
                    equal_indices,
                )
            prediction = prediction.index_copy(
                0,
                equal_indices,
                prediction_equal,
            )

        target, correction = self.build_target(
            velocity,
            du_dt,
            t,
            r,
        )
        squared_error = (prediction - target).square()
        reduce_dims = tuple(range(1, squared_error.ndim))
        per_sample_error = squared_error.sum(dim=reduce_dims)
        adaptive_weight = (
            per_sample_error.detach() + self.norm_eps
        ).pow(self.norm_p)
        loss = (per_sample_error / adaptive_weight).mean()

        velocity_norm = velocity.flatten(1).norm(dim=1)
        correction_norm = correction.flatten(1).norm(dim=1)
        correction_ratio = correction_norm / (velocity_norm + 1e-8)
        nonzero_mask = ((t - r) > 0).to(correction_ratio.dtype)
        conditional_ratio = (
            (correction_ratio * nonzero_mask).sum()
            / nonzero_mask.sum().clamp_min(1.0)
        )
        stats = {
            "loss": loss.detach(),
            "raw_mse": squared_error.mean().detach(),
            "correction_ratio": conditional_ratio.detach(),
            "correction_ratio_all": correction_ratio.mean().detach(),
            "mean_interval": (t - r).mean().detach(),
            "nonzero_interval_fraction": nonzero_mask.mean().detach(),
        }
        return loss, stats
