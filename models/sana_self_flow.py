from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn


def mix_dual_rate_tokens(view_a, view_b, quality_a, quality_b, mask):
    if view_a.shape != view_b.shape:
        raise ValueError("codec views must have identical latent grids")
    expected_mask = (view_a.shape[0], 1, view_a.shape[2], view_a.shape[3])
    if mask.shape != expected_mask or mask.dtype != torch.bool:
        raise ValueError(
            f"token mask must be bool with shape {expected_mask}"
        )
    mixed = torch.where(mask, view_b, view_a)
    quality_map = torch.where(
        mask[:, 0],
        quality_b[:, None, None],
        quality_a[:, None, None],
    )
    return mixed, quality_map


def restrict_mask_to_asymmetric_samples(mask, asymmetric_samples):
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.dtype != torch.bool:
        raise ValueError(
            "token mask must be bool with shape [batch, 1, height, width]"
        )
    if asymmetric_samples.shape != (mask.shape[0],):
        raise ValueError("asymmetric_samples must have shape [batch]")
    if asymmetric_samples.dtype != torch.bool:
        raise TypeError("asymmetric_samples must use torch.bool")
    return mask & asymmetric_samples[:, None, None, None]


def unwrap_sana_transformer(model):
    """Return the SANA module while preserving PEFT-wrapped LoRA layers."""
    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(candidate, "transformer_blocks"):
        raise TypeError("expected a SanaTransformer2DModel-compatible module")
    return candidate


class TokenQualityConditioner(nn.Module):
    """Map a per-token codec quality index to SANA AdaLN residuals."""

    def __init__(self, num_levels, hidden_dim, num_layers, embedding_dim=128):
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2")
        self.num_levels = int(num_levels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.embedding = nn.Embedding(self.num_levels, embedding_dim)
        self.in_projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.block_projection = nn.Linear(hidden_dim, 6 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.layer_gain = nn.Parameter(torch.ones(num_layers, 6))
        self._initialize()

    def _initialize(self):
        positions = torch.arange(
            self.num_levels, dtype=self.embedding.weight.dtype
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                self.embedding.embedding_dim,
                2,
                dtype=self.embedding.weight.dtype,
            )
            * (-torch.log(torch.tensor(10000.0)) / self.embedding.embedding_dim)
        )
        with torch.no_grad():
            self.embedding.weight[:, 0::2] = torch.sin(positions * frequencies)
            self.embedding.weight[:, 1::2] = torch.cos(
                positions * frequencies[: self.embedding.weight[:, 1::2].shape[1]]
            )
        nn.init.zeros_(self.block_projection.weight)
        nn.init.zeros_(self.block_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, quality_tokens):
        if quality_tokens.ndim != 2:
            raise ValueError("quality_tokens must have shape [batch, tokens]")
        if quality_tokens.dtype != torch.long:
            raise TypeError("quality_tokens must use torch.long")
        if torch.any(quality_tokens < 0).item() or torch.any(
            quality_tokens >= self.num_levels
        ).item():
            raise ValueError("quality token is outside the configured range")
        hidden = self.activation(self.in_projection(self.embedding(quality_tokens)))
        batch, tokens, _ = hidden.shape
        block = self.block_projection(hidden).reshape(
            batch, tokens, 6, self.hidden_dim
        )
        output = self.output_projection(hidden).reshape(
            batch, tokens, 2, self.hidden_dim
        )
        return block, output

    def block_residual(self, block_modulation, layer_index):
        gain = self.layer_gain[layer_index].view(1, 1, 6, 1)
        return block_modulation * gain


class RepresentationProjector(nn.Module):
    """Student-only residual MLP used to align shallow and deep features."""

    def __init__(self, hidden_dim, expansion=2):
        super().__init__()
        inner_dim = int(hidden_dim * expansion)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states):
        normalized = self.norm(hidden_states)
        return normalized + self.fc2(self.activation(self.fc1(normalized)))


def _quality_conditioned_block_forward(
    block,
    hidden_states,
    encoder_hidden_states,
    timestep,
    quality_modulation,
    height,
    width,
):
    batch_size = hidden_states.shape[0]
    base_modulation = block.scale_shift_table[None, None] + timestep.reshape(
        batch_size, 1, 6, -1
    )
    modulation = base_modulation + quality_modulation
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        value.squeeze(2) for value in modulation.chunk(6, dim=2)
    )

    norm_hidden_states = block.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    attn_output = block.attn1(norm_hidden_states.to(hidden_states.dtype))
    hidden_states = hidden_states + gate_msa * attn_output

    if getattr(block, "attn2", None) is not None:
        attn_output = block.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
        )
        hidden_states = hidden_states + attn_output

    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(
        0, 3, 1, 2
    )
    ff_output = block.ff(norm_hidden_states)
    ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
    return hidden_states + gate_mlp * ff_output


def quality_conditioned_sana_forward(
    sana,
    hidden_states,
    encoder_hidden_states,
    timestep,
    quality_map,
    quality_conditioner,
    conditioning_scale=1.0,
    capture_layer=None,
    stop_layer=None,
    continuous_conditioning=False,
    checkpoint_blocks=False,
    precision_embedding=None,
):
    """Run SANA with spatial q conditioning and optionally expose one layer."""
    batch_size, _, height, width = hidden_states.shape
    patch_size = int(sana.config.patch_size)
    if patch_size != 1:
        raise ValueError(
            "dual-rate Self-Flow currently requires SANA patch_size=1 for exact "
            "latent-cell/token alignment"
        )
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    if quality_map.shape != (batch_size, post_patch_height, post_patch_width):
        raise ValueError(
            "quality_map must match the post-patch token grid; got "
            f"{tuple(quality_map.shape)}"
        )

    hidden_states = sana.patch_embed(hidden_states)
    timestep_modulation, embedded_timestep = sana.time_embed(
        timestep,
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    encoder_hidden_states = sana.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(
        batch_size, -1, hidden_states.shape[-1]
    )
    encoder_hidden_states = sana.caption_norm(encoder_hidden_states)

    quality_tokens = quality_map.reshape(batch_size, -1).to(
        device=hidden_states.device,
        dtype=torch.float32 if continuous_conditioning else torch.long,
    )
    if precision_embedding is not None:
        if not continuous_conditioning:
            raise ValueError("precision embedding requires continuous log-step conditioning")
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            embedded_precision = precision_embedding(quality_tokens.reshape(batch_size, 1, height, width))
        block_modulation, output_modulation = quality_conditioner(embedded_precision.flatten(2).transpose(1, 2))
    else:
        block_modulation, output_modulation = quality_conditioner(quality_tokens)
    scale = torch.as_tensor(
        conditioning_scale, device=hidden_states.device, dtype=hidden_states.dtype
    )
    block_modulation = block_modulation.to(hidden_states.dtype) * scale
    output_modulation = output_modulation.to(hidden_states.dtype) * scale

    captured = None
    for layer_index, block in enumerate(sana.transformer_blocks):
        modulation = quality_conditioner.block_residual(block_modulation, layer_index)
        if checkpoint_blocks and torch.is_grad_enabled():
            from functools import partial
            from torch.utils.checkpoint import checkpoint

            # Bind this iteration's block; a late-bound closure breaks recompute.
            hidden_states = checkpoint(
                partial(_quality_conditioned_block_forward, block),
                hidden_states, encoder_hidden_states, timestep_modulation,
                modulation, post_patch_height, post_patch_width,
                use_reentrant=False,
            )
        else:
            hidden_states = _quality_conditioned_block_forward(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                modulation,
                post_patch_height,
                post_patch_width,
            )
        if layer_index == capture_layer:
            captured = hidden_states
        if layer_index == stop_layer:
            return None, captured

    if capture_layer is not None and captured is None:
        raise ValueError(f"capture_layer {capture_layer} was not executed")

    output_base = sana.scale_shift_table[None, None] + embedded_timestep[
        :, None, None
    ].to(sana.scale_shift_table.device)
    output_shift, output_scale = (
        output_base + output_modulation
    ).chunk(2, dim=2)
    hidden_states = sana.norm_out.norm(hidden_states)
    hidden_states = hidden_states * (1 + output_scale.squeeze(2))
    hidden_states = hidden_states + output_shift.squeeze(2)
    hidden_states = sana.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_height,
        post_patch_width,
        patch_size,
        patch_size,
        -1,
    )
    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
    output = hidden_states.reshape(
        batch_size,
        -1,
        post_patch_height * patch_size,
        post_patch_width * patch_size,
    )
    return output, captured


def _is_teacher_ema_key(name):
    return (
        "lora_" in name
        or name.startswith("text_projection.")
        or name.startswith("quality_conditioner.")
    )


def _student_teacher_state(student_dit, text_projection, quality_conditioner, layers):
    sana = unwrap_sana_transformer(student_dit)
    state = {}
    component_names = (
        "patch_embed",
        "time_embed",
        "caption_projection",
        "caption_norm",
    )
    for component_name in component_names:
        component = getattr(sana, component_name)
        for name, value in component.state_dict().items():
            state[f"{component_name}.{name}"] = value
    for layer_index in range(layers):
        for name, value in sana.transformer_blocks[layer_index].state_dict().items():
            state[f"transformer_blocks.{layer_index}.{name}"] = value
    for name, value in text_projection.state_dict().items():
        state[f"text_projection.{name}"] = value
    for name, value in quality_conditioner.state_dict().items():
        state[f"quality_conditioner.{name}"] = value
    return {
        name: value for name, value in state.items() if _is_teacher_ema_key(name)
    }


class SelfFlowTeacher(nn.Module):
    """Truncated no-gradient EMA copy used only for the layer-20 target."""

    def __init__(
        self,
        student_dit,
        text_projection,
        quality_conditioner,
        target_layer,
    ):
        super().__init__()
        sana = unwrap_sana_transformer(student_dit)
        layer_count = int(target_layer) + 1
        if layer_count > len(sana.transformer_blocks):
            raise ValueError("teacher target layer exceeds the SANA depth")
        self.target_layer = int(target_layer)
        self.patch_embed = deepcopy(sana.patch_embed)
        self.time_embed = deepcopy(sana.time_embed)
        self.caption_projection = deepcopy(sana.caption_projection)
        self.caption_norm = deepcopy(sana.caption_norm)
        self.transformer_blocks = nn.ModuleList(
            deepcopy(list(sana.transformer_blocks[:layer_count]))
        )
        self.text_projection = deepcopy(text_projection)
        self.quality_conditioner = deepcopy(quality_conditioner)
        self.config = SimpleNamespace(
            patch_size=int(sana.config.patch_size),
            timestep_scale=float(sana.config.timestep_scale),
        )
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def update_from_student(
        self,
        student_dit,
        text_projection,
        quality_conditioner,
        decay,
    ):
        source = _student_teacher_state(
            student_dit,
            text_projection,
            quality_conditioner,
            self.target_layer + 1,
        )
        target = {
            name: value
            for name, value in self.state_dict().items()
            if _is_teacher_ema_key(name)
        }
        if source.keys() != target.keys():
            missing = sorted(target.keys() - source.keys())
            unexpected = sorted(source.keys() - target.keys())
            raise RuntimeError(
                f"teacher/student state mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        decay = float(decay)
        for name, target_tensor in target.items():
            source_tensor = source[name].detach().to(target_tensor.device)
            if torch.is_floating_point(target_tensor) and decay > 0:
                target_tensor.mul_(decay).add_(source_tensor, alpha=1 - decay)
            else:
                target_tensor.copy_(source_tensor)

    def get_ema_state_dict(self):
        return {
            name: value
            for name, value in self.state_dict().items()
            if _is_teacher_ema_key(name)
        }

    def load_ema_state_dict(self, state):
        expected = set(self.get_ema_state_dict())
        if set(state) != expected:
            raise RuntimeError(
                "Teacher EMA checkpoint keys do not match the active model"
            )
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"unexpected Teacher EMA keys: {result.unexpected_keys}"
            )

    def forward(
        self,
        latent,
        fixed_text_embed,
        timestep,
        quality_index,
        conditioning_scale,
    ):
        batch_size, _, height, width = latent.shape
        condition = self.text_projection(fixed_text_embed).expand(
            batch_size, -1, -1
        )
        quality_map = quality_index[:, None, None].expand(
            batch_size, height, width
        )
        _, feature = quality_conditioned_sana_forward(
            self,
            latent,
            condition.to(latent.device),
            timestep,
            quality_map,
            self.quality_conditioner,
            conditioning_scale=conditioning_scale,
            capture_layer=self.target_layer,
            stop_layer=self.target_layer,
        )
        return feature
