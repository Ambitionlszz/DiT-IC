"""DC-AE decoder checkpointing without changing module/state-dict names."""

import torch
from torch.utils.checkpoint import checkpoint


def decode_latent(vae, latent, checkpoint_blocks=False):
    if not checkpoint_blocks or not torch.is_grad_enabled():
        return vae.decode(latent).sample
    base = vae.get_base_model() if hasattr(vae, "get_base_model") else vae
    if base.use_slicing or base.use_tiling:
        raise ValueError("BT training uses full-frame decoder checkpointing, without slicing/tiling")
    decoder = base.decoder
    hidden = checkpoint(decoder.conv_in, latent, use_reentrant=False)
    if decoder.in_shortcut:
        hidden = hidden + latent.repeat_interleave(decoder.in_shortcut_repeats, dim=1)
    for block in reversed(decoder.up_blocks):
        hidden = checkpoint(block, hidden, use_reentrant=False)
    hidden = decoder.norm_out(hidden.movedim(1, -1)).movedim(-1, 1)
    hidden = decoder.conv_act(hidden)
    return checkpoint(decoder.conv_out, hidden, use_reentrant=False)


def decode_latent_with_pixel_adapter(vae, latent, adapter, sana_tokens, token_hw, delta,
                                    checkpoint_blocks=False, return_features=False):
    """One native AE pass, with an explicit update before norm/activation/RGB."""
    base = vae.get_base_model() if hasattr(vae, "get_base_model") else vae
    if base.use_slicing or base.use_tiling:
        raise ValueError("PiT requires the whole-image AE grid, without slicing or tiling")
    decoder = base.decoder

    def run(module, value):
        if checkpoint_blocks and torch.is_grad_enabled():
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    hidden = run(decoder.conv_in, latent)
    if decoder.in_shortcut:
        hidden = hidden + latent.repeat_interleave(decoder.in_shortcut_repeats, dim=1)
    for block in reversed(decoder.up_blocks):
        hidden = run(block, hidden)

    def rgb_tail(value, tokens, steps):
        # One checkpoint covers PiT and the high-resolution norm/activation/RGB
        # head. Inner PiT checkpoints would repeat that recomputation again.
        features = None
        options = {"checkpoint_blocks": False} if checkpoint_blocks else {}
        if return_features:
            value, features = adapter(value, tokens, token_hw, steps, return_features=True, **options)
        else:
            value = adapter(value, tokens, token_hw, steps, **options)
        value = decoder.norm_out(value.movedim(1, -1)).movedim(-1, 1)
        return decoder.conv_out(decoder.conv_act(value)), features

    if checkpoint_blocks and torch.is_grad_enabled():
        return checkpoint(rgb_tail, hidden, sana_tokens, delta, use_reentrant=False)
    return rgb_tail(hidden, sana_tokens, delta)
