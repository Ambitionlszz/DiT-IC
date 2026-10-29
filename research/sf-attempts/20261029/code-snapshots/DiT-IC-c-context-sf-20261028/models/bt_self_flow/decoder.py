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
