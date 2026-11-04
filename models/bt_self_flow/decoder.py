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
    return _decode_with_adapter(vae, latent, adapter, (sana_tokens, token_hw, delta),
                                checkpoint_blocks, return_features)


def decode_latent_with_received_update(vae, latent, adapter, mu, update, precision_features, s_data,
                                       checkpoint_blocks=False, return_features=False):
    return _decode_with_adapter(vae, latent, adapter, (mu, update, precision_features, s_data),
                                checkpoint_blocks, return_features)


def decode_latent_with_mu_adapter(vae, latent, adapter, mu, checkpoint_blocks=False, return_features=False):
    return _decode_with_adapter(vae, latent, adapter, (mu,), checkpoint_blocks, return_features)


def decode_latent_with_precision_mu_adapter(vae, latent, adapter, mu, precision_features,
                                          checkpoint_blocks=False, return_features=False):
    return _decode_with_adapter(vae, latent, adapter, (mu, precision_features), checkpoint_blocks, return_features)


def _decode_with_adapter(vae, latent, adapter, conditions, checkpoint_blocks, return_features):
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

    def rgb_tail(value, *received):
        # One checkpoint covers PiT and the high-resolution norm/activation/RGB
        # head. Inner PiT checkpoints would repeat that recomputation again.
        features = None
        options = {"checkpoint_blocks": False} if checkpoint_blocks else {}
        if return_features:
            value, features = adapter(value, *received, return_features=True, **options)
        else:
            value = adapter(value, *received, **options)
        value = decoder.norm_out(value.movedim(1, -1)).movedim(-1, 1)
        return decoder.conv_out(decoder.conv_act(value)), features

    if checkpoint_blocks and torch.is_grad_enabled():
        return checkpoint(rgb_tail, hidden, *conditions, use_reentrant=False)
    return rgb_tail(hidden, *conditions)


def _endpoint_decoder(vae):
    """The f32 DC-AE H/8 metric ends after native decoder up_blocks[3]."""
    base = vae.get_base_model() if hasattr(vae, "get_base_model") else vae
    if base.use_slicing or base.use_tiling:
        raise ValueError("endpoint SF requires whole-image DC-AE decoding")
    decoder = base.decoder
    if len(decoder.up_blocks) != 6:
        raise ValueError("endpoint SF H/8 feature requires the six-stage f32 DC-AE")
    return decoder


def _endpoint_run(module, value, checkpoint_blocks):
    if checkpoint_blocks and torch.is_grad_enabled():
        return checkpoint(module, value, use_reentrant=False)
    return module(value)


def _check_endpoint_shape(feature, latent):
    if feature.shape[0] != latent.shape[0] or feature.shape[-2:] != (
            latent.shape[-2] * 4, latent.shape[-1] * 4):
        raise ValueError("native up_blocks[3] did not produce the required H/8 grid")


def decode_latent_prefix(vae, latent, checkpoint_blocks=False):
    """Read the current AE H/8 feature, taking an unscaled AE-domain latent.

    This helper has no teacher, state, hooks or automatic detach. A caller
    constructing a target must wrap it in no_grad; a student retains its graph.
    """
    decoder = _endpoint_decoder(vae)
    hidden = _endpoint_run(decoder.conv_in, latent, checkpoint_blocks)
    if decoder.in_shortcut:
        hidden = hidden + latent.repeat_interleave(decoder.in_shortcut_repeats, dim=1)
    for index in range(5, 2, -1):
        hidden = _endpoint_run(decoder.up_blocks[index], hidden, checkpoint_blocks)
    _check_endpoint_shape(hidden, latent)
    return hidden


def decode_latent_with_endpoint_feature(vae, latent, adapter=None, conditions=(),
                                      checkpoint_blocks=False, condition_on_endpoint=False):
    """One actual AE decode, explicitly returning its live H/8 intermediate.

    Capturing a tensor outside checkpoint closures avoids mutable forward-hook
    state when autograd recomputes blocks. The optional adapter uses exactly
    the existing pre-RGB insertion and received conditions.
    """
    decoder = _endpoint_decoder(vae)
    feature = decode_latent_prefix(vae, latent, checkpoint_blocks)
    hidden = feature
    for index in range(2, -1, -1):
        hidden = _endpoint_run(decoder.up_blocks[index], hidden, checkpoint_blocks)
    if adapter is None:
        hidden = decoder.norm_out(hidden.movedim(1, -1)).movedim(-1, 1)
        hidden = decoder.conv_act(hidden)
        return _endpoint_run(decoder.conv_out, hidden, checkpoint_blocks), feature

    if condition_on_endpoint:
        # This is the actual live student intermediate. Make it an explicit
        # checkpoint input rather than a mutable hook/cache or detached target.
        conditions = (feature, *conditions)

    def rgb_tail(value, *received):
        options = {"checkpoint_blocks": False} if checkpoint_blocks else {}
        value = adapter(value, *received, **options)
        value = decoder.norm_out(value.movedim(1, -1)).movedim(-1, 1)
        return decoder.conv_out(decoder.conv_act(value))

    if checkpoint_blocks and torch.is_grad_enabled():
        image = checkpoint(rgb_tail, hidden, *conditions, use_reentrant=False)
    else:
        image = rgb_tail(hidden, *conditions)
    return image, feature
