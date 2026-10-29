"""Native C packets with a two-block PiT adapter inside the AE decoder."""

import torch
from torch.utils.checkpoint import checkpoint

from models.sana_self_flow import unwrap_sana_transformer
from .decoder import decode_latent_with_pixel_adapter
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, RateControl, fixed_side_bits
from .pixel_adapter import PixelDecoderAdapter


METHOD = "sana_conditioned_pixel_adapter_v1"


def native_sana_features(sana, coarse, text, timestep, checkpoint_blocks=False):
    """Execute native children once; expose the final tokens before norm_out."""
    batch, _, height, width = coarse.shape
    if int(sana.config.patch_size) != 1:
        raise ValueError("pixel adaptation requires one native SANA token per latent cell")
    tokens = sana.patch_embed(coarse)
    modulation, embedded = sana.time_embed(timestep, batch_size=batch, hidden_dtype=tokens.dtype)
    text = sana.caption_projection(text).view(batch, -1, tokens.shape[-1])
    text = sana.caption_norm(text)
    for block in sana.transformer_blocks:
        args = (tokens, None, text, None, modulation, height, width)
        if checkpoint_blocks and torch.is_grad_enabled():
            tokens = checkpoint(block, *args, use_reentrant=False)
        else:
            tokens = block(*args)
    prediction = sana.proj_out(sana.norm_out(tokens, embedded, sana.scale_shift_table))
    prediction = prediction.reshape(batch, height, width, 1, 1, -1)
    prediction = prediction.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, height, width)
    return prediction, tokens, (height, width)


class CPixelAdapterCodec(BTCodec):
    decoder_method = METHOD

    def __init__(self, *args, pixel_config=None, **kwargs):
        if kwargs.get("backend", "C") != "C" or kwargs.get("representation", False):
            raise ValueError("PiT uses native C and disables the legacy Self-Flow path")
        kwargs["backend"] = "C"
        super().__init__(*args, **kwargs)
        if not self.codec.precision_conditioning:
            raise ValueError("PiT requires the unchanged precision-v2 C frontend")
        settings = dict(pixel_config or {})
        self.pixel_seed = int(settings.pop("seed", 903))
        sana = unwrap_sana_transformer(self.DiT)
        vae = self.vae.get_base_model() if hasattr(self.vae, "get_base_model") else self.vae
        rgb_projection = vae.decoder.conv_out
        if hasattr(rgb_projection, "get_base_layer"):
            rgb_projection = rgb_projection.get_base_layer()
        feature_channels = int(rgb_projection.in_channels)
        hidden = int(sana.config.num_attention_heads * sana.config.attention_head_dim)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.pixel_seed)
            self.pixel_adapter = PixelDecoderAdapter(feature_channels, hidden, **settings)
        self.self_flow_enabled = False

    def decode_received(self, mu, delta, return_features=False):
        self.prediction_calls += 1
        prediction, tokens, token_hw = native_sana_features(
            unwrap_sana_transformer(self.DiT), mu, self._fixed_text_condition(mu.shape[0], mu.device),
            self.self_flow_timestep(mu.shape[0], mu.device), self.checkpoint_blocks)
        latent = self.sched.step(prediction, mu)
        image, features = decode_latent_with_pixel_adapter(
            self.vae, latent / self.vae.config.scaling_factor, self.pixel_adapter,
            tokens, token_hw, delta, self.checkpoint_blocks, return_features)
        output = {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1)}
        if return_features:
            output.update(sana_tokens=tokens, sana_token_hw=token_hw, **features)
        return output

    def repair(self, *args, **kwargs):
        raise ValueError("PiT decoding needs the actual delta; call decode_received(mu, delta)")

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        if transition != 1. or control.mixed:
            raise ValueError("PiT uses native one-step C and whole-image quality")
        view = self.codec.quantize_analysis(source, control)
        output = self.decode_received(view["mean"], view["delta"], return_features=capture)
        h, w = view["allocation_codes"].shape[-2:]
        result = BTForwardResult(source.clean, view["mean"], output["latent"], output["image"],
                                 view["likelihoods"], fixed_side_bits(h, w, "C", False),
                                 view["allocation_codes"], source=source)
        if capture:
            result.pixel_features = {name: output[name] for name in
                                     ("sana_tokens", "sana_token_hw", "pixel_tokens", "token_hw")}
        return result

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError("decompress requires eval mode")
        packet = BTImagePacket.from_bytes(packet) if isinstance(packet, bytes) else packet
        if packet.backend != "C":
            raise ValueError("PiT requires an unchanged native C packet")
        view = self.codec.decompress(packet)
        output = self.decode_received(view["mean"], view["delta"])
        self.last_decode_stats = {"repair_latent_rms":
            (output["latent"].float() - view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return output["image"][..., :h, :w]

    def export_state(self):
        return dict(super().export_state(), pixel_adapter=self.pixel_adapter.state_dict())

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False, initialize_pixel_adapter=False):
        if not codec_only and "pixel_adapter" not in state and not initialize_pixel_adapter:
            raise ValueError("PiT checkpoint is missing pixel_adapter; shared C initialization must be explicit")
        super().load_bt_state(state, codec_only, initialize_conditioner)
        if not codec_only and "pixel_adapter" in state:
            self.pixel_adapter.load_state_dict(state["pixel_adapter"], strict=True)
