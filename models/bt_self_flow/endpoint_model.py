"""Native C receiver with optional D and training-only AE endpoint features."""

from contextlib import contextmanager

import torch

from models.sana_self_flow import unwrap_sana_transformer
from .decoder import (decode_latent, decode_latent_prefix,
                      decode_latent_with_endpoint_feature,
                      decode_latent_with_precision_mu_adapter)
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, RateControl, fixed_side_bits
from .pixel_adapter import AEConditionedHybridPixelAdapter, PrecisionMuPixelAdapter


LEGACY_D_METHOD = "precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4"
HYBRID_D_METHOD = "ae_conditioned_hybrid_pit_v1"


class EndpointAblationCodec(BTCodec):
    """All four groups share the same native SANA repair and AE path.

    SF has no inference module or additional state. `capture` exposes a live
    feature already computed by the actual AE decode; it does not edit it.
    """

    def __init__(self, *args, endpoint_config, pixel_config=None,
                 pixel_method=LEGACY_D_METHOD, **kwargs):
        kwargs["representation"] = False
        super().__init__(*args, **kwargs)
        if self.backend != "C" or not self.codec.precision_consistent:
            raise ValueError("endpoint SF requires precision-consistent native C")
        self.endpoint_settings = dict(endpoint_config)
        self.sf_enabled = bool(endpoint_config.get("enabled", False))
        self.d_enabled = pixel_config is not None
        self.hybrid_d = self.d_enabled and pixel_method == HYBRID_D_METHOD
        vae = self.vae.get_base_model() if hasattr(self.vae, "get_base_model") else self.vae
        if len(vae.decoder.up_blocks) != 6:
            raise ValueError("endpoint SF requires the six-stage f32 DC-AE")
        if self.d_enabled:
            if pixel_method not in (LEGACY_D_METHOD, HYBRID_D_METHOD):
                raise ValueError("unknown endpoint pixel decoder method")
            settings = dict(pixel_config)
            seed = settings.pop("seed", 903)
            projection = vae.decoder.conv_out
            projection = projection.get_base_layer() if hasattr(projection, "get_base_layer") else projection
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                if self.hybrid_d:
                    if settings.get("patch_size", 16) != 16:
                        raise ValueError("native AE H/8 conditioning requires 16-pixel PiT regions")
                    self.pixel_adapter = AEConditionedHybridPixelAdapter(
                        int(projection.in_channels), int(vae.config.decoder_block_out_channels[3]), **settings)
                else:
                    sana = unwrap_sana_transformer(self.DiT)
                    self.pixel_adapter = PrecisionMuPixelAdapter(
                        int(projection.in_channels), int(sana.config.in_channels), **settings)
        self.decoder_method = pixel_method if self.d_enabled else "native"

    @contextmanager
    def decoder_switches(self, *, d=None):
        """D can be bypassed; SF exists only as a training objective."""
        previous = self.d_enabled
        if d is True and not hasattr(self, "pixel_adapter"):
            raise ValueError("cannot enable D absent from this checkpoint")
        try:
            if d is not None:
                self.d_enabled = bool(d)
            yield self
        finally:
            self.d_enabled = previous

    def endpoint_feature(self, latent):
        """Current student AE metric; caller chooses target no_grad semantics."""
        return decode_latent_prefix(
            self.vae, latent / self.vae.config.scaling_factor, self.checkpoint_blocks)

    def decode_received(self, mu, delta, capture=False):
        prediction, _ = self.predict(mu, precision=delta)
        latent = self.sched.step(prediction, mu)
        ae_latent = latent / self.vae.config.scaling_factor
        feature = None
        if self.d_enabled:
            precision = self.codec.precision_embedding(delta.float().log())
            if self.hybrid_d:
                image, feature = decode_latent_with_endpoint_feature(
                    self.vae, ae_latent, self.pixel_adapter, (precision,), self.checkpoint_blocks,
                    condition_on_endpoint=True)
            elif capture:
                image, feature = decode_latent_with_endpoint_feature(
                    self.vae, ae_latent, self.pixel_adapter, (mu, precision), self.checkpoint_blocks)
            else:
                image, _ = decode_latent_with_precision_mu_adapter(
                    self.vae, ae_latent, self.pixel_adapter, mu, precision, self.checkpoint_blocks, False)
        elif capture:
            image, feature = decode_latent_with_endpoint_feature(
                self.vae, ae_latent, checkpoint_blocks=self.checkpoint_blocks)
        else:
            image = decode_latent(self.vae, ae_latent, self.checkpoint_blocks)
        return {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1),
                "feature": feature if capture else None}

    def forward(self, image, rate_control=None, transition=1., capture=False, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported endpoint forward arguments: {sorted(kwargs)}")
        control = rate_control or RateControl()
        clean = self.encode_clean_latent(image)
        source = self.codec.analyze(clean, control)
        return self.forward_analysis(source, control, transition, capture)

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        if control.mixed or transition != 1.:
            raise ValueError("endpoint SF uses whole-image native C views")
        view = self.codec.quantize_analysis(source, control)
        output = self.decode_received(view["mean"], view["delta"], capture=capture)
        h, w = view["allocation_codes"].shape[-2:]
        return BTForwardResult(source.clean, view["mean"], output["latent"], output["image"],
                               view["likelihoods"], fixed_side_bits(h, w, "C", False),
                               view["allocation_codes"], feature=output["feature"], source=source)

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError("decompress requires eval mode")
        packet = BTImagePacket.from_bytes(packet) if isinstance(packet, bytes) else packet
        if packet.backend != "C":
            raise ValueError("endpoint SF uses the unchanged C packet format")
        view = self.codec.decompress(packet)
        output = self.decode_received(view["mean"], view["delta"])
        self.last_decode_stats = {"repair_latent_rms":
            (output["latent"].float() - view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return output["image"][..., :h, :w]

    def export_state(self):
        state = super().export_state()
        if hasattr(self, "pixel_adapter"):
            state["pixel_adapter"] = self.pixel_adapter.state_dict()
        return state

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False,
                      initialize_pixel_adapter=False):
        if not codec_only:
            if any(name in state for name in ("spatial_projector", "projector")):
                raise ValueError("endpoint SF cannot load a checkpoint with a feature projector")
            module = getattr(self, "pixel_adapter", None)
            if module is not None and "pixel_adapter" not in state and not initialize_pixel_adapter:
                raise ValueError("checkpoint is missing pixel_adapter; initialize explicitly")
            if module is None and "pixel_adapter" in state:
                raise ValueError("checkpoint contains D absent from this ablation")
            if module is not None and "pixel_adapter" in state:
                if self.hybrid_d:
                    module.validate_checkpoint(state["pixel_adapter"])
                else:
                    for key in ("qk_source_code", "precision_adaln_version"):
                        expected = module.state_dict()[key]
                        if key not in state["pixel_adapter"] or not torch.equal(
                                state["pixel_adapter"][key].cpu(), expected.cpu()):
                            raise ValueError("incompatible mu-QK precision D checkpoint")
        super().load_bt_state(state, codec_only, initialize_conditioner)
        if not codec_only and hasattr(self, "pixel_adapter") and "pixel_adapter" in state:
            self.pixel_adapter.load_state_dict(state["pixel_adapter"], strict=True)
