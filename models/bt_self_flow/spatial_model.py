"""One receiver path for C, C+D, C+SF, C+D+SF and P-only controls."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from .decoder import decode_latent, decode_latent_with_precision_mu_adapter
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, RateControl, fixed_side_bits
from .pixel_adapter import PrecisionMuPixelAdapter
from .spatial_self_flow import SpatialProjector


class SpatialAblationCodec(BTCodec):
    """SF's raw convolution output is the actual input to the following block."""

    def __init__(self, *args, spatial_config, pixel_config=None, **kwargs):
        kwargs["representation"] = False
        super().__init__(*args, **kwargs)
        if self.backend != "C" or not self.codec.precision_consistent:
            raise ValueError("spatial SF requires the precision-consistent C receiver")
        self.spatial_settings = dict(spatial_config)
        self.sf_enabled = bool(spatial_config.get("enabled", False))
        self.p_enabled = bool(spatial_config.get("projector_enabled", self.sf_enabled))
        self.d_enabled = pixel_config is not None
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads * sana.config.attention_head_dim)
        self.spatial_projector = SpatialProjector(hidden) if self.p_enabled else None
        self.student_layer = int(spatial_config.get("student_layer", 8)) - 1
        self.teacher_layer = int(spatial_config.get("teacher_layer", 20)) - 1
        if not 0 <= self.student_layer < self.teacher_layer < len(sana.transformer_blocks):
            raise ValueError("spatial student/teacher depths are outside SANA")
        if self.d_enabled:
            settings = dict(pixel_config)
            seed = settings.pop("seed", 903)
            vae = self.vae.get_base_model() if hasattr(self.vae, "get_base_model") else self.vae
            projection = vae.decoder.conv_out
            projection = projection.get_base_layer() if hasattr(projection, "get_base_layer") else projection
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.pixel_adapter = PrecisionMuPixelAdapter(
                    int(projection.in_channels), int(sana.config.in_channels), **settings)
        self.decoder_method = ("precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4"
                               if self.d_enabled else "native")

    @contextmanager
    def decoder_switches(self, *, sf=None, d=None):
        """Inference diagnostic; does not undo either module's training history."""
        previous = self.p_enabled, self.d_enabled
        if sf is True and self.spatial_projector is None:
            raise ValueError("cannot enable a projector absent from this checkpoint")
        if d is True and not hasattr(self, "pixel_adapter"):
            raise ValueError("cannot enable D absent from this checkpoint")
        try:
            if sf is not None:
                self.p_enabled = bool(sf)
            if d is not None:
                self.d_enabled = bool(d)
            yield self
        finally:
            self.p_enabled, self.d_enabled = previous

    def predict(self, coarse, log_rho=None, transition=1., capture=False, precision=None,
                projector_mask=None):
        if transition != 1. or precision is None or precision.shape != (
                coarse.shape[0], 1, coarse.shape[2] // 2, coarse.shape[3] // 2):
            raise ValueError("spatial receiver requires native C and decoded local precision")
        self.prediction_calls += 1
        log_step = F.interpolate(precision.float().log(), size=coarse.shape[-2:], mode="nearest").squeeze(1)
        return quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device),
            self.self_flow_timestep(coarse.shape[0], coarse.device), log_step,
            self.quality_conditioner, continuous_conditioning=True,
            capture_layer=self.student_layer if capture else None,
            checkpoint_blocks=self.checkpoint_blocks,
            precision_embedding=self.codec.precision_embedding,
            spatial_projector=self.spatial_projector if self.p_enabled else None,
            projector_layer=self.student_layer, projector_mask=projector_mask)

    def decode_received(self, mu, delta, capture=False, projector_mask=None):
        prediction, feature = self.predict(mu, precision=delta, capture=capture,
                                           projector_mask=projector_mask)
        latent = self.sched.step(prediction, mu)
        if self.d_enabled:
            precision = self.codec.precision_embedding(delta.float().log())
            image, _ = decode_latent_with_precision_mu_adapter(
                self.vae, latent / self.vae.config.scaling_factor, self.pixel_adapter,
                mu, precision, self.checkpoint_blocks, False)
        else:
            image = decode_latent(self.vae, latent / self.vae.config.scaling_factor, self.checkpoint_blocks)
        return {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1), "feature": feature}

    def forward(self, image, rate_control=None, transition=1., capture=False, projector_mask=None, **kwargs):
        control = rate_control or RateControl()
        clean = self.encode_clean_latent(image)
        source = self.codec.analyze(clean, control)
        return self.forward_analysis(source, control, transition, capture, projector_mask)

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False, projector_mask=None):
        control = rate_control or RateControl()
        if control.mixed or transition != 1.:
            raise ValueError("spatial ablation uses whole-image, native C views")
        view = self.codec.quantize_analysis(source, control)
        output = self.decode_received(view["mean"], view["delta"], capture, projector_mask)
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
            raise ValueError("spatial SF uses the unchanged C packet format")
        view = self.codec.decompress(packet)
        output = self.decode_received(view["mean"], view["delta"])
        self.last_decode_stats = {"repair_latent_rms":
            (output["latent"].float() - view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return output["image"][..., :h, :w]

    def export_state(self):
        state = super().export_state()
        if self.spatial_projector is not None:
            state["spatial_projector"] = self.spatial_projector.state_dict()
        if hasattr(self, "pixel_adapter"):
            state["pixel_adapter"] = self.pixel_adapter.state_dict()
        return state

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False,
                      initialize_pixel_adapter=False, initialize_spatial_projector=False):
        if not codec_only:
            for name, initialize in (("pixel_adapter", initialize_pixel_adapter),
                                     ("spatial_projector", initialize_spatial_projector)):
                module = getattr(self, name, None)
                if module is not None and name not in state and not initialize:
                    raise ValueError(f"checkpoint is missing {name}; initialize explicitly")
                if module is None and name in state:
                    raise ValueError(f"checkpoint contains {name} absent from this ablation")
            if "pixel_adapter" in state and hasattr(self, "pixel_adapter"):
                for key in ("qk_source_code", "precision_adaln_version"):
                    expected = self.pixel_adapter.state_dict()[key]
                    if key not in state["pixel_adapter"] or not torch.equal(
                            state["pixel_adapter"][key].cpu(), expected.cpu()):
                        raise ValueError("incompatible mu-QK precision D checkpoint")
        super().load_bt_state(state, codec_only, initialize_conditioner)
        if not codec_only:
            for name in ("pixel_adapter", "spatial_projector"):
                module = getattr(self, name, None)
                if module is not None and name in state:
                    module.load_state_dict(state[name], strict=True)
