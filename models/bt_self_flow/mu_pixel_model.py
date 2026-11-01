"""New C plus mu-only LN PiT: pixel-Q/K control A and mu-Q/K main model B."""

from models.sana_self_flow import unwrap_sana_transformer
from .decoder import decode_latent_with_mu_adapter, decode_latent_with_precision_mu_adapter
from .precision_pixel_model import PrecisionCPixelAdapterCodec
from .pixel_adapter import MuPixelAdapter, PrecisionMuPixelAdapter


CONTROL_METHOD = "precision_consistent_mu_adaln_pixel_adapter_v3"
METHOD = "precision_consistent_mu_qk_pixel_adapter_v3"
PRECISION_METHOD = "precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4"


class MuQKCPixelCodec(PrecisionCPixelAdapterCodec):
    decoder_method = METHOD
    qk_source = "mu"

    def make_pixel_adapter(self, feature_channels, sana_dim, settings):
        latent_channels = int(unwrap_sana_transformer(self.DiT).config.in_channels)
        return MuPixelAdapter(feature_channels, latent_channels, qk_source=self.qk_source, **settings)

    def decode_received(self, mu, delta, return_features=False):
        # Unmodified new C prediction, including its existing precision input.
        # The pixel adapter itself sees only AE features and received mu.
        prediction, _ = self.predict(mu, precision=delta)
        latent = self.sched.step(prediction, mu)
        image, features = decode_latent_with_mu_adapter(
            self.vae, latent / self.vae.config.scaling_factor, self.pixel_adapter, mu,
            self.checkpoint_blocks, return_features)
        output = {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1)}
        if return_features:
            output.update(mu=mu, **features)
        return output

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False, initialize_pixel_adapter=False):
        if initialize_pixel_adapter and not codec_only:
            # Shared backend_init can contain old D weights. Explicit handoff
            # keeps this architecture's freshly initialized D instead.
            state = {name: values for name, values in state.items() if name != "pixel_adapter"}
        if not codec_only and "pixel_adapter" in state:
            code = state["pixel_adapter"].get("qk_source_code")
            if code is None or code.shape != self.pixel_adapter.qk_source_code.shape or (
                    code.item() != self.pixel_adapter.qk_source_code.item()):
                raise ValueError("incompatible mu PiT checkpoint or Q/K source; initialize D explicitly")
        super().load_bt_state(state, codec_only, initialize_conditioner, initialize_pixel_adapter)


class MuPixelQKCPixelCodec(MuQKCPixelCodec):
    decoder_method = CONTROL_METHOD
    qk_source = "pixels"


class PrecisionMuQKCPixelCodec(MuQKCPixelCodec):
    decoder_method = PRECISION_METHOD

    def make_pixel_adapter(self, feature_channels, sana_dim, settings):
        channels = int(unwrap_sana_transformer(self.DiT).config.in_channels)
        return PrecisionMuPixelAdapter(feature_channels, channels, **settings)

    def decode_received(self, mu, delta, return_features=False):
        prediction, _ = self.predict(mu, precision=delta)
        latent = self.sched.step(prediction, mu)
        precision = self.codec.precision_embedding(delta.float().log())
        image, features = decode_latent_with_precision_mu_adapter(
            self.vae, latent / self.vae.config.scaling_factor, self.pixel_adapter, mu, precision,
            self.checkpoint_blocks, return_features)
        output = {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1)}
        if return_features:
            output.update(mu=mu, delta=delta, precision_features=precision, **features)
        return output

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False, initialize_pixel_adapter=False):
        if not codec_only and not initialize_pixel_adapter and "pixel_adapter" in state:
            version = state["pixel_adapter"].get("precision_adaln_version")
            if version is None or version.shape != self.pixel_adapter.precision_adaln_version.shape or version.item() != 1:
                raise ValueError("checkpoint is missing or has incompatible precision AdaLN; initialize D explicitly")
        super().load_bt_state(state, codec_only, initialize_conditioner, initialize_pixel_adapter)
