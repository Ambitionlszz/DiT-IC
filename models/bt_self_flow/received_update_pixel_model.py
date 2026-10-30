"""New C + receiver/update-conditioned PiT, without extra bits or SANA calls."""

from models.sana_self_flow import unwrap_sana_transformer
from .decoder import decode_latent_with_received_update
from .pixel_adapter import ReceivedUpdatePixelAdapter
from .precision_pixel_model import PrecisionCPixelAdapterCodec


METHOD = "precision_consistent_received_update_pixel_adapter_v2"


class ReceivedUpdateCPixelCodec(PrecisionCPixelAdapterCodec):
    decoder_method = METHOD

    def make_pixel_adapter(self, feature_channels, sana_dim, settings):
        latent_channels = int(unwrap_sana_transformer(self.DiT).config.in_channels)
        return ReceivedUpdatePixelAdapter(feature_channels, latent_channels, **settings)

    def decode_received(self, mu, delta, return_features=False):
        # Use precisely C's native precision-conditioned prediction and scheduler.
        # No last-layer hidden capture and no second generator invocation.
        prediction, _ = self.predict(mu, precision=delta)
        latent = self.sched.step(prediction, mu)
        update = latent.float() - mu.float()
        precision_features = self.codec.precision_embedding(delta.float().log())
        image, features = decode_latent_with_received_update(
            self.vae, latent / self.vae.config.scaling_factor, self.pixel_adapter,
            mu, update, precision_features, self.distortion_scale.s_data,
            self.checkpoint_blocks, return_features)
        output = {"prediction": prediction, "latent": latent, "image": image.clamp(-1, 1)}
        if return_features:
            output.update(mu=mu, update=update, delta=delta, **features)
        return output

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False, initialize_pixel_adapter=False):
        if initialize_pixel_adapter and not codec_only:
            # A shared backend_init may contain a v1 adapter. Explicit stage
            # initialization starts v2 at its own isolated seed/zero writeback;
            # ordinary deployment/resume still strictly loads every v2 tensor.
            state = {name: values for name, values in state.items() if name != "pixel_adapter"}
        super().load_bt_state(state, codec_only, initialize_conditioner, initialize_pixel_adapter)
