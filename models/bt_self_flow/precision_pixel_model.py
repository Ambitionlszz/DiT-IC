"""Precision-consistent C with the two-block PiT pixel decoder.

The final tokens come from the same precision-conditioned SANA invocation that
predicts the native C update. Both branches consume the decoded local delta;
neither the source planner nor extra side information is used at the receiver.
"""

from torch.nn import functional as F

from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from .pixel_model import CPixelAdapterCodec


METHOD = "precision_consistent_pixel_adapter_v1"


class PrecisionCPixelAdapterCodec(CPixelAdapterCodec):
    decoder_method = METHOD

    def __init__(self, *args, **kwargs):
        codec = kwargs.get("codec_params") or {}
        if not codec.get("precision_conditioning") or not codec.get("precision_consistent"):
            raise ValueError("precision PiT requires the precision-consistent v3 C frontend")
        super().__init__(*args, **kwargs)

    def sana_features(self, mu, delta):
        if delta.shape != (mu.shape[0], 1, mu.shape[2] // 2, mu.shape[3] // 2):
            raise ValueError("precision PiT requires the decoded local quantization steps")
        sana = unwrap_sana_transformer(self.DiT)
        log_step = F.interpolate(delta.float().log(), size=mu.shape[-2:], mode="nearest").squeeze(1)
        prediction, tokens = quality_conditioned_sana_forward(
            sana, mu, self._fixed_text_condition(mu.shape[0], mu.device),
            self.self_flow_timestep(mu.shape[0], mu.device), log_step,
            self.quality_conditioner, continuous_conditioning=True,
            capture_layer=len(sana.transformer_blocks) - 1,
            checkpoint_blocks=self.checkpoint_blocks,
            precision_embedding=self.codec.precision_embedding,
        )
        return prediction, tokens, mu.shape[-2:]
