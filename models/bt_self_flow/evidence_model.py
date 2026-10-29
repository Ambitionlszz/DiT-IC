"""One shared C: privileged cross-rate evidence during training, low packet at test.

Both roles repair the SAME low-rate state. Only the cross-attention evidence
changes. There is no teacher copy, prediction head, or additional packet field.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from .decoder import decode_latent
from .model import BTCodec, BTForwardResult
from .packet import RateControl, fixed_side_bits


METHOD = "cross_rate_evidence_self_flow_v1"


class EvidenceSelfFlowCodec(BTCodec):
    def __init__(self, *args, evidence_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.backend != "C" or not self.codec.precision_consistent:
            raise ValueError("evidence Self-Flow requires precision-consistent C")
        self.initialize_evidence(evidence_config or {})

    def initialize_evidence(self, settings):
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads * sana.config.attention_head_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.get("seed", 2903))
            self.evidence_projection = nn.Linear(128, hidden)
            nn.init.zeros_(self.evidence_projection.weight)
            nn.init.zeros_(self.evidence_projection.bias)
        self.text_proj.requires_grad_(False)
        self.evidence_config = dict(settings)

    def evidence_tokens(self, mean, precision):
        sana = unwrap_sana_transformer(self.DiT)
        log_step = F.interpolate(precision.float().log(), size=mean.shape[-2:], mode="nearest")
        with torch.autocast(device_type=mean.device.type, enabled=False):
            embedded = self.codec.precision_embedding(log_step)
        return sana.patch_embed(mean) + self.evidence_projection(embedded.flatten(2).transpose(1, 2))

    def predict(self, coarse, log_rho=None, transition=1., capture=False, precision=None,
                evidence=None, capture_layer=None):
        if precision is None or precision.shape != (coarse.shape[0], 1, coarse.shape[2] // 2, coarse.shape[3] // 2):
            raise ValueError("evidence generation requires the decoded low-state precision")
        self.prediction_calls += 1
        mean_e, delta_e = (coarse, precision) if evidence is None else evidence
        context = self.evidence_tokens(mean_e, delta_e)
        log_step = F.interpolate(precision.float().log(), size=coarse.shape[-2:], mode="nearest").squeeze(1)
        return quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse, context,
            self.self_flow_timestep(coarse.shape[0], coarse.device), log_step,
            self.quality_conditioner, continuous_conditioning=True,
            checkpoint_blocks=self.checkpoint_blocks, precision_embedding=self.codec.precision_embedding,
            context_is_projected=True,
            capture_layer=capture_layer if capture_layer is not None else (self.student_layer if capture else None),
        )

    def repair(self, coarse, scale_info=None, transition=1., capture=False, precision=None,
               evidence=None, capture_layer=None):
        prediction, feature = self.predict(coarse, transition=transition, capture=capture,
                                           precision=precision, evidence=evidence, capture_layer=capture_layer)
        latent = self.sched.step(prediction, coarse)
        image = decode_latent(self.vae, latent / self.vae.config.scaling_factor,
                              self.checkpoint_blocks).clamp(-1, 1)
        return latent, image, feature

    def result_from_view(self, source, view, control, capture=False):
        latent, image, feature = self.repair(view["mean"], precision=view["delta"], capture=capture)
        h, w = view["allocation_codes"].shape[-2:]
        result = BTForwardResult(source.clean, view["mean"], latent, image, view["likelihoods"],
                                 fixed_side_bits(h, w, "C", False), view["allocation_codes"],
                                 feature=feature, source=source)
        return result

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        return self.result_from_view(source, self.codec.quantize_analysis(source, control), control, capture)

    def export_state(self):
        return dict(super().export_state(), evidence_projection=self.evidence_projection.state_dict())

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False, initialize_evidence=False):
        super().load_bt_state(state, codec_only=codec_only, initialize_conditioner=initialize_conditioner)
        if codec_only:
            return
        if "evidence_projection" in state:
            self.evidence_projection.load_state_dict(state["evidence_projection"], strict=True)
        elif not initialize_evidence:
            raise ValueError("evidence checkpoint is missing its deployment context projection")
