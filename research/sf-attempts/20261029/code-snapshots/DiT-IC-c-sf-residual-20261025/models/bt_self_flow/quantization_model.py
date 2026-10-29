"""Deployable C decoder with a shared reconstruction/Self-Flow residual."""

import torch

from models.sana_self_flow import unwrap_sana_transformer
from .decoder import decode_latent
from .model import BTCodec, BTForwardResult
from .packet import RateControl, fixed_side_bits
from .quantization_residual import (QuantizationResidualHead, c_sana_forward,
                                    gaussian_residual_loss, normalized_residual)


class CQuantizationResidualCodec(BTCodec):
    def __init__(self, *args, **kwargs):
        if kwargs.get("backend", "C") != "C":
            raise ValueError("quantization residual SF requires backend C")
        kwargs.update(backend="C", representation=False)
        super().__init__(*args, **kwargs)
        if not self.codec.precision_conditioning:
            raise ValueError("quantization residual SF requires precision v2")
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads*sana.config.attention_head_dim)
        # Construct dormant parameters before DDP wrapping; their optimizer
        # states remain absent until the 10K activation. Do not alter base RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3903)
            self.residual_head = QuantizationResidualHead(hidden)
        self.residual_head.register_buffer("active", torch.tensor(False))
        self.residual_intervention = "normal"

    def activate_residual(self):
        self.residual_head.active.fill_(True)

    def repair_received(self, coarse, delta, residual_override=None):
        if not self.residual_head.active.item() and residual_override is None:
            return super().repair(coarse)
        time = self.self_flow_timestep(coarse.shape[0], coarse.device)
        self.prediction_calls += 1
        head = None if self.residual_intervention == "zero" or residual_override is not None else self.residual_head
        prediction, _, _, _ = c_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device), time,
            checkpoint_blocks=self.checkpoint_blocks, head=head, log_step=delta.log(),
            student_layer=self.student_layer, injection_layer=self.teacher_layer,
            residual_override=residual_override, shuffle_mean=self.residual_intervention == "shuffle")
        latent = self.sched.step(prediction, coarse)
        image = decode_latent(self.vae, latent/self.vae.config.scaling_factor, self.checkpoint_blocks).clamp(-1, 1)
        return latent, image, None

    def repair(self, coarse, scale_info=None, transition=1., capture=False):
        if self.residual_head.active.item():
            if scale_info is None or "delta" not in scale_info:
                raise ValueError("active residual decoder requires the actual decoded quantization step")
            return self.repair_received(coarse, scale_info["delta"])
        return super().repair(coarse, scale_info, transition, capture)

    def forward(self, image, rate_control=None, transition=1., capture=False,
                sf_teacher=None, sf_mask=None, **kwargs):
        control = rate_control or RateControl()
        source = self.codec.analyze(self.encode_clean_latent(image), control)
        return self.forward_analysis(source, control, transition, capture,
                                     sf_teacher=sf_teacher, sf_mask=sf_mask)

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False,
                         sf_teacher=None, sf_mask=None):
        control = rate_control or RateControl()
        view = self.codec.quantize_analysis(source, control, erase_mask=sf_mask)
        latent, image, _ = self.repair_received(view["mean"], view["delta"])
        h, w = view["allocation_codes"].shape[-2:]
        result = BTForwardResult(source.clean, view["mean"], latent, image, view["likelihoods"],
                                 fixed_side_bits(h, w, "C", False), view["allocation_codes"], source=source)
        if sf_teacher is not None:
            if not self.residual_head.active.item():
                raise ValueError("teacher requires an active residual head")
            with torch.no_grad():
                native_delta = self.codec.steps(control, source.allocation)
                fine_delta = torch.minimum(native_delta, view["delta"].detach())
                valid = (fine_delta < view["delta"].detach()-1e-7).flatten(1).any(1)
                selection = valid.nonzero().flatten()
                if selection.numel() == 0:
                    # A zero-weight dummy retains the same DDP graph shape.
                    selection = torch.zeros(1, dtype=torch.long, device=valid.device)
                fine = self.codec.quantize_analysis(source, control, delta_override=fine_delta,
                                                     compute_likelihood=False)
                time = self.self_flow_timestep(selection.numel(), view["mean"].device)
                low_feature = sf_teacher(view["mean"][selection].detach(), self.fixed_text_embed, time)
                high_feature = sf_teacher(fine["mean"][selection], self.fixed_text_embed, time)
                target = normalized_residual(high_feature, low_feature)
            # Complete SF gradient isolation from analysis/router/g_s, including
            # the log-step condition. Shared prefix/head still learn normally.
            _, _, mean, variance = c_sana_forward(
                unwrap_sana_transformer(self.DiT), view["mean"][selection].detach(),
                self._fixed_text_condition(selection.numel(), view["mean"].device), time,
                stop_layer=self.student_layer, checkpoint_blocks=self.checkpoint_blocks,
                head=self.residual_head, log_step=view["delta"][selection].detach().log(),
                student_layer=self.student_layer, injection_layer=self.teacher_layer)
            result.auxiliary = dict(sf_loss=gaussian_residual_loss(mean, variance, target, valid[selection])
                                            *selection.numel()/valid.numel(),
                                    sf_valid=valid.float().mean().detach(),
                                    sf_target_rms=target.square().mean().sqrt().detach(),
                                    sf_mean_rms=mean.float().square().mean().sqrt().detach(),
                                    sf_std=variance.sqrt().mean().detach())
        return result

    @torch.no_grad()
    def decompress(self, packet):
        from .packet import BTImagePacket
        if self.training:
            raise ValueError("decompress requires eval mode")
        packet = BTImagePacket.from_bytes(packet) if isinstance(packet, bytes) else packet
        if packet.backend != "C":
            raise ValueError("packet backend does not match the C decoder")
        view = self.codec.decompress(packet)
        latent, image, _ = self.repair_received(view["mean"], view["delta"])
        self.last_decode_stats = {"repair_latent_rms": (latent.float()-view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return image[..., :h, :w]

    def export_state(self):
        return dict(super().export_state(), residual=self.residual_head.state_dict())

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False):
        super().load_bt_state(state, codec_only, initialize_conditioner)
        if not codec_only:
            if "residual" in state:
                self.residual_head.load_state_dict(state["residual"], strict=True)
            elif not initialize_conditioner:
                raise ValueError("residual decoder checkpoint is missing its head/activation state")
