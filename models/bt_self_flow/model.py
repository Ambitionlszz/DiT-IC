"""Pure SANA BT path: fixed native time, one endpoint update, paid scales."""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderDC

from models.DiT_IC import Codec
from models.sana_self_flow import (RepresentationProjector, SelfFlowTeacher,
                                   quality_conditioned_sana_forward, unwrap_sana_transformer)
from .codec import BTLatentCodec
from .conditioning import ContinuousDistortionConditioner, residual_update
from .decoder import decode_latent
from .packet import BTImagePacket, RateControl, fixed_side_bits, pack_codes, unpack_codes
from .scale import DistortionScale
from .precision import install_stable_attention


@dataclass
class BTForwardResult:
    clean: torch.Tensor
    coarse: torch.Tensor
    latent: torch.Tensor
    image: torch.Tensor | None
    likelihoods: dict
    side_bits: int
    allocation_codes: torch.Tensor
    scale_info: dict = field(default_factory=dict)
    feature: torch.Tensor | None = None
    source: object = None


def pad_image(image):
    h, w = image.shape[-2:]
    ph, pw = (-h) % 256, (-w) % 256
    if not ph and not pw:
        return image
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(image, (0, pw, 0, ph), mode=mode)


class BTCodecStage(nn.Module):
    """Scratch codec pretraining; deliberately does not construct SANA."""

    def __init__(self, sana_path, codec_params=None):
        super().__init__()
        self.vae = AutoencoderDC.from_pretrained(sana_path, subfolder="vae")
        self.vae.requires_grad_(False).eval()
        install_stable_attention(vae=self.vae)
        self.codec = BTLatentCodec(**(codec_params or {}))
        self.distortion_scale = DistortionScale()
        self.backend = "C"

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_clean_latent(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self.vae.encode(image.float()).latent * self.vae.config.scaling_factor

    def forward(self, image, rate_control=None, **kwargs):
        clean = self.encode_clean_latent(image)
        source = self.codec.analyze(clean)
        view = self.codec.quantize_analysis(source, rate_control or RateControl())
        h, w = view["mean"].shape[-2:]
        return BTForwardResult(clean, view["mean"], view["mean"], None, view["likelihoods"],
                               fixed_side_bits(h//2, w//2, "C", bool(rate_control and rate_control.mixed)),
                               view["allocation_codes"], source=source)

    def export_state(self):
        return {"codec": self.codec.state_dict(), "distortion_scale": self.distortion_scale.state_dict()}

    def load_bt_state(self, state, codec_only=False):
        self.codec.load_state_dict(state["codec"], strict=True)
        self.distortion_scale.load_state_dict(state["distortion_scale"], strict=True)


class BTCodec(Codec):
    """Reuse Pure SANA loading/LoRA helpers without its GAN or old SF protocol."""

    def __init__(self, sana_path, text_path, backend="D", codec_params=None,
                 dit_rank=128, vae_rank=32, checkpoint_blocks=True,
                 representation=False, student_layer=8, teacher_layer=20, device="cpu"):
        if backend not in ("C", "D") or (representation and backend != "D"):
            raise ValueError("BT uses C or D; representation learning requires D")
        super().__init__(device=device, model_type="BTCodec", dit_path=sana_path,
                         text_encoder_path=text_path, codec_param=codec_params or {},
                         training=False, freeze_codec=False, codec_path=None,
                         DiT_params={"DiT_lora": True, "lora_rank_DiT": dit_rank, "lora_alpha_DiT": dit_rank},
                         vae_params={"vae_lora": True, "train_vae_lora": True, "decoder_lora_only": True,
                                     "lora_rank_vae": vae_rank, "lora_alpha_vae": vae_rank})
        self.backend = backend
        self.prediction_calls = 0
        self.last_compress_stats = {}
        self.last_decode_stats = {}
        self.checkpoint_blocks = checkpoint_blocks
        self.distortion_scale = DistortionScale()
        sana = unwrap_sana_transformer(self.DiT)
        install_stable_attention(sana=sana, vae=self.vae)
        hidden = int(sana.config.num_attention_heads * sana.config.attention_head_dim)
        if int(sana.config.patch_size) != 1:
            raise ValueError("BT requires one SANA token per latent cell")
        if backend == "D":
            # Must not perturb the shared pretrained/LoRA/text initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1903)
                self.quality_conditioner = ContinuousDistortionConditioner(hidden, len(sana.transformer_blocks))
        self.student_layer = int(student_layer)-1
        self.teacher_layer = int(teacher_layer)-1
        if not 0 <= self.student_layer < self.teacher_layer < len(sana.transformer_blocks):
            raise ValueError("student and teacher capture depths are out of range")
        if representation:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2903)
                self.representation_projector = RepresentationProjector(hidden, 2)
        if checkpoint_blocks:
            sana.enable_gradient_checkpointing()
            if getattr(self.vae, "_supports_gradient_checkpointing", False):
                self.vae.enable_gradient_checkpointing()

    @torch.no_grad()
    def encode_clean_latent(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            return super().encode_clean_latent(image.float())

    def set_codec_frozen(self, frozen):
        self.freeze_codec = bool(frozen)
        self.codec.requires_grad_(not frozen)
        self.codec.train(self.training and not frozen)

    def predict(self, coarse, log_rho=None, transition=1.0, capture=False):
        self.prediction_calls += 1
        time = self.self_flow_timestep(coarse.shape[0], coarse.device)
        condition = self._fixed_text_condition(coarse.shape[0], coarse.device)
        if self.backend == "C":
            prediction = self.DiT(coarse, encoder_hidden_states=condition,
                                  timestep=time, return_dict=False)[0]
            return prediction, None
        return quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse, condition, time, log_rho,
            self.quality_conditioner, conditioning_scale=transition,
            capture_layer=self.student_layer if capture else None,
            continuous_conditioning=True, checkpoint_blocks=self.checkpoint_blocks,
        )

    def repair(self, coarse, scale_info=None, transition=1.0, capture=False):
        prediction, feature = self.predict(coarse, None if self.backend == "C" else scale_info["log_rho"],
                                            transition, capture)
        if self.backend == "C":
            latent = self.sched.step(prediction, coarse)
        else:
            reference = self.sched.base_scheduler.sigmas[0].to(coarse.device)
            latent = residual_update(coarse, prediction, scale_info["scale"], reference, transition)
        image = decode_latent(self.vae, latent / self.vae.config.scaling_factor, self.checkpoint_blocks).clamp(-1, 1)
        return latent, image, feature

    def forward(self, image, rate_control=None, transition=1.0, capture=False, **kwargs):
        control = rate_control or RateControl()
        clean = self.encode_clean_latent(image)
        source = self.codec.analyze(clean)
        view = self.codec.quantize_analysis(source, control)
        coarse = view["mean"]
        scale_info = self.distortion_scale(clean, coarse) if self.backend == "D" else {}
        latent, reconstruction, feature = self.repair(coarse, scale_info, transition, capture)
        if feature is not None and self.representation_projector is not None:
            feature = self.representation_projector(feature)
        h, w = view["allocation_codes"].shape[-2:]
        return BTForwardResult(clean, coarse, latent, reconstruction, view["likelihoods"],
                               fixed_side_bits(h, w, self.backend, control.mixed),
                               view["allocation_codes"], scale_info, feature, source)

    @torch.no_grad()
    def compress(self, image, rate_control=None):
        if self.training or image.shape[0] != 1:
            raise ValueError("compress requires eval mode and a single image")
        control = rate_control or RateControl()
        original = tuple(image.shape[-2:])
        clean = self.encode_clean_latent(pad_image(image))
        encoded = self.codec.compress(clean, control)
        self.last_compress_stats = {"base_latent_mse": (encoded["mean"]-clean).square().mean().item()}
        scales = b""
        if self.backend == "D":
            info = self.distortion_scale(clean, encoded["mean"])
            self.last_compress_stats["scale_saturation"] = info["saturation"].item()
            scales = pack_codes(info["codes"].flatten().tolist(), 4)
        mask = pack_codes(control.high_mask.flatten().tolist(), 1) if control.mixed else b""
        return BTImagePacket(self.backend, original, encoded["y_shape"], encoded["hyper_shape"],
                             control.low_code, control.high_code, encoded["hyper_stream"],
                             encoded["main_stream"], encoded["allocation"], scales, mask)

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError("decompress requires eval mode")
        if isinstance(packet, bytes):
            packet = BTImagePacket.from_bytes(packet)
        if packet.backend != self.backend:
            raise ValueError("packet backend does not match the decoder")
        view = self.codec.decompress(packet)
        info = None
        if self.backend == "D":
            h, w = packet.y_shape
            codes = torch.tensor(unpack_codes(packet.scales, 4, h*w), device=view["mean"].device).reshape(1, 1, h, w)
            scale, log_rho = self.distortion_scale.decode(codes)
            info = {"scale": scale, "log_rho": log_rho}
        latent, image, _ = self.repair(view["mean"], info)
        self.last_decode_stats = {"repair_latent_rms": (latent.float()-view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return image[..., :h, :w]

    def export_state(self):
        state = {"codec": self.codec.state_dict(), "distortion_scale": self.distortion_scale.state_dict(),
                 "vae": {k: v for k, v in self.vae.state_dict().items() if "lora_" in k},
                 "dit": {k: v for k, v in self.DiT.state_dict().items() if "lora_" in k},
                 "text": self.text_proj.state_dict()}
        if self.quality_conditioner is not None:
            state["conditioner"] = self.quality_conditioner.state_dict()
        if self.representation_projector is not None:
            state["projector"] = self.representation_projector.state_dict()
        return state

    def load_bt_state(self, state, codec_only=False, initialize_conditioner=False):
        self.codec.load_state_dict(state["codec"], strict=True)
        self.distortion_scale.load_state_dict(state["distortion_scale"], strict=True)
        if codec_only:
            return
        for name, component in (("vae", self.vae), ("dit", self.DiT)):
            expected = {k for k in component.state_dict() if "lora_" in k}
            if set(state[name]) != expected:
                raise ValueError(f"incompatible {name} LoRA checkpoint")
            component.load_state_dict(state[name], strict=False)
        self.text_proj.load_state_dict(state["text"], strict=True)
        if self.quality_conditioner is not None:
            if "conditioner" in state:
                self.quality_conditioner.load_state_dict(state["conditioner"], strict=True)
            elif not initialize_conditioner:
                raise ValueError("D/E checkpoint is missing its conditioner")
        if self.representation_projector is not None and "projector" in state:
            self.representation_projector.load_state_dict(state["projector"], strict=True)


class BTCrossRateTeacher(SelfFlowTeacher):
    @torch.no_grad()
    def forward(self, coarse, fixed_text_embed, timestep, log_rho):
        condition = self.text_projection(fixed_text_embed).expand(coarse.shape[0], -1, -1)
        _, feature = quality_conditioned_sana_forward(
            self, coarse, condition, timestep, log_rho, self.quality_conditioner,
            capture_layer=self.target_layer, stop_layer=self.target_layer,
            continuous_conditioning=True,
        )
        return feature


def cross_rate_loss(student_feature, teacher_feature, teacher_rms, student_rms):
    valid = (teacher_rms < student_rms).detach().repeat_interleave(2, -2).repeat_interleave(2, -1)
    weights = valid.flatten(1).float()
    distance = 1-F.cosine_similarity(student_feature.float(), teacher_feature.detach().float(), dim=-1)
    return (distance * weights).sum() / weights.sum().clamp_min(1), weights.mean()
