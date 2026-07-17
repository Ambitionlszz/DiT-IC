import importlib
import os
import re

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model

from .losses import LPIPSWithDiscriminator
from .scheduler import OneStepFlowScheduler, make_1step_sched


def filter_supported_modules(model):
    pattern = re.compile(
        r"^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv_inverted|"
        r"conv_point|to_k|to_q|to_v|to_out\.0)$"
    )
    supported = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return [
        name
        for name, module in model.named_modules()
        if pattern.match(name) and isinstance(module, supported)
    ]


def latent_models(model, **kwargs):
    model_map = {
        "Codec": "models.latent_codec.LatentCodec",
    }
    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)


class Codec(nn.Module):
    """Pure SANA image-compression pipeline with a fixed text condition."""

    def __init__(
        self,
        device="cuda",
        model_type="Codec",
        dit_path=None,
        codec_path=None,
        use_ema=False,
        use_merge=False,
        training=False,
        DiT_params=None,
        vae_params=None,
        codec_param=None,
        loss_params=None,
        text_encoder_path=None,
        **kwargs,
    ):
        super().__init__()

        DiT_params = DiT_params or {}
        vae_params = vae_params or {}
        codec_param = codec_param or {}
        loss_params = loss_params or {}

        if use_merge:
            vae_config = AutoencoderDC.load_config(dit_path, subfolder="vae")
            self.vae = AutoencoderDC.from_config(vae_config)
        else:
            self.vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")
        self.vae.requires_grad_(False)

        self.build_DiT(dit_path, device, use_merge)
        self.codec = latent_models(model_type, **codec_param)
        self._build_fixed_text_condition(text_encoder_path)
        self.loss = (
            LPIPSWithDiscriminator(device, **loss_params)
            if training
            else None
        )

        if use_merge:
            if codec_path is None:
                raise ValueError("codec_path is required when use_merge=True")
            self._load_merged_state_dict(codec_path)
            if training and vae_params.get("vae_lora", False):
                self.build_vae_lora(
                    vae_params["lora_rank_vae"],
                    vae_params["lora_alpha_vae"],
                )
            if training and DiT_params.get("DiT_lora", False):
                self.build_DiT_lora(
                    DiT_params["lora_rank_DiT"],
                    DiT_params["lora_alpha_DiT"],
                )
        else:
            if vae_params.get("vae_lora", False):
                self.build_vae_lora(
                    vae_params["lora_rank_vae"],
                    vae_params["lora_alpha_vae"],
                )
            if DiT_params.get("DiT_lora", False):
                self.build_DiT_lora(
                    DiT_params["lora_rank_DiT"],
                    DiT_params["lora_alpha_DiT"],
                )
            if codec_path is not None:
                self.load_ckpt_dict(codec_path, use_ema)

        print("Pure SANA initialization complete")

    def _build_fixed_text_condition(self, text_encoder_path):
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must point to a fixed text embedding")

        if os.path.isdir(text_encoder_path):
            candidates = [
                os.path.join(text_encoder_path, "fixed_text_embed_gemma.pt"),
                os.path.join(
                    os.path.dirname(text_encoder_path),
                    "fixed_text_embed_gemma.pt",
                ),
            ]
            embed_path = next((path for path in candidates if os.path.exists(path)), None)
        else:
            embed_path = text_encoder_path if os.path.exists(text_encoder_path) else None

        if embed_path is None:
            raise FileNotFoundError(
                "Could not find fixed_text_embed_gemma.pt from "
                f"text_encoder_path={text_encoder_path!r}"
            )

        embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
        self.register_buffer("fixed_text_embed", embed_data["fixed_text_embed"])
        self.text_proj = nn.Linear(
            embed_data["hidden_size"],
            self.DiT.config.caption_channels,
        )
        print(
            f"[TextCond] Loaded fixed embedding from {embed_path}, "
            f"shape={list(self.fixed_text_embed.shape)}"
        )

    def _load_merged_state_dict(self, codec_path):
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        codec_state = {}
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("codec."):
                codec_state[key[6:]] = value
            else:
                model_state[key] = value
        self.codec.load_state_dict(codec_state, strict=False)
        self.load_state_dict(model_state, strict=False)

    def merge_lora(self):
        if hasattr(self.vae, "merge_and_unload"):
            self.vae = self.vae.merge_and_unload()
        if hasattr(self.DiT, "merge_and_unload"):
            self.DiT = self.DiT.merge_and_unload()

    def load_ckpt_dict(self, codec_path, use_ema=False):
        print("[Pure SANA] Loading trained weights")
        checkpoint = torch.load(codec_path, map_location="cpu", weights_only=True)
        state = checkpoint["ema" if use_ema else "model"]

        self.codec.load_state_dict(state["state_dict_codec"], strict=False)
        self.codec.update()
        self.vae.load_state_dict(state["state_dict_vae"], strict=False)
        self.DiT.load_state_dict(state["state_dict_DiT"], strict=False)
        self.text_proj.load_state_dict(state["state_dict_text_proj"])

    def build_DiT(self, dit_path, device="cuda", use_merge=False):
        base_scheduler = make_1step_sched(dit_path, device)
        self.sched = OneStepFlowScheduler(base_scheduler)

        if use_merge:
            config = SanaTransformer2DModel.load_config(
                dit_path,
                subfolder="transformer",
            )
            self.DiT = SanaTransformer2DModel.from_config(config)
        else:
            self.DiT = SanaTransformer2DModel.from_pretrained(
                dit_path,
                subfolder="transformer",
            )
        self.DiT.requires_grad_(False)

    def build_vae_lora(self, lora_rank_vae, lora_alpha_vae):
        config = LoraConfig(
            r=lora_rank_vae,
            lora_alpha=lora_alpha_vae,
            target_modules=filter_supported_modules(self.vae),
            bias="none",
            init_lora_weights="gaussian",
        )
        self.vae = get_peft_model(self.vae, config)
        self.vae.requires_grad_(False)
        for name, parameter in self.vae.named_parameters():
            if "lora" in name:
                parameter.requires_grad = True
        print("VAE-LoRA ready")

    def build_DiT_lora(self, lora_rank_DiT, lora_alpha_DiT):
        target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
            "conv_inverted",
            "conv_point",
        ]
        config = LoraConfig(
            r=lora_rank_DiT,
            lora_alpha=lora_alpha_DiT,
            target_modules=target_modules,
            bias="none",
            init_lora_weights="gaussian",
        )
        self.DiT = get_peft_model(self.DiT, config)
        self.DiT.requires_grad_(False)
        for name, parameter in self.DiT.named_parameters():
            if "lora" in name:
                parameter.requires_grad = True
        print("DiT-LoRA ready")

    def get_entropy_parameters(self):
        return [
            parameter
            for name, parameter in self.codec.named_parameters()
            if name.endswith(".quantiles") and parameter.requires_grad
        ]

    def get_trainable_params(self):
        parameters = [
            parameter
            for name, parameter in self.codec.named_parameters()
            if not name.endswith(".quantiles") and parameter.requires_grad
        ]
        parameters.extend(
            parameter
            for name, parameter in self.vae.named_parameters()
            if "lora" in name and parameter.requires_grad
        )
        parameters.extend(
            parameter
            for name, parameter in self.DiT.named_parameters()
            if "lora" in name and parameter.requires_grad
        )
        parameters.extend(self.text_proj.parameters())
        return parameters

    def get_state_dicts_for_save(self):
        codec = getattr(self.codec, "module", self.codec)
        vae = getattr(self.vae, "module", self.vae)
        dit = getattr(self.DiT, "module", self.DiT)
        text_proj = getattr(self.text_proj, "module", self.text_proj)
        return {
            "state_dict_codec": codec.state_dict(),
            "state_dict_vae": {
                key: value for key, value in vae.state_dict().items() if "lora" in key
            },
            "state_dict_DiT": {
                key: value for key, value in dit.state_dict().items() if "lora" in key
            },
            "state_dict_text_proj": text_proj.state_dict(),
        }

    def get_last_layer(self):
        return self.codec.g_s.synthesis_transform[-1].weight

    def _decode_with_sana(self, latent):
        condition = self.text_proj(self.fixed_text_embed).expand(
            latent.shape[0],
            -1,
            -1,
        )
        timestep = self.sched.base_scheduler.timesteps.expand(latent.shape[0])
        timestep = timestep * self.DiT.config.timestep_scale
        model_prediction = self.DiT(
            latent,
            encoder_hidden_states=condition.to(latent.device),
            timestep=timestep.to(latent.device),
            return_dict=False,
        )[0]
        denoised = self.sched.step(model_prediction, latent)
        return self.vae.decode(
            denoised / self.vae.config.scaling_factor,
            return_dict=False,
        )[0].clamp(-1, 1)

    def forward(self, image):
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        codec_output = self.codec(latent)
        output_image = self._decode_with_sana(codec_output["mean"])
        return output_image, codec_output["likelihoods"]

    def compress(self, image):
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        return self.codec.compress(latent)

    def decompress(self, strings, shape):
        latent = self.codec.decompress(strings, shape)
        return self._decode_with_sana(latent)
