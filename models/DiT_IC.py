import importlib
import os
import re

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model

from decode_modes import validate_decode_checkpoint, validate_decode_mode
from .compression_meanflow import (
    ADALN_CONDITIONING,
    ADALN_CONDITIONING_ID,
    MEANFLOW_MODE,
    MEANFLOW_MODE_ID,
    CompressionMeanFlow,
    SanaAdaLNConditioningEmbedding,
)
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
        freeze_codec=False,
        decode_mode="sana",
        meanflow_params=None,
        require_pure_init=False,
        **kwargs,
    ):
        super().__init__()

        DiT_params = DiT_params or {}
        vae_params = vae_params or {}
        codec_param = codec_param or {}
        loss_params = loss_params or {}
        self.decode_mode = validate_decode_mode(decode_mode)
        self._init_for_training = training
        self.require_pure_init = require_pure_init
        if freeze_codec and codec_path is None:
            raise ValueError("freeze_codec=True requires a pretrained codec_path")

        meanflow_params = dict(meanflow_params or {})
        enabled_flag = meanflow_params.get("enabled")
        configured_mode = meanflow_params.get(
            "mode",
            MEANFLOW_MODE if enabled_flag else "disabled",
        )
        if enabled_flag is False:
            configured_mode = "disabled"
        self.meanflow_enabled = configured_mode != "disabled"
        self.meanflow_weight = float(meanflow_params.get("weight", 0.05))
        self.latest_meanflow_loss = None
        self.latest_meanflow_stats = None
        if self.meanflow_enabled:
            if configured_mode != MEANFLOW_MODE:
                raise ValueError(
                    "Pure SANA supports only MeanFlow 5A mode='meanflow'"
                )
            if meanflow_params.get(
                "conditioning_mode",
                ADALN_CONDITIONING,
            ) != ADALN_CONDITIONING:
                raise ValueError(
                    "Pure SANA MeanFlow supports only "
                    "conditioning_mode='adaln'"
                )
            if self.decode_mode != "sana":
                raise ValueError(
                    "MeanFlow AdaLN requires decode_mode='sana'"
                )
            if self.meanflow_weight <= 0:
                raise ValueError("MeanFlow weight must be positive")

        if use_merge:
            vae_config = AutoencoderDC.load_config(dit_path, subfolder="vae")
            self.vae = AutoencoderDC.from_config(vae_config)
        else:
            self.vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")
        self.vae.requires_grad_(False)

        self.build_DiT(dit_path, device, use_merge)
        self.codec = latent_models(model_type, **codec_param)
        self.freeze_codec = freeze_codec
        if self.freeze_codec:
            self.codec.requires_grad_(False)
        self._build_fixed_text_condition(text_encoder_path)
        if self.meanflow_enabled:
            adaln_embedding_dim = (
                int(self.DiT.config.num_attention_heads)
                * int(self.DiT.config.attention_head_dim)
            )
            self._enable_meanflow_adaln_conditioning(
                adaln_embedding_dim,
            )
            self.meanflow = CompressionMeanFlow(
                mode=configured_mode,
                conditioning_mode=meanflow_params.get(
                    "conditioning_mode",
                    ADALN_CONDITIONING,
                ),
                adaln_embedding_dim=adaln_embedding_dim,
                frequency_embedding_size=int(
                    meanflow_params.get(
                        "frequency_embedding_size",
                        256,
                    )
                ),
                equal_time_ratio=float(
                    meanflow_params.get("equal_time_ratio", 0.5)
                ),
                time_distribution=meanflow_params.get(
                    "time_distribution",
                    "logit_normal",
                ),
                logit_mean=float(
                    meanflow_params.get("logit_mean", -0.4)
                ),
                logit_std=float(
                    meanflow_params.get("logit_std", 1.0)
                ),
                norm_p=float(meanflow_params.get("norm_p", 1.0)),
                norm_eps=float(
                    meanflow_params.get("norm_eps", 0.01)
                ),
            )
        else:
            self.meanflow = None
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
                self.load_ckpt_dict(
                    codec_path,
                    use_ema,
                    require_pure_source=(
                        training and self.require_pure_init
                    ),
                )

        if self.freeze_codec:
            self.codec.eval()
            self.assert_frozen_codec_contract()

        print("Pure SANA initialization complete")

    def _build_fixed_text_condition(self, text_encoder_path):
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must point to a fixed text embedding")

        if os.path.isdir(text_encoder_path):
            candidates = [
                os.path.join(
                    os.path.dirname(text_encoder_path),
                    "fixed_text_embed_gemma.pt",
                ),
                os.path.join(text_encoder_path, "fixed_text_embed_gemma.pt"),
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
        meanflow_state = {
            key[len("meanflow."):]: value
            for key, value in model_state.items()
            if key.startswith("meanflow.")
        }
        self._validate_meanflow_checkpoint(
            bool(meanflow_state),
            "merged checkpoint",
            meanflow_state.get("training_mode_id"),
            meanflow_state.get("conditioning_mode_id"),
        )
        if self.meanflow_enabled and not meanflow_state:
            prefix = "DiT.time_embed.emb."
            remapped_state = {}
            for key, value in model_state.items():
                if (
                    key.startswith(prefix)
                    and not key.startswith(prefix + "base_embedding.")
                ):
                    key = (
                        prefix
                        + "base_embedding."
                        + key[len(prefix):]
                    )
                remapped_state[key] = value
            model_state = remapped_state
        validate_decode_checkpoint(
            self.decode_mode,
            any(key.startswith("g_s_fusion.") for key in model_state),
            training=self._init_for_training,
            allow_missing_gs_fusion=False,
            source="merged checkpoint",
        )
        self.codec.load_state_dict(codec_state, strict=False)
        self.load_state_dict(model_state, strict=False)
        if meanflow_state:
            self.meanflow.load_state_dict(meanflow_state)

    def _validate_meanflow_checkpoint(
        self,
        has_state,
        source,
        checkpoint_mode_id=None,
        checkpoint_conditioning_mode_id=None,
        require_pure_source=False,
    ):
        if require_pure_source and has_state:
            raise ValueError(
                "Frozen MeanFlow comparison must start from a Pure SANA "
                "checkpoint without MeanFlow state"
            )
        if self.meanflow_enabled and not has_state:
            if self._init_for_training:
                print(
                    f"[MeanFlow] {source} has no AdaLN adapter; "
                    "using zero initialization."
                )
                return
            raise ValueError(
                f"MeanFlow inference was requested, but {source} has no "
                "MeanFlow state"
            )
        if not self.meanflow_enabled and has_state:
            raise ValueError(
                f"{source} contains MeanFlow weights, but MeanFlow is "
                "disabled in the model config"
            )
        if not has_state:
            return
        if checkpoint_mode_id is None:
            raise ValueError(
                f"{source} has no MeanFlow training-mode provenance"
            )
        if int(checkpoint_mode_id.item()) != MEANFLOW_MODE_ID:
            raise ValueError(
                f"{source} is not a MeanFlow 5A checkpoint"
            )
        if checkpoint_conditioning_mode_id is None:
            raise ValueError(
                f"{source} has no MeanFlow conditioning provenance"
            )
        if (
            int(checkpoint_conditioning_mode_id.item())
            != ADALN_CONDITIONING_ID
        ):
            raise ValueError(
                f"{source} is not an AdaLN-conditioned MeanFlow checkpoint"
            )

    def merge_lora(self):
        if hasattr(self.vae, "merge_and_unload"):
            self.vae = self.vae.merge_and_unload()
        if hasattr(self.DiT, "merge_and_unload"):
            self.DiT = self.DiT.merge_and_unload()

    def load_ckpt_dict(
        self,
        codec_path,
        use_ema=False,
        require_pure_source=False,
    ):
        print("[Pure SANA] Loading trained weights")
        checkpoint = torch.load(codec_path, map_location="cpu", weights_only=True)
        state_key = "ema" if use_ema else "model"
        if state_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain {state_key!r}: {codec_path}")
        state = checkpoint[state_key]
        required = {
            "state_dict_codec",
            "state_dict_vae",
            "state_dict_DiT",
            "state_dict_text_proj",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(
                "Checkpoint is not a fixed-text Pure SANA/all-off checkpoint; "
                f"missing entries: {missing}"
            )
        has_gs_fusion = "state_dict_g_s_fusion" in state
        validate_decode_checkpoint(
            self.decode_mode,
            has_gs_fusion,
            training=self._init_for_training,
            allow_missing_gs_fusion=False,
            source=f"checkpoint {codec_path!r}",
        )
        meanflow_state = state.get("state_dict_meanflow")
        self._validate_meanflow_checkpoint(
            meanflow_state is not None,
            f"checkpoint {codec_path!r}",
            (
                None
                if meanflow_state is None
                else meanflow_state.get("training_mode_id")
            ),
            (
                None
                if meanflow_state is None
                else meanflow_state.get("conditioning_mode_id")
            ),
            require_pure_source=require_pure_source,
        )

        codec_result = self.codec.load_state_dict(
            state["state_dict_codec"],
            strict=False,
        )
        active_codec_missing = [
            key
            for key in codec_result.missing_keys
            if not key.endswith("_quantized_cdf")
            and not key.endswith("_cdf_length")
            and not key.endswith("_offset")
            and not key.endswith("scale_table")
        ]
        if active_codec_missing:
            raise RuntimeError(
                "Checkpoint is missing active Pure SANA codec weights: "
                + ", ".join(active_codec_missing)
            )
        self.codec.update()
        expected_vae_lora = {
            key for key in self.vae.state_dict() if "lora" in key
        }
        expected_dit_lora = {
            key for key in self.DiT.state_dict() if "lora" in key
        }
        missing_adapter_keys = sorted(
            expected_vae_lora.difference(state["state_dict_vae"])
            | expected_dit_lora.difference(state["state_dict_DiT"])
        )
        if missing_adapter_keys:
            raise RuntimeError(
                "Checkpoint is missing configured LoRA weights: "
                + ", ".join(missing_adapter_keys)
            )
        vae_result = self.vae.load_state_dict(
            state["state_dict_vae"],
            strict=False,
        )
        dit_result = self.DiT.load_state_dict(
            state["state_dict_DiT"],
            strict=False,
        )
        unexpected_adapter_keys = (
            list(vae_result.unexpected_keys) + list(dit_result.unexpected_keys)
        )
        if unexpected_adapter_keys:
            raise RuntimeError(
                "Checkpoint LoRA keys do not match the configured VAE/DiT: "
                + ", ".join(unexpected_adapter_keys)
            )
        self.text_proj.load_state_dict(state["state_dict_text_proj"])
        if meanflow_state is not None:
            self.meanflow.load_state_dict(meanflow_state)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_codec:
            self.codec.eval()
        return self

    def assert_frozen_codec_contract(self):
        if not self.freeze_codec:
            return
        trainable = [
            name for name, parameter in self.codec.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            preview = ", ".join(trainable[:5])
            raise RuntimeError(
                f"freeze_codec=True but codec parameters remain trainable: {preview}"
            )
        if self.codec.training:
            raise RuntimeError("freeze_codec=True requires codec.eval()")

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

    def _enable_meanflow_adaln_conditioning(self, embedding_dim):
        if getattr(self.DiT.config, "guidance_embeds", False):
            raise ValueError(
                "MeanFlow AdaLN conditioning requires "
                "SANA guidance_embeds=False"
            )
        if not hasattr(self.DiT.time_embed, "emb"):
            raise RuntimeError(
                "The loaded SANA model does not expose "
                "AdaLayerNormSingle.emb"
            )
        self.DiT.time_embed.emb = SanaAdaLNConditioningEmbedding(
            self.DiT.time_embed.emb,
            embedding_dim,
        )

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
        if self.meanflow_enabled:
            parameters.extend(self.meanflow.parameters())
        return parameters

    def get_state_dicts_for_save(self):
        codec = getattr(self.codec, "module", self.codec)
        vae = getattr(self.vae, "module", self.vae)
        dit = getattr(self.DiT, "module", self.DiT)
        text_proj = getattr(self.text_proj, "module", self.text_proj)
        state_dicts = {
            "state_dict_codec": codec.state_dict(),
            "state_dict_vae": {
                key: value for key, value in vae.state_dict().items() if "lora" in key
            },
            "state_dict_DiT": {
                key: value for key, value in dit.state_dict().items() if "lora" in key
            },
            "state_dict_text_proj": text_proj.state_dict(),
        }
        if self.meanflow_enabled:
            meanflow = getattr(self.meanflow, "module", self.meanflow)
            state_dicts["state_dict_meanflow"] = meanflow.state_dict()
        return state_dicts

    def get_last_layer(self):
        codec_last_layer = self.codec.g_s.synthesis_transform[-1].weight
        if codec_last_layer.requires_grad:
            return codec_last_layer

        decoder_lora = [
            parameter
            for name, parameter in self.vae.named_parameters()
            if parameter.requires_grad
            and "decoder" in name.lower()
            and "lora" in name.lower()
        ]
        if decoder_lora:
            return decoder_lora[-1]

        trainable = self.get_trainable_params()
        return trainable[-1] if trainable else None

    def _fixed_text_condition(self, batch_size, device):
        condition = self.text_proj(self.fixed_text_embed).expand(
            batch_size,
            -1,
            -1,
        )
        return condition.to(device)

    def _predict_mean_velocity(
        self,
        latent,
        t,
        r,
        base_condition,
    ):
        interval_embedding = self.meanflow.condition_adaln(t, r)
        timestep = self.sched.base_scheduler.timesteps.expand(latent.shape[0])
        timestep = timestep * self.DiT.config.timestep_scale
        timestep = torch.cat(
            (
                timestep.to(
                    device=interval_embedding.device,
                    dtype=interval_embedding.dtype,
                )[:, None],
                interval_embedding,
            ),
            dim=1,
        )
        model_prediction = self.DiT(
            latent,
            encoder_hidden_states=base_condition,
            timestep=timestep,
            return_dict=False,
        )[0]
        sigma = self.sched.base_scheduler.sigmas[0].to(
            device=model_prediction.device,
            dtype=model_prediction.dtype,
        )
        return sigma * model_prediction

    def _meanflow_endpoint(self, latent, base_condition):
        batch_size = latent.shape[0]
        t = torch.ones(
            batch_size,
            device=latent.device,
            dtype=latent.dtype,
        )
        r = torch.zeros_like(t)
        return latent - self._predict_mean_velocity(
            latent,
            t,
            r,
            base_condition,
        )

    def _decode_with_sana(self, latent):
        condition = self._fixed_text_condition(
            latent.shape[0],
            latent.device,
        )
        if self.meanflow_enabled:
            denoised = self._meanflow_endpoint(latent, condition)
        else:
            timestep = self.sched.base_scheduler.timesteps.expand(
                latent.shape[0]
            )
            timestep = timestep * self.DiT.config.timestep_scale
            model_prediction = self.DiT(
                latent,
                encoder_hidden_states=condition,
                timestep=timestep.to(latent.device),
                return_dict=False,
            )[0]
            denoised = self.sched.step(model_prediction, latent)
        return self.vae.decode(
            denoised / self.vae.config.scaling_factor,
            return_dict=False,
        )[0].clamp(-1, 1)

    def forward(self, image):
        self.latest_meanflow_loss = None
        self.latest_meanflow_stats = None
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        codec_output = self.codec(latent)
        compressed_latent = codec_output["mean"]
        output_image = self._decode_with_sana(compressed_latent)
        if self.meanflow_enabled and self.training:
            base_condition = self._fixed_text_condition(
                compressed_latent.shape[0],
                compressed_latent.device,
            )
            meanflow_loss, stats = self.meanflow.compute_loss(
                lambda z, t, r, batch_indices: self._predict_mean_velocity(
                    z,
                    t,
                    r,
                    base_condition.index_select(0, batch_indices),
                ),
                latent,
                compressed_latent,
            )
            self.latest_meanflow_loss = (
                self.meanflow_weight * meanflow_loss
            )
            stats["weighted_loss"] = (
                self.latest_meanflow_loss.detach()
            )
            self.latest_meanflow_stats = stats
        return output_image, codec_output["likelihoods"]

    def compress(self, image):
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        return self.codec.compress(latent)

    def decompress(self, strings, shape):
        latent = self.codec.decompress(strings, shape)
        return self._decode_with_sana(latent)
