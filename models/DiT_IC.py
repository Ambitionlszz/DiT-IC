import importlib
import os
import re

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model

from decode_modes import validate_decode_checkpoint, validate_decode_mode
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
        train_text_projection=True,
        strict_stage_loading=False,
        decode_mode="sana",
        **kwargs,
    ):
        super().__init__()

        DiT_params = DiT_params or {}
        vae_params = vae_params or {}
        codec_param = codec_param or {}
        loss_params = loss_params or {}
        self.decode_mode = validate_decode_mode(decode_mode)
        self._init_for_training = training
        self.train_text_projection = bool(train_text_projection)
        self.strict_stage_loading = bool(strict_stage_loading)
        vae_lora_enabled = bool(vae_params.get("vae_lora", False))
        train_vae_lora = bool(
            vae_params.get("train_vae_lora", vae_lora_enabled)
        )
        if train_vae_lora and not vae_lora_enabled:
            raise ValueError("train_vae_lora=True requires vae_lora=True")
        if train_vae_lora and not vae_params.get("decoder_lora_only", False):
            raise ValueError("trainable VAE LoRA must be decoder-only")
        self.freeze_vae = not train_vae_lora
        if freeze_codec and codec_path is None:
            raise ValueError("freeze_codec=True requires a pretrained codec_path")

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
        if not self.train_text_projection:
            self.text_proj.requires_grad_(False)
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
                )

        if self.freeze_codec:
            self.codec.eval()
            self.assert_frozen_codec_contract()
        if self.freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()
        else:
            self._set_vae_encoder_eval()
        self.assert_reconstruction_training_contract()

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
        if any(key.startswith("meanflow.") for key in model_state):
            raise ValueError(
                "Merged checkpoint contains MeanFlow state; "
                "use the meanflow branch."
            )
        validate_decode_checkpoint(
            self.decode_mode,
            any(key.startswith("g_s_fusion.") for key in model_state),
            training=self._init_for_training,
            allow_missing_gs_fusion=False,
            source="merged checkpoint",
        )
        self.codec.load_state_dict(codec_state, strict=False)
        self.load_state_dict(model_state, strict=False)

    def merge_lora(self):
        if hasattr(self.vae, "merge_and_unload"):
            self.vae = self.vae.merge_and_unload()
        if hasattr(self.DiT, "merge_and_unload"):
            self.DiT = self.DiT.merge_and_unload()

    def load_ckpt_dict(
        self,
        codec_path,
        use_ema=False,
    ):
        print("[Pure SANA] Loading trained weights")
        checkpoint = torch.load(codec_path, map_location="cpu", weights_only=True)
        checkpoint_stage = checkpoint.get("stage")
        if self.strict_stage_loading:
            allowed_stages = {
                "variable_quality_codec",
                "multirate_reconstruction",
            }
            if checkpoint_stage not in allowed_stages:
                raise ValueError(
                    "strict_stage_loading requires a variable-rate checkpoint; "
                    f"got stage={checkpoint_stage!r} from {codec_path!r}"
                )
            if checkpoint_stage == "variable_quality_codec":
                if (
                    int(checkpoint.get("schema_version", 0)) not in (8, 9)
                    or checkpoint.get("variable_rate_mechanism")
                    != "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
                ):
                    raise ValueError(
                        "multi-rate Plain requires a schema-version 8 or 9 LIC "
                        "checkpoint with endpoint-interpolated q_e/q_d and "
                        "complete w_e/w_d control"
                    )
                global_step = int(checkpoint.get("global_step", -1))
                phase_a_steps = int(checkpoint.get("phase_a_steps", -1))
                if phase_a_steps < 1 or global_step <= phase_a_steps:
                    raise ValueError(
                        "multi-rate Plain requires a LIC checkpoint trained "
                        "past the Phase A boundary"
                    )
            elif (
                int(checkpoint.get("schema_version", 0)) != 3
                or checkpoint.get("variable_rate_mechanism")
                != "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
            ):
                raise ValueError(
                    "multi-rate Plain requires a schema-version 3 P0 checkpoint "
                    "derived from the complete q_e/q_d and w_e/w_d LIC"
                )
        state_key = "ema" if use_ema else "model"
        if state_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain {state_key!r}: {codec_path}")
        state = checkpoint[state_key]
        if checkpoint_stage == "variable_quality_codec":
            if "state_dict_codec" not in state:
                raise KeyError("Stage 1 checkpoint lacks state_dict_codec")
            self.codec.load_state_dict(state["state_dict_codec"], strict=True)
            self.codec.update()
            print(
                "[Pure SANA] Loaded frozen V4 LIC; keeping fresh DiT LoRA "
                "and text projection initialization"
            )
            return

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
        if "state_dict_meanflow" in state:
            raise ValueError(
                f"checkpoint {codec_path!r} contains MeanFlow state; "
                "use the meanflow branch."
            )

        codec_result = self.codec.load_state_dict(
            state["state_dict_codec"],
            strict=checkpoint_stage == "multirate_reconstruction",
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

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_codec:
            self.codec.eval()
        if self.freeze_vae:
            self.vae.eval()
        else:
            self._set_vae_encoder_eval()
        return self

    def _set_vae_encoder_eval(self):
        encoders = [
            module
            for name, module in self.vae.named_modules()
            if name == "encoder" or name.endswith(".encoder")
        ]
        if len(encoders) != 1:
            raise RuntimeError(
                "decoder-LoRA training requires exactly one VAE encoder module"
            )
        encoders[0].requires_grad_(False)
        encoders[0].eval()

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

    def assert_reconstruction_training_contract(self):
        if not self.freeze_codec:
            return
        self.assert_frozen_codec_contract()
        trainable_vae = [
            name for name, parameter in self.vae.named_parameters()
            if parameter.requires_grad
        ]
        if self.freeze_vae and trainable_vae:
            raise RuntimeError(
                "multi-rate Plain training requires a frozen VAE; trainable: "
                + ", ".join(trainable_vae[:5])
            )
        if not self.freeze_vae:
            unexpected_vae = [
                name
                for name in trainable_vae
                if "decoder." not in name or "lora_" not in name
            ]
            if unexpected_vae:
                raise RuntimeError(
                    "multi-rate Plain permits only decoder LoRA in the VAE; "
                    "trainable: " + ", ".join(unexpected_vae[:5])
                )
            if not trainable_vae:
                raise RuntimeError(
                    "decoder-LoRA training requires trainable VAE decoder LoRA"
                )
        if self._init_for_training:
            if not self.train_text_projection:
                raise RuntimeError(
                    "multi-rate Plain training requires a trainable text projection"
                )
            frozen_text = [
                name for name, parameter in self.text_proj.named_parameters()
                if not parameter.requires_grad
            ]
            if frozen_text:
                raise RuntimeError(
                    "text projection parameters unexpectedly frozen: "
                    + ", ".join(frozen_text)
                )
            trainable_dit = [
                name for name, parameter in self.DiT.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable_dit or any(
                "lora" not in name.lower() for name in trainable_dit
            ):
                raise RuntimeError(
                    "multi-rate Plain training requires only DiT LoRA parameters "
                    "to be trainable"
                )

    def assert_end_to_end_variable_rate_contract(self):
        if self.freeze_codec:
            raise RuntimeError(
                "end-to-end variable-rate training requires freeze_codec=False"
            )
        trainable_codec = [
            name for name, parameter in self.codec.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_codec:
            raise RuntimeError(
                "end-to-end variable-rate training requires a trainable codec"
            )
        frozen_codec = [
            name for name, parameter in self.codec.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen_codec:
            raise RuntimeError(
                "end-to-end variable-rate training requires every codec "
                "parameter to be trainable: " + ", ".join(frozen_codec[:5])
            )

        self._set_vae_encoder_eval()
        trainable_vae = [
            name for name, parameter in self.vae.named_parameters()
            if parameter.requires_grad
        ]
        unexpected_vae = [
            name for name in trainable_vae
            if "decoder." not in name or "lora_" not in name
        ]
        if unexpected_vae or not trainable_vae:
            raise RuntimeError(
                "end-to-end variable-rate training permits only trainable "
                "decoder LoRA in the VAE"
            )

        trainable_dit = [
            name for name, parameter in self.DiT.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_dit or any(
            "lora" not in name.lower() for name in trainable_dit
        ):
            raise RuntimeError(
                "end-to-end variable-rate training requires only DiT LoRA "
                "parameters to be trainable"
            )
        if any(
            not parameter.requires_grad
            for parameter in self.text_proj.parameters()
        ):
            raise RuntimeError(
                "end-to-end variable-rate training requires a trainable "
                "text projection"
            )

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
        parameters.extend(
            parameter
            for parameter in self.text_proj.parameters()
            if parameter.requires_grad
        )
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

    def _decode_with_sana(self, latent):
        condition = self._fixed_text_condition(
            latent.shape[0],
            latent.device,
        )
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

    def encode_clean_latent(self, image):
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        return latent

    def decode_with_model(self, latent):
        return self._decode_with_sana(latent)

    def forward(self, image, quality_index=None):
        latent = self.encode_clean_latent(image)
        codec_output = self.codec(latent, quality_index)
        compressed_latent = codec_output["mean"]
        output_image = self.decode_with_model(compressed_latent)
        return output_image, codec_output["likelihoods"]

    def compress(self, image, quality_index=None):
        latent = self.encode_clean_latent(image)
        return self.codec.compress(latent, quality_index)

    def decompress(self, packet_or_strings, shape=None, quality_index=None):
        latent = self.codec.decompress(
            packet_or_strings,
            shape=shape,
            quality_index=quality_index,
        )
        return self.decode_with_model(latent)
