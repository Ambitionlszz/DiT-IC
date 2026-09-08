import importlib
import os
import re

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model

from decode_modes import validate_decode_checkpoint, validate_decode_mode
from .feature_relay import DecoderFeatureRelay, RELAY_MODES
from .losses import LPIPSWithDiscriminator
from .sana_self_flow import (
    RepresentationProjector,
    SelfFlowTeacher,
    TokenQualityConditioner,
    quality_conditioned_sana_forward,
    mix_dual_rate_tokens,
    unwrap_sana_transformer,
)
from .scheduler import OneStepFlowScheduler, make_1step_sched


SELF_FLOW_STAGE = "dual_rate_self_flow_reconstruction"
SELF_FLOW_SCHEMA_VERSION = 4
SELF_FLOW_MECHANISM = "adjacent_dual_rate_token_mix_ema_l8_l20_v1"


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
        allow_phase_a_handoff=False,
        decode_mode="sana",
        decoder_relay_params=None,
        self_flow_params=None,
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
        self.allow_phase_a_handoff = bool(allow_phase_a_handoff)
        self.self_flow_params = dict(self_flow_params or {})
        self.self_flow_enabled = bool(
            self.self_flow_params.get("enabled", False)
        )
        self.quality_conditioner = None
        self.representation_projector = None
        if self.self_flow_enabled and use_merge:
            raise ValueError("dual-rate Self-Flow does not support use_merge=True")
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
        self.decoder_relay = self._build_decoder_relay(decoder_relay_params)
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
            self._build_self_flow_modules()
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
            self._build_self_flow_modules()
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

    def _build_self_flow_modules(self):
        if not self.self_flow_enabled:
            return
        sana = unwrap_sana_transformer(self.DiT)
        if int(sana.config.patch_size) != 1:
            raise ValueError(
                "dual-rate Self-Flow requires SANA patch_size=1"
            )
        num_layers = len(sana.transformer_blocks)
        student_layer = int(self.self_flow_params.get("student_layer", 8))
        teacher_layer = int(self.self_flow_params.get("teacher_layer", 20))
        if not 1 <= student_layer < teacher_layer <= num_layers:
            raise ValueError(
                "Self-Flow layers must satisfy 1 <= student < teacher <= depth"
            )
        self.self_flow_student_layer = student_layer - 1
        self.self_flow_teacher_layer = teacher_layer - 1
        hidden_dim = int(
            sana.config.num_attention_heads * sana.config.attention_head_dim
        )
        num_levels = int(self.self_flow_params.get("num_quality_levels", 6))
        self.quality_conditioner = TokenQualityConditioner(
            num_levels=num_levels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=int(
                self.self_flow_params.get("quality_embedding_dim", 128)
            ),
        )
        if (
            self._init_for_training
            and bool(
                self.self_flow_params.get("representation_enabled", True)
            )
        ):
            self.representation_projector = RepresentationProjector(
                hidden_dim=hidden_dim,
                expansion=int(
                    self.self_flow_params.get("projector_expansion", 2)
                ),
            )
        if not self._init_for_training:
            self.quality_conditioner.requires_grad_(False)
        print(
            "[SelfFlow] resolved Student layer "
            f"{student_layer} (index {self.self_flow_student_layer}), Teacher "
            f"layer {teacher_layer} (index {self.self_flow_teacher_layer}), "
            f"token grid patch_size={sana.config.patch_size}"
        )

    def _build_decoder_relay(self, relay_params):
        relay_params = dict(relay_params or {})
        mode = relay_params.pop("mode", "none")
        if mode == "none":
            self.decoder_relay_mode = "none"
            return None
        if mode not in RELAY_MODES:
            raise ValueError(
                f"decoder_relay_params.mode must be 'none' or one of "
                f"{RELAY_MODES}, got {mode!r}"
            )

        stage_indices = tuple(
            relay_params.pop("stage_block_indices", (3, 2, 1))
        )
        decoder_channels = tuple(self.vae.config.decoder_block_out_channels)
        if any(
            index < 0 or index >= len(decoder_channels)
            for index in stage_indices
        ):
            raise ValueError(
                "decoder relay stage_block_indices are incompatible with the VAE"
            )
        stage_channels = [decoder_channels[index] for index in stage_indices]

        # Relay construction must not perturb downstream LoRA initialization.
        cpu_rng_state = torch.random.get_rng_state()
        try:
            relay = DecoderFeatureRelay(
                mode=mode,
                stage_channels=stage_channels,
                stage_block_indices=stage_indices,
                **relay_params,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
        self.decoder_relay_mode = mode
        return relay

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
                SELF_FLOW_STAGE,
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
                before_handoff = phase_a_steps < 1 or global_step < phase_a_steps
                phase_a_boundary = global_step == phase_a_steps
                if before_handoff or (
                    phase_a_boundary and not self.allow_phase_a_handoff
                ):
                    raise ValueError(
                        "multi-rate Plain requires a LIC checkpoint trained "
                        "past the Phase A boundary, unless "
                        "allow_phase_a_handoff=True explicitly selects the "
                        "completed latent warmup"
                    )
            elif checkpoint_stage == "multirate_reconstruction":
                if (
                    int(checkpoint.get("schema_version", 0)) != 3
                    or checkpoint.get("variable_rate_mechanism")
                    != "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
                ):
                    raise ValueError(
                        "multi-rate Plain requires a schema-version 3 P0 checkpoint "
                        "derived from the complete q_e/q_d and w_e/w_d LIC"
                    )
            elif (
                int(checkpoint.get("schema_version", 0)) != SELF_FLOW_SCHEMA_VERSION
                or checkpoint.get("self_flow_mechanism") != SELF_FLOW_MECHANISM
            ):
                raise ValueError(
                    "dual-rate Self-Flow requires its schema-version 4 checkpoint"
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
                "[Pure SANA] Loaded variable-rate LIC handoff; keeping fresh "
                "DiT/VAE LoRA, text projection, and decoder relay initialization"
            )
            return

        required = {
            "state_dict_codec",
            "state_dict_vae",
            "state_dict_DiT",
            "state_dict_text_proj",
        }
        if checkpoint_stage == SELF_FLOW_STAGE:
            if not self.self_flow_enabled:
                raise ValueError(
                    "Self-Flow checkpoint requires self_flow_params.enabled=true"
                )
            required.add("state_dict_quality_conditioner")
            if self.representation_projector is not None:
                required.add("state_dict_representation_projector")
        missing = sorted(required.difference(state))
        if missing:
            raise KeyError(
                "Checkpoint is not a fixed-text Pure SANA/all-off checkpoint; "
                f"missing entries: {missing}"
            )
        has_decoder_relay = "state_dict_decoder_relay" in state
        if self.decoder_relay is not None and not has_decoder_relay:
            raise KeyError(
                "checkpoint lacks state_dict_decoder_relay required by the config"
            )
        if self.decoder_relay is None and has_decoder_relay:
            raise ValueError(
                "checkpoint contains decoder relay weights but the config disables it"
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
            strict=checkpoint_stage == "multirate_reconstruction"
            or checkpoint_stage == SELF_FLOW_STAGE,
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
        if checkpoint_stage == SELF_FLOW_STAGE:
            self.quality_conditioner.load_state_dict(
                state["state_dict_quality_conditioner"], strict=True
            )
            has_projector_state = (
                "state_dict_representation_projector" in state
            )
            if self._init_for_training and (
                (self.representation_projector is None) != (not has_projector_state)
            ):
                raise ValueError(
                    "Self-Flow projector state does not match the active ablation"
                )
            if self.representation_projector is not None:
                self.representation_projector.load_state_dict(
                    state["state_dict_representation_projector"], strict=True
                )
        if self.decoder_relay is not None:
            self.decoder_relay.load_state_dict(
                state["state_dict_decoder_relay"],
                strict=True,
            )

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
            if self.self_flow_enabled:
                if self.decoder_relay is not None:
                    raise RuntimeError(
                        "Self-Flow requires decoder_relay_params.mode=none"
                    )
                if self.quality_conditioner is None or not all(
                    parameter.requires_grad
                    for parameter in self.quality_conditioner.parameters()
                ):
                    raise RuntimeError(
                        "Self-Flow requires a trainable token quality conditioner"
                    )
                representation_enabled = bool(
                    self.self_flow_params.get("representation_enabled", True)
                )
                if representation_enabled != (
                    self.representation_projector is not None
                ):
                    raise RuntimeError(
                        "representation projector does not match the ablation config"
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
        if self.decoder_relay is not None:
            parameters.extend(
                parameter
                for parameter in self.decoder_relay.parameters()
                if parameter.requires_grad
            )
        if self.quality_conditioner is not None:
            parameters.extend(
                parameter
                for parameter in self.quality_conditioner.parameters()
                if parameter.requires_grad
            )
        if self.representation_projector is not None:
            parameters.extend(
                parameter
                for parameter in self.representation_projector.parameters()
                if parameter.requires_grad
            )
        return parameters

    def get_self_flow_optimizer_groups(self, optimizer_config):
        if not self.self_flow_enabled:
            raise RuntimeError("Self-Flow optimizer groups requested while disabled")
        groups = [
            {
                "name": "dit_text",
                "params": [
                    *[
                        parameter
                        for name, parameter in self.DiT.named_parameters()
                        if "lora" in name and parameter.requires_grad
                    ],
                    *[
                        parameter
                        for parameter in self.text_proj.parameters()
                        if parameter.requires_grad
                    ],
                ],
                "lr": float(optimizer_config["dit_lr"]),
            },
            {
                "name": "vae_decoder",
                "params": [
                    parameter
                    for name, parameter in self.vae.named_parameters()
                    if "lora" in name and parameter.requires_grad
                ],
                "lr": float(optimizer_config["vae_lr"]),
            },
            {
                "name": "self_flow",
                "params": [
                    *[
                        parameter
                        for parameter in self.quality_conditioner.parameters()
                        if parameter.requires_grad
                    ],
                    *([] if self.representation_projector is None else [
                        parameter
                        for parameter in self.representation_projector.parameters()
                        if parameter.requires_grad
                    ]),
                ],
                "lr": float(optimizer_config["condition_lr"]),
            },
        ]
        groups = [group for group in groups if group["params"]]
        grouped_ids = [
            id(parameter)
            for group in groups
            for parameter in group["params"]
        ]
        trainable_ids = [id(parameter) for parameter in self.get_trainable_params()]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("Self-Flow optimizer groups contain duplicates")
        if set(grouped_ids) != set(trainable_ids):
            raise RuntimeError(
                "Self-Flow optimizer groups do not cover all trainable parameters"
            )
        return groups

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
        if self.quality_conditioner is not None:
            state_dicts["state_dict_quality_conditioner"] = (
                self.quality_conditioner.state_dict()
            )
        if self.representation_projector is not None:
            state_dicts["state_dict_representation_projector"] = (
                self.representation_projector.state_dict()
            )
        if self.decoder_relay is not None:
            decoder_relay = getattr(
                self.decoder_relay,
                "module",
                self.decoder_relay,
            )
            state_dicts["state_dict_decoder_relay"] = (
                decoder_relay.state_dict()
            )
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

    def _get_vae_decoder(self):
        cached = self.__dict__.get("_decoder_relay_target")
        if cached is not None:
            return cached
        decoders = [
            module
            for name, module in self.vae.named_modules()
            if name == "decoder" or name.endswith(".decoder")
        ]
        if len(decoders) != 1:
            raise RuntimeError(
                "decoder relay requires exactly one VAE decoder module"
            )
        self.__dict__["_decoder_relay_target"] = decoders[0]
        return decoders[0]

    def _vae_decode_with_relay(self, latent, decoder_bridge):
        if self.decoder_relay is None:
            return self.vae.decode(latent, return_dict=False)[0]

        decoder = self._get_vae_decoder()
        handles = []
        for block_index in self.decoder_relay.stage_block_indices:
            block = decoder.up_blocks[block_index]

            def relay_hook(module, inputs, output, index=block_index):
                return self.decoder_relay(index, output, decoder_bridge)

            handles.append(block.register_forward_hook(relay_hook))
        try:
            return self.vae.decode(latent, return_dict=False)[0]
        finally:
            for handle in handles:
                handle.remove()

    def _normalize_quality_index(self, quality_index, batch_size, device):
        if quality_index is None:
            raise ValueError(
                "quality_index is required when Self-Flow q conditioning is active"
            )
        if isinstance(quality_index, int):
            quality_index = torch.full(
                (batch_size,), quality_index, dtype=torch.long, device=device
            )
        elif isinstance(quality_index, torch.Tensor):
            quality_index = quality_index.reshape(-1).to(
                device=device, dtype=torch.long
            )
            if quality_index.numel() == 1:
                quality_index = quality_index.expand(batch_size)
        else:
            raise TypeError("quality_index must be an int or integer tensor")
        if quality_index.shape != (batch_size,):
            raise ValueError("quality_index must be scalar or have shape [batch]")
        return quality_index

    def self_flow_timestep(self, batch_size, device):
        timestep = self.sched.base_scheduler.timesteps.expand(batch_size)
        return (timestep * self.DiT.config.timestep_scale).to(device)

    def _self_flow_sana_prediction(
        self,
        latent,
        quality_map,
        conditioning_scale,
        capture_layer=None,
    ):
        if not self.self_flow_enabled:
            raise RuntimeError("Self-Flow SANA path is not enabled")
        condition = self._fixed_text_condition(latent.shape[0], latent.device)
        return quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT),
            latent,
            condition,
            self.self_flow_timestep(latent.shape[0], latent.device),
            quality_map,
            self.quality_conditioner,
            conditioning_scale=conditioning_scale,
            capture_layer=capture_layer,
        )

    def _decode_with_sana(
        self,
        latent,
        decoder_bridge=None,
        quality_index=None,
        conditioning_scale=1.0,
    ):
        condition = self._fixed_text_condition(
            latent.shape[0],
            latent.device,
        )
        timestep = self.self_flow_timestep(latent.shape[0], latent.device)
        if self.self_flow_enabled:
            quality_index = self._normalize_quality_index(
                quality_index, latent.shape[0], latent.device
            )
            quality_map = quality_index[:, None, None].expand(
                latent.shape[0], latent.shape[2], latent.shape[3]
            )
            model_prediction, _ = self._self_flow_sana_prediction(
                latent, quality_map, conditioning_scale
            )
        else:
            model_prediction = self.DiT(
                latent,
                encoder_hidden_states=condition,
                timestep=timestep,
                return_dict=False,
            )[0]
        denoised = self.sched.step(model_prediction, latent)
        return self._vae_decode_with_relay(
            denoised / self.vae.config.scaling_factor,
            decoder_bridge,
        ).clamp(-1, 1)

    def _decode_self_flow_student(
        self, latent, quality_map, conditioning_scale
    ):
        model_prediction, student_feature = self._self_flow_sana_prediction(
            latent,
            quality_map,
            conditioning_scale,
            capture_layer=self.self_flow_student_layer,
        )
        denoised = self.sched.step(model_prediction, latent)
        reconstruction = self._vae_decode_with_relay(
            denoised / self.vae.config.scaling_factor, None
        ).clamp(-1, 1)
        self.__dict__["_self_flow_probe_feature"] = student_feature
        projected_feature = (
            self.representation_projector(student_feature)
            if self.representation_projector is not None
            else None
        )
        return reconstruction, projected_feature

    def build_self_flow_teacher(self):
        if not self.self_flow_enabled:
            raise RuntimeError("cannot build a Self-Flow teacher when disabled")
        return SelfFlowTeacher(
            self.DiT,
            self.text_proj,
            self.quality_conditioner,
            self.self_flow_teacher_layer,
        )

    def forward_self_flow(
        self,
        image,
        quality_a,
        quality_b,
        mask_ratio,
        mask_seed,
        conditioning_scale,
        student_input="mixed",
        teacher_quality="higher",
        use_second_view=True,
    ):
        if not self.self_flow_enabled:
            raise RuntimeError("forward_self_flow requires self_flow_params.enabled")
        if self.decoder_relay is not None:
            raise RuntimeError("Self-Flow does not define mixed decoder relay features")
        batch_size = image.shape[0]
        quality_a = self._normalize_quality_index(
            quality_a, batch_size, image.device
        )
        quality_b = self._normalize_quality_index(
            quality_b, batch_size, image.device
        )
        with torch.no_grad():
            clean_latent = self.encode_clean_latent(image)
            view_a = self.codec(clean_latent, quality_a)
            view_b = (
                self.codec(clean_latent, quality_b)
                if use_second_view
                else view_a
            )
        latent_a = view_a["mean"]
        latent_b = view_b["mean"]
        generator = torch.Generator(device=latent_a.device)
        generator.manual_seed(int(mask_seed))
        mask = torch.rand(
            (batch_size, 1, latent_a.shape[2], latent_a.shape[3]),
            generator=generator,
            device=latent_a.device,
        ) < float(mask_ratio)
        if student_input == "uniform":
            mask = torch.zeros_like(mask)
        elif student_input != "mixed":
            raise ValueError("student_input must be mixed or uniform")
        mixed_latent, quality_map = mix_dual_rate_tokens(
            latent_a, latent_b, quality_a, quality_b, mask
        )
        if teacher_quality == "higher":
            teacher_quality_index = torch.maximum(quality_a, quality_b)
            choose_b = quality_b > quality_a
            if torch.any(choose_b).item() and not use_second_view:
                raise ValueError("higher-quality teacher requires the second codec view")
        elif teacher_quality == "student":
            teacher_quality_index = quality_a
            choose_b = torch.zeros_like(quality_a, dtype=torch.bool)
        else:
            raise ValueError("teacher_quality must be higher or student")
        teacher_latent = torch.where(
            choose_b[:, None, None, None], latent_b, latent_a
        )
        reconstruction, student_feature = self._decode_self_flow_student(
            mixed_latent, quality_map, conditioning_scale
        )
        return {
            "reconstruction": reconstruction,
            "student_feature": student_feature,
            "teacher_latent": teacher_latent.detach(),
            "teacher_quality": teacher_quality_index,
            "likelihoods_a": view_a["likelihoods"],
            "likelihoods_b": view_b["likelihoods"],
            "mask": mask,
            "quality_map": quality_map,
        }

    def encode_clean_latent(self, image):
        latent = self.vae.encode(image).latent * self.vae.config.scaling_factor
        return latent

    def decode_with_model(
        self, latent, decoder_bridge=None, quality_index=None
    ):
        return self._decode_with_sana(
            latent, decoder_bridge, quality_index=quality_index
        )

    def forward(self, image, quality_index=None, self_flow_batch=None):
        if self_flow_batch is not None:
            if quality_index is not None:
                raise ValueError(
                    "quality_index and self_flow_batch are mutually exclusive"
                )
            auxiliary_output = self.forward_self_flow(
                image, **self_flow_batch
            )
            output_image = auxiliary_output["reconstruction"]
        else:
            latent = self.encode_clean_latent(image)
            codec_output = self.codec(latent, quality_index)
            compressed_latent = codec_output["mean"]
            decoder_bridge = (
                codec_output["decoder_bridge"]
                if self.decoder_relay_mode == "bitstream"
                else None
            )
            output_image = self.decode_with_model(
                compressed_latent,
                decoder_bridge,
                quality_index=quality_index,
            )
            auxiliary_output = codec_output["likelihoods"]
        return output_image, auxiliary_output

    def compress(self, image, quality_index=None):
        latent = self.encode_clean_latent(image)
        return self.codec.compress(latent, quality_index)

    def decompress(self, packet_or_strings, shape=None, quality_index=None):
        decode_quality = getattr(packet_or_strings, "quality_index", quality_index)
        if self.decoder_relay_mode == "bitstream":
            latent, decoder_bridge = self.codec.decompress(
                packet_or_strings,
                shape=shape,
                quality_index=quality_index,
                return_bridge=True,
            )
        else:
            latent = self.codec.decompress(
                packet_or_strings,
                shape=shape,
                quality_index=quality_index,
            )
            decoder_bridge = None
        return self.decode_with_model(
            latent, decoder_bridge, quality_index=decode_quality
        )
