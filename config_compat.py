from copy import deepcopy

from decode_modes import validate_decode_mode


_REMOVED_FEATURE_FLAGS = (
    "use_aux_encoder",
    "use_aux_decoder",
    "use_variance_guide",
    "use_self_dist",
)
def normalize_pure_sana_config(config):
    """Map legacy main-branch YAML fields onto the Pure SANA interface."""
    config = deepcopy(config)
    model = config.setdefault("model", {})
    params = model.setdefault("params", {})
    train = config.setdefault("train", {})
    optimizer = config.setdefault("optimizer", {})
    loss_params = params.setdefault("loss_params", {})

    enabled_removed_features = [
        name for name in _REMOVED_FEATURE_FLAGS if params.get(name, False)
    ]
    if enabled_removed_features:
        names = ", ".join(enabled_removed_features)
        raise ValueError(
            "Pure SANA cannot enable removed DiT-IC features: " + names
        )
    if params.get("use_text_condition", True) is False:
        raise ValueError(
            "Pure SANA requires the fixed text condition; "
            "legacy latent-conditioned checkpoints are not compatible."
        )

    # Main treats train.lrd as the effective rate weight.  Resolve the legacy
    # and Pure-SANA names once so stale duplicate fields cannot silently select
    # a different objective.
    train_lrd = train.get("lrd")
    train_rate_weight = train.get("rate_weight")
    loss_rate_weight = loss_params.get("rate_weight")

    if (
        train_lrd is not None
        and train_rate_weight is not None
        and train_lrd != train_rate_weight
    ):
        raise ValueError(
            "Conflicting rate weights: "
            f"train.lrd={train_lrd!r} and "
            f"train.rate_weight={train_rate_weight!r}"
        )

    train_weight = (
        train_lrd if train_lrd is not None else train_rate_weight
    )
    if (
        train_weight is not None
        and loss_rate_weight is not None
        and train_weight != loss_rate_weight
    ):
        raise ValueError(
            "Conflicting rate weights: "
            f"train={train_weight!r} and "
            f"model.params.loss_params.rate_weight={loss_rate_weight!r}"
        )

    rate_weight = train_weight
    if rate_weight is None:
        rate_weight = loss_rate_weight
    if rate_weight is None:
        rate_weight = loss_params.get("kl_weight", 1.0)
    train["rate_weight"] = rate_weight
    loss_params["rate_weight"] = rate_weight

    # The official auxiliary optimizer and the local entropy optimizer own the
    # same entropy-bottleneck quantile parameters.
    aux_lr = optimizer.get("aux_lr")
    entropy_lr = optimizer.get("entropy_lr")
    if (
        aux_lr is not None
        and entropy_lr is not None
        and aux_lr != entropy_lr
    ):
        raise ValueError(
            "Conflicting entropy learning rates: "
            f"aux_lr={aux_lr!r} and entropy_lr={entropy_lr!r}"
        )
    if aux_lr is None and entropy_lr is not None:
        optimizer["aux_lr"] = entropy_lr
    if entropy_lr is None and aux_lr is not None:
        optimizer["entropy_lr"] = aux_lr

    # Allow freeze controls to live in either the train or model section.
    if "freeze_codec" in train and "freeze_codec" not in params:
        params["freeze_codec"] = bool(train["freeze_codec"])
    train.setdefault("resume", False)

    quality_control = config.get("quality_control") or {}
    if quality_control.get("enabled", False):
        num_levels = int(quality_control.get("num_levels", 0))
        if num_levels != 6:
            raise ValueError(
                "Variable-rate Pure SANA requires quality_control.num_levels=6"
            )
        codec_param = params.setdefault("codec_param", {})
        configured_variable_rate = codec_param.get("variable_rate")
        if configured_variable_rate is False:
            raise ValueError(
                "quality_control.enabled conflicts with codec_param.variable_rate=False"
            )
        codec_param["variable_rate"] = True
        codec_param["num_quality_levels"] = num_levels

        rate_weights = quality_control.get("rate_weights")
        if rate_weights is not None:
            values = [float(value) for value in rate_weights]
            if len(values) != num_levels:
                raise ValueError(
                    "quality_control.rate_weights must contain one value per quality"
                )
            if any(value <= 0 for value in values) or any(
                left <= right for left, right in zip(values, values[1:])
            ):
                raise ValueError(
                    "quality_control.rate_weights must be positive and strictly "
                    "decreasing from low-quality q0 to high-quality q5"
                )
            phase = train.get("e2e_phase")
            if phase not in ("nogan", "gan"):
                raise ValueError(
                    "end-to-end variable-rate training requires "
                    "train.e2e_phase=nogan or gan"
                )
            training_mode = train.get("training_mode", "scratch_e2e")
            freeze_codec = bool(params.get("freeze_codec", False))
            freeze_quality_endpoints = bool(
                train.get("freeze_quality_endpoints", False)
            )
            if training_mode == "scratch_e2e":
                if freeze_codec:
                    raise ValueError(
                        "scratch_e2e variable-rate training requires "
                        "freeze_codec=False"
                    )
                if freeze_quality_endpoints:
                    raise ValueError(
                        "scratch_e2e must not freeze the quality endpoints"
                    )
            elif training_mode == "phase1_warmup_e2e":
                if not freeze_quality_endpoints:
                    raise ValueError(
                        "phase1_warmup_e2e requires "
                        "train.freeze_quality_endpoints=True"
                    )
                if freeze_codec:
                    raise ValueError(
                        "phase1_warmup_e2e requires freeze_codec=False in "
                        "both NoGAN and GAN phases"
                    )
            else:
                raise ValueError(
                    "train.training_mode must be scratch_e2e or "
                    "phase1_warmup_e2e"
                )

        if params.get("freeze_codec", False):
            vae_params = params.get("vae_params") or {}
            train_vae_lora = vae_params.get(
                "train_vae_lora", vae_params.get("vae_lora", False)
            )
            if train_vae_lora and not vae_params.get("decoder_lora_only", False):
                raise ValueError(
                    "multi-rate Plain training requires "
                    "decoder-only VAE LoRA"
                )
            if not params.get("train_text_projection", True):
                raise ValueError(
                    "multi-rate Plain training requires train_text_projection=True"
                )
            if float(rate_weight) != 0.0:
                raise ValueError(
                    "multi-rate Plain checkpoint ranking requires train.rate_weight=0"
                )

    params["decode_mode"] = validate_decode_mode(
        params.get("decode_mode", "sana")
    )
    meanflow_params = dict(params.get("meanflow_params") or {})
    if meanflow_params:
        enabled_flag = meanflow_params.get("enabled")
        mode = meanflow_params.get(
            "mode",
            "meanflow" if enabled_flag else "disabled",
        )
        if enabled_flag is False:
            mode = "disabled"
        if mode != "disabled":
            raise ValueError(
                "This baseline branch does not implement MeanFlow; "
                "use the meanflow branch for that checkpoint or config."
            )
        params.pop("meanflow_params", None)

    return config


def prepare_inference_config(
    config,
    codec_path=None,
    use_ema=None,
    use_merge=None,
):
    config = normalize_pure_sana_config(config)
    params = config["model"]["params"]
    if codec_path is not None:
        params["codec_path"] = codec_path
    if not params.get("codec_path"):
        raise ValueError(
            "codec_path must be provided either in the YAML or via --codec_path"
        )
    if use_ema is not None:
        params["use_ema"] = use_ema
    if use_merge is not None:
        params["use_merge"] = use_merge
    params["training"] = False
    return config
