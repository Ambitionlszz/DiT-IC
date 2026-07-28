from copy import deepcopy

from decode_modes import validate_decode_mode


_REMOVED_FEATURE_FLAGS = (
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
    use_aux_encoder = bool(params.get("use_aux_encoder", False))
    use_aux_decoder = bool(params.get("use_aux_decoder", False))
    if use_aux_encoder != use_aux_decoder:
        raise ValueError("This experiment requires AuxE and AuxD together")
    if use_aux_encoder and not params.get("elic_path"):
        raise ValueError("AuxE requires model.params.elic_path")
    if use_aux_encoder and not params.get("text_encoder_path"):
        raise ValueError("AuxED requires the fixed text_encoder_path")

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
