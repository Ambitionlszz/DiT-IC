from copy import deepcopy


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

    # Legacy training configs called the real-rate weight `lrd`/`kl_weight`.
    rate_weight = train.get("rate_weight", train.get("lrd"))
    if rate_weight is None:
        rate_weight = loss_params.get("rate_weight", loss_params.get("kl_weight", 1.0))
    train.setdefault("rate_weight", rate_weight)
    loss_params.setdefault("rate_weight", rate_weight)

    # The old auxiliary optimizer and the current entropy optimizer own the
    # same entropy-bottleneck quantile parameters.
    if "entropy_lr" not in optimizer and "aux_lr" in optimizer:
        optimizer["entropy_lr"] = optimizer["aux_lr"]

    # Allow freeze controls to live in either the train or model section.
    if "freeze_codec" in train and "freeze_codec" not in params:
        params["freeze_codec"] = bool(train["freeze_codec"])
    train.setdefault("resume", False)

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
