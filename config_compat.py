import os
from copy import deepcopy

from decode_modes import validate_decode_mode


_REMOVED_FEATURE_FLAGS = (
    "use_aux_encoder",
    "use_aux_decoder",
    "use_variance_guide",
    "use_self_dist",
)
FROZEN_MEANFLOW_MODES = ("pure", "meanflow")


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

    # The old auxiliary optimizer and the current entropy optimizer own the
    # same entropy-bottleneck quantile parameters.
    if "entropy_lr" not in optimizer and "aux_lr" in optimizer:
        optimizer["entropy_lr"] = optimizer["aux_lr"]

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
        enabled = mode != "disabled"
        meanflow_params["enabled"] = enabled
        meanflow_params["mode"] = mode
        if enabled:
            if mode != "meanflow":
                raise ValueError(
                    "Pure SANA supports only MeanFlow 5A mode='meanflow'"
                )
            conditioning_mode = meanflow_params.setdefault(
                "conditioning_mode",
                "adaln",
            )
            if conditioning_mode != "adaln":
                raise ValueError(
                    "Pure SANA MeanFlow supports only "
                    "conditioning_mode='adaln'"
                )
            if params["decode_mode"] != "sana":
                raise ValueError(
                    "MeanFlow AdaLN requires decode_mode='sana'"
                )
            weight = float(meanflow_params.get("weight", 0.05))
            if weight <= 0:
                raise ValueError("MeanFlow weight must be positive")
            meanflow_params["weight"] = weight
            meanflow_params.setdefault("equal_time_ratio", 0.5)
            max_steps = train.get("max_steps")
            disc_start = loss_params.get("disc_start")
            if (
                max_steps is not None
                and disc_start is not None
                and disc_start <= max_steps
            ):
                raise ValueError(
                    "MeanFlow training must be no-GAN: set disc_start "
                    "greater than train.max_steps"
                )
        params["meanflow_params"] = meanflow_params

    return config


def configure_frozen_decode_ablation(config, mode, init_checkpoint):
    """Configure a fixed-bitstream SANA-vs-g_s warm-start comparison."""
    mode = validate_decode_mode(mode)
    if not init_checkpoint:
        raise ValueError("frozen decode ablation requires --init_checkpoint")

    init_checkpoint = os.path.abspath(os.path.expanduser(init_checkpoint))
    if not os.path.isfile(init_checkpoint):
        raise FileNotFoundError(
            f"Pure SANA initialization checkpoint not found: {init_checkpoint}"
        )

    params = config["model"]["params"]
    configured_mode = params.get("decode_mode", "sana")
    if configured_mode != mode:
        raise ValueError(
            f"config decode_mode={configured_mode!r} does not match "
            f"--frozen_decode_ablation={mode!r}"
        )
    params.update(
        {
            "codec_path": init_checkpoint,
            "freeze_codec": True,
            "use_merge": False,
            "allow_missing_gs_fusion_for_init": mode == "gs_residual",
        }
    )
    config.setdefault("experiment", {})["frozen_decode_ablation"] = {
        "mode": mode,
        "init_checkpoint": init_checkpoint,
    }
    return config


def configure_frozen_meanflow_ablation(config, mode, init_checkpoint):
    """Configure a fixed-bitstream Pure-vs-AdaLN-MeanFlow comparison."""
    if mode not in FROZEN_MEANFLOW_MODES:
        raise ValueError(
            f"mode must be one of {FROZEN_MEANFLOW_MODES}, got {mode!r}"
        )
    if not init_checkpoint:
        raise ValueError(
            "frozen MeanFlow ablation requires --init_checkpoint"
        )

    init_checkpoint = os.path.abspath(os.path.expanduser(init_checkpoint))
    if not os.path.isfile(init_checkpoint):
        raise FileNotFoundError(
            f"Pure SANA initialization checkpoint not found: {init_checkpoint}"
        )

    params = config["model"]["params"]
    if params.get("decode_mode", "sana") != "sana":
        raise ValueError(
            "Frozen MeanFlow comparison requires decode_mode='sana'"
        )
    params.update(
        {
            "codec_path": init_checkpoint,
            "freeze_codec": True,
            "use_merge": False,
            "require_pure_init": True,
        }
    )
    meanflow_params = params.setdefault("meanflow_params", {})
    meanflow_params.update(
        {
            "enabled": mode == "meanflow",
            "mode": "meanflow" if mode == "meanflow" else "disabled",
            "conditioning_mode": "adaln",
            "equal_time_ratio": 0.5,
        }
    )

    max_steps = config["train"]["max_steps"]
    disc_start = params["loss_params"].get("disc_start", 0)
    if disc_start <= max_steps:
        raise ValueError(
            "Frozen Pure-vs-MeanFlow comparison must be no-GAN: "
            "set disc_start greater than train.max_steps"
        )

    config.setdefault("experiment", {})["frozen_meanflow_ablation"] = {
        "mode": mode,
        "init_checkpoint": init_checkpoint,
    }
    config["train"]["resume"] = False
    config["train"].pop("resume_checkpoint", None)
    exp_name = config["train"].get("exp_name") or "pure_sana"
    config["train"]["exp_name"] = f"{exp_name}_frozen_{mode}"
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
