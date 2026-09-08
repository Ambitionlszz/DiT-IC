import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


SELF_FLOW_CHECKPOINT_SCHEMA_VERSION = 5
SELF_FLOW_STAGE = "dual_rate_self_flow_reconstruction"
SELF_FLOW_MECHANISM = "iid_dual_rate_token_mix_ema_l8_l20_v2"


def validate_self_flow_config(train_config):
    params = train_config["model"]["params"]
    config = dict(params.get("self_flow_params") or {})
    if not config.get("enabled", False):
        return config
    quality = train_config.get("quality_control") or {}
    if not quality.get("enabled", False) or int(quality.get("num_levels", 0)) != 6:
        raise ValueError("Self-Flow requires the six-level variable-rate codec")
    if not params.get("freeze_codec", False):
        raise ValueError("Self-Flow requires a frozen codec")
    if config.get("sampling") != "iid_uniform":
        raise ValueError("Self-Flow sampling must be iid_uniform")
    if quality.get("sampling") != "iid_uniform_per_sample":
        raise ValueError(
            "Self-Flow quality control must use iid_uniform_per_sample"
        )
    if config.get("student_input") not in ("mixed", "uniform"):
        raise ValueError("student_input must be mixed or uniform")
    if config.get("teacher_quality") not in ("higher", "student"):
        raise ValueError("teacher_quality must be higher or student")
    teacher_enabled = bool(config.get("teacher_enabled", True))
    representation_enabled = bool(
        config.get("representation_enabled", True)
    )
    if representation_enabled and not teacher_enabled:
        raise ValueError("representation loss requires the EMA teacher")
    if float(train_config["train"]["rate_weight"]) != 0.0:
        raise ValueError("Self-Flow mixed views do not define a rate objective")
    loss_params = params["loss_params"]
    if (
        float(loss_params.get("disc_weight", 0.0)) != 0.0
        or int(loss_params["disc_start"]) <= int(train_config["train"]["max_steps"])
    ):
        raise ValueError("Self-Flow is a NoGAN post-training stage")
    warmup_steps = int(config.get("uniform_warmup_steps", 2000))
    ramp_steps = int(config.get("ramp_steps", 5000))
    if warmup_steps < 0 or ramp_steps < 1:
        raise ValueError("Self-Flow warmup must be non-negative with a positive ramp")
    mask_ratio = float(config.get("mask_ratio_target", 0.25))
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio_target must be in [0, 1]")
    if config.get("student_input") == "mixed" and not config.get(
        "use_second_view", True
    ):
        raise ValueError("mixed Student input requires the second codec view")
    if (
        teacher_enabled
        and config.get("teacher_quality") == "higher"
        and not config.get("use_second_view", True)
    ):
        raise ValueError("higher-rate Teacher requires the second codec view")
    required_lrs = ("dit_lr", "vae_lr", "condition_lr")
    missing_lrs = [name for name in required_lrs if name not in train_config["optimizer"]]
    if missing_lrs:
        raise ValueError(f"Self-Flow optimizer lacks learning rates: {missing_lrs}")
    return config


def sample_reflected_adjacent_pairs(
    batch_size,
    num_levels,
    generator,
    same_rate=False,
):
    """Sample q_a uniformly and q_b from a reflected +/-1 proposal."""
    if same_rate:
        shared_quality = torch.randint(
            0, num_levels, (), generator=generator, dtype=torch.long
        )
        quality_a = shared_quality.expand(batch_size).clone()
        return quality_a, quality_a.clone()
    quality_a = torch.randint(
        0, num_levels, (batch_size,), generator=generator, dtype=torch.long
    )
    direction = torch.randint(
        0, 2, (batch_size,), generator=generator, dtype=torch.long
    )
    direction = direction * 2 - 1
    quality_b = (quality_a + direction).clamp(0, num_levels - 1)
    return quality_a, quality_b


def sample_iid_uniform_pairs(
    batch_size,
    num_levels,
    generator,
    same_rate=False,
):
    """Sample both source qualities independently from the full rate range."""
    quality_a = torch.randint(
        0, num_levels, (batch_size,), generator=generator, dtype=torch.long
    )
    if same_rate:
        return quality_a, quality_a.clone()
    quality_b = torch.randint(
        0, num_levels, (batch_size,), generator=generator, dtype=torch.long
    )
    return quality_a, quality_b


def deterministic_self_flow_seed(global_seed, rank, train_step, micro_step):
    value = (
        int(global_seed)
        + 1_000_003 * int(rank)
        + 10_007 * int(train_step)
        + 97 * int(micro_step)
    )
    return value % (2**63 - 1)


def self_flow_schedule(train_step, config):
    warmup_steps = int(config.get("uniform_warmup_steps", 2000))
    ramp_steps = int(config.get("ramp_steps", 5000))
    if train_step < warmup_steps:
        progress = 0.0
    else:
        progress = min(1.0, (train_step - warmup_steps) / float(ramp_steps))
    mask_ratio = (
        float(config.get("mask_ratio_target", 0.25)) * progress
        if config.get("student_input") == "mixed"
        else 0.0
    )
    return {
        "ramp_progress": progress,
        "mask_ratio": mask_ratio,
        "conditioning_scale": (
            progress if config.get("conditioning_enabled", True) else 0.0
        ),
        "representation_scale": progress,
    }


def select_post_burnin_best(
    test_loss,
    best_loss,
    train_step,
    max_steps,
    selection_start_fraction=0.7,
):
    """Select checkpoints only after burn-in without poisoning best_loss early."""
    if train_step <= float(selection_start_fraction) * int(max_steps):
        return best_loss, False
    is_best = test_loss < best_loss
    return (test_loss if is_best else best_loss), is_best


def all_token_representation_loss(student_feature, teacher_feature):
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "Student and Teacher representation grids must match exactly"
        )
    similarity = F.cosine_similarity(
        student_feature.float(), teacher_feature.detach().float(), dim=-1
    )
    return (1.0 - similarity).mean()


def likelihood_bpp(likelihoods, image):
    num_pixels = image.shape[0] * image.shape[2] * image.shape[3]
    return sum(
        torch.log(value).sum() / (-math.log(2) * num_pixels)
        for value in likelihoods.values()
    )


def calibrate_representation_weight(
    base_loss,
    representation_loss,
    probe_parameters,
    target_weight,
    maximum_gradient_ratio,
    minimum_weight,
):
    base_gradients = torch.autograd.grad(
        base_loss,
        probe_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    representation_gradients = torch.autograd.grad(
        representation_loss,
        probe_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def squared_norm(gradients):
        terms = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
        if not terms:
            return torch.zeros((), device=base_loss.device)
        return torch.stack(terms).sum()

    base_squared = squared_norm(base_gradients)
    representation_squared = squared_norm(representation_gradients)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(base_squared, op=dist.ReduceOp.SUM)
        dist.all_reduce(representation_squared, op=dist.ReduceOp.SUM)
    base_norm = base_squared.sqrt()
    representation_norm = representation_squared.sqrt()
    if not torch.isfinite(base_norm).item() or not torch.isfinite(
        representation_norm
    ).item():
        raise FloatingPointError("non-finite gradient norm during gamma calibration")
    if representation_norm.item() <= 0:
        raise RuntimeError(
            "representation loss has zero probe gradient: "
            f"base_norm={base_norm.item():.9g}, "
            f"rep_norm={representation_norm.item():.9g}, "
            f"rep_loss={representation_loss.item():.9g}"
        )
    resolved = min(
        float(target_weight),
        float(maximum_gradient_ratio) * base_norm.item() / representation_norm.item(),
    )
    if resolved < float(minimum_weight):
        raise RuntimeError(
            f"calibrated representation weight {resolved:.6g} is below the "
            f"pre-registered minimum {float(minimum_weight):.6g}"
        )
    return resolved, base_norm.item(), representation_norm.item()
