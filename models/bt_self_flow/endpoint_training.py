"""Endpoint SF gradient routing; manual distributed averaging stays shared."""

import torch
import torch.distributed as dist

from .endpoint_self_flow import (TARGETS, adjacent_quality, endpoint_sf_coefficient, sf_parameters,
                                 endpoint_spatial_diagnostics, endpoint_spatial_l1)
from .losses import reconstruction_objective
from .spatial_training import average_gradients, synchronize_model


def endpoint_backward(model, teacher, image, control, image_loss, config, step, rank, world,
                      scaler, amp_dtype, full_precision=False, micro=0):
    """Backpropagate original RD plus weighted endpoint feature SF.

    All examples/ranks use the same quality per update. At the highest quality
    every target control is RD-only, including same-rate and clean attribution
    controls. The teacher and its current-AE target prefix preserve student RNG.
    """
    settings = config["endpoint_self_flow"]
    divisor = config["train"]["accumulation_steps"]
    coefficient = endpoint_sf_coefficient(settings)
    target_kind = settings.get("target", "adjacent_ema")
    if target_kind not in TARGETS:
        raise ValueError(f"unknown endpoint SF target: {target_kind}")
    enabled = bool(settings.get("enabled", False))
    high = adjacent_quality(config, control)
    active = enabled and bool(getattr(model, "sf_enabled", enabled)) and high is not None
    target_control = control if target_kind != "adjacent_ema" else high
    with torch.autocast(device_type=image.device.type, dtype=amp_dtype,
                        enabled=image.is_cuda and not full_precision, cache_enabled=False):
        result = model(image, rate_control=control, capture=active)
        main, metrics = reconstruction_objective(
            result, image, control, model.distortion_scale.s_data, image_loss,
            objective=config.get("objective"), quality=config.get("quality"))
    sf = main.new_zeros(())
    diagnostics = {}
    if active:
        if result.feature is None or result.feature.ndim != 4:
            raise ValueError("active endpoint SF requires captured BCHW AE decoder features")
        if teacher is None and target_kind != "clean_latent":
            raise ValueError("endpoint SF requires a frozen deployment EMA teacher")
        devices = [image.device.index] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad(), torch.autocast(
                device_type=image.device.type, dtype=amp_dtype,
                enabled=image.is_cuda and not full_precision, cache_enabled=False):
            latent = (result.clean.detach() if target_kind == "clean_latent"
                      else teacher(result.clean, target_control))
            target = model.endpoint_feature(latent)
        error, local_count = endpoint_spatial_l1(result.feature, target, settings)
        count = torch.tensor(local_count, device=image.device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(count)
        # The later manual gradient average divides by world. Scaling the
        # local sum here yields the mean over all global images/elements.
        sf = error * world / count.clamp_min(1)
        diagnostics = endpoint_spatial_diagnostics(result.feature, target, settings)
    weighted_sf = coefficient * sf
    loss = main + weighted_sf
    finite = torch.isfinite(loss.detach()).to(torch.int32)
    if world > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if bool(finite):
        allowed = sf_parameters(model) if weighted_sf.requires_grad else []
        additions = torch.autograd.grad(scaler.scale(weighted_sf / divisor), allowed,
                                        retain_graph=True, allow_unused=True) if allowed else []
        scaler.scale(main / divisor).backward()
        for parameter, gradient in zip(allowed, additions):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient.detach()
                else:
                    parameter.grad.add_(gradient.detach())
    metrics.update(sf=weighted_sf.detach(), sf_raw=sf.detach(),
                   sf_weight=main.new_tensor(coefficient if active else 0.),
                   sf_active=main.new_tensor(float(active)),
                   teacher_q=main.new_tensor(target_control.value if active else control.value),
                   **diagnostics)
    return result, loss, metrics
