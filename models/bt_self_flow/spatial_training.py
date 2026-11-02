"""Explicit main/SF gradient routing with one averaged reduction per update."""

import torch
import torch.distributed as dist

from .losses import reconstruction_objective
from .spatial_self_flow import active_samples, higher_quality, sf_parameters, spatial_response_error


@torch.no_grad()
def synchronize_model(model, world):
    if world > 1:
        for value in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(value, src=0)


@torch.no_grad()
def average_gradients(parameters, world, bucket_bytes=32 * 1024**2):
    """Keep globally unused gradients as None, including an all-bypassed P."""
    if world <= 1:
        return
    parameters = list(parameters)
    present = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                           dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    buckets, bucket, size, signature = [], [], 0, None
    for parameter, used in zip(parameters, present.tolist()):
        if not used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad
        key = gradient.device, gradient.dtype
        if bucket and (size + gradient.numel() * gradient.element_size() > bucket_bytes or key != signature):
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(gradient)
        size += gradient.numel() * gradient.element_size()
        signature = key
    if bucket:
        buckets.append(bucket)
    for gradients in buckets:
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
            offset += gradient.numel()


def spatial_backward(model, teacher, image, control, image_loss, config, step, rank, world,
                     scaler, amp_dtype, full_precision=False, micro=0):
    settings = config["spatial_self_flow"]
    divisor = config["train"]["accumulation_steps"]
    enabled = bool(settings.get("enabled", False))
    active = active_samples(settings, image.shape[0], config["train"]["seed"], step, rank, micro,
                            image.device) if model.spatial_projector is not None else None
    with torch.autocast(device_type=image.device.type, dtype=amp_dtype,
                        enabled=image.is_cuda and not full_precision, cache_enabled=False):
        result = model(image, rate_control=control, capture=enabled, projector_mask=active)
        main, metrics = reconstruction_objective(
            result, image, control, model.distortion_scale.s_data, image_loss,
            objective=config.get("objective"), quality=config.get("quality"))
    sf = main.new_zeros(())
    high = higher_quality(config, control, step, rank, micro)
    if enabled:
        # Every rank participates even if it has no active local sample.
        local_count = int(active.sum()) * result.feature.shape[1] * result.feature.shape[2]
        count = torch.tensor(local_count, device=image.device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(count)
        if local_count:
            if teacher is None:
                raise ValueError("enabled spatial SF requires a no-gradient EMA teacher")
            devices = [image.device.index] if image.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad(), torch.autocast(
                    device_type=image.device.type, dtype=amp_dtype,
                    enabled=image.is_cuda and not full_precision, cache_enabled=False):
                target = teacher(result.clean[active], high)
            error, _ = spatial_response_error(result.feature[active], target, settings.get("epsilon", 1e-6))
            # average_gradients divides by world; undo it here for a true
            # global active-element mean, rather than a mean of local means.
            sf = error * world / count.clamp_min(1)
    loss = main + sf
    finite = torch.isfinite(loss.detach()).to(torch.int32)
    if world > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if bool(finite):
        allowed = sf_parameters(model) if sf.requires_grad else []
        additions = torch.autograd.grad(scaler.scale(sf / divisor), allowed,
                                        retain_graph=True, allow_unused=True) if allowed else []
        scaler.scale(main / divisor).backward()
        for parameter, gradient in zip(allowed, additions):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient.detach()
                else:
                    parameter.grad.add_(gradient.detach())
    metrics.update(sf=sf.detach(), sf_weight=main.new_tensor(1. if enabled else 0.),
                   teacher_q=main.new_tensor(high.value),
                   projector_active=main.new_tensor(float(active.float().mean()) if active is not None else 0.))
    return result, loss, metrics
