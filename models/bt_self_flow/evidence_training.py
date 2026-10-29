"""Exact parameter-specific objectives with explicit, averaged data parallelism.

DDP's reducer does not support autograd.grad. This path deliberately uses no
DDP hooks: it synchronizes initial state, accumulates each role's gradients,
then averages main gradients once per optimizer update before native clipping.
"""

import random

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .data import counter_seed
from .losses import estimated_bpp, reconstruction_objective
from .model import BTForwardResult
from .packet import RateControl


def higher_control(config, low, step, rank):
    settings = config["evidence_self_flow"]
    values = config["quality"]["values"]
    candidates = [q for q in values if q > low.value + 1e-5]
    if settings.get("teacher_evidence", "high") == "low" or not candidates:
        return low
    rng = random.Random(counter_seed(config["train"]["seed"], step,
                                      settings.get("teacher_stream", 9001) + rank))
    return RateControl(rng.choice(candidates))


def sf_coefficient(settings, step):
    if not settings.get("sf_enabled", True):
        return 0.
    return settings.get("coefficient", .05) * max(0., min(1.,
        (step-settings.get("warmup", 1000))/settings.get("ramp", 2000)))


def all_finite(value, world):
    finite = torch.isfinite(value.detach()).all().to(torch.int32)
    if world > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


@torch.no_grad()
def synchronize_initial_state(model, world):
    if world > 1:
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def average_gradients(parameters, world, bucket_bytes=32 * 1024**2):
    """Preserve globally unused gradients as None (AdamW must not update them)."""
    if world <= 1:
        return
    parameters = list(parameters)
    present = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                           dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    used = present.tolist()
    buckets, bucket, size, signature = [], [], 0, None
    for parameter, active in zip(parameters, used):
        if not active:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grad = parameter.grad
        key = (grad.device, grad.dtype)
        if bucket and (size + grad.numel()*grad.element_size() > bucket_bytes or key != signature):
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append(grad)
        size += grad.numel()*grad.element_size()
        signature = key
    if bucket:
        buckets.append(bucket)
    for gradients in buckets:
        flat = torch.cat([g.reshape(-1) for g in gradients])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat[offset:offset+gradient.numel()].view_as(gradient))
            offset += gradient.numel()


def evidence_backward(model, image, control, image_loss, config, step, rank, world, scaler,
                      codec_parameters, backend_parameters, amp_dtype, full_precision=False):
    """Teacher backward is freed before student generation; codec graph is native.

    Before global clipping:
      codec:   d(D_s + .1*base + lambda_R*R_low)
      backend: d((1-beta)*D_s + beta*D_t + gamma*SF)
    Only H20's use as the SF target is detached; teacher reconstruction trains
    the same current backend. No teacher/high-rate loss reaches the codec.
    """
    settings = config["evidence_self_flow"]
    high_control = higher_control(config, control, step, rank)
    privileged = high_control.value > control.value + 1e-5
    beta = settings.get("beta", .25) if privileged else 0.
    capture = privileged and settings.get("sf_enabled", True)
    gamma = sf_coefficient(settings, step) if privileged else 0.
    divisor = config["train"]["accumulation_steps"]
    autocast = lambda: torch.autocast(device_type=image.device.type, dtype=amp_dtype,
                                     enabled=image.is_cuda and not full_precision, cache_enabled=False)
    with autocast():
        clean = model.encode_clean_latent(image)
        low_source = model.codec.analyze(clean, control)
        low = model.codec.quantize_analysis(low_source, control)
    teacher_distortion = target = high = None
    teacher_metrics = {}
    if privileged:
        # Privileged quantization noise / stochastic layers must not consume the
        # native student's RNG stream. This scope also covers checkpoint replay.
        devices = [image.device.index] if image.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            teacher_seed = counter_seed(config["train"]["seed"], step, 9101+rank)
            torch.random.default_generator.manual_seed(teacher_seed)
            if image.is_cuda:
                torch.cuda.manual_seed(teacher_seed)
            with autocast():
                with torch.no_grad():
                    # Analysis is quality-conditioned: never quantize low_source at q_high.
                    high_source = model.codec.analyze(clean, high_control)
                    high = model.codec.quantize_analysis(high_source, high_control)
                _, teacher_image, feature = model.repair(
                    low["mean"].detach(), precision=low["delta"].detach(),
                    evidence=(high["mean"], high["delta"]),
                    capture_layer=model.teacher_layer if capture else None)
                teacher_distortion, teacher_metrics = image_loss(teacher_image, image)
            target = feature.detach() if capture else None
            if not all_finite(teacher_distortion, world):
                # All ranks abandon together; retain low-state diagnostics only.
                diagnostic = BTForwardResult(clean, low["mean"], low["mean"], None,
                    low["likelihoods"], 0, low["allocation_codes"], source=low_source)
                return diagnostic, teacher_distortion, {"teacher_distortion": teacher_distortion}
            if beta:
                torch.autograd.backward(scaler.scale(beta * teacher_distortion / divisor),
                                        inputs=backend_parameters)
            teacher_distortion = teacher_distortion.detach()
            del teacher_image, feature
    with autocast():
        result = model.result_from_view(low_source, low, control, capture=capture)
        native_loss, metrics = reconstruction_objective(
            result, image, control, model.distortion_scale.s_data, image_loss,
            objective=config.get("objective"), quality=config.get("quality"))
        distortion = sum(metrics[key] for key in ("weighted_mse", "weighted_lpips", "weighted_dists"))
        sf = (1-F.cosine_similarity(result.feature.float(), target.float(), dim=-1, eps=1e-6)).mean() \
            if capture else native_loss.new_zeros(())
        backend_loss = (1-beta)*distortion + gamma*sf
        checked_loss = native_loss + backend_loss
    if all_finite(checked_loss, world):
        gradients = torch.autograd.grad(scaler.scale(native_loss/divisor), codec_parameters,
                                        retain_graph=True, allow_unused=True)
        for parameter, gradient in zip(codec_parameters, gradients):
            if gradient is not None:
                if parameter.grad is None:
                    parameter.grad = gradient.detach()
                else:
                    parameter.grad.add_(gradient.detach())
        torch.autograd.backward(scaler.scale(backend_loss/divisor), inputs=backend_parameters)
    if teacher_distortion is None:
        teacher_distortion = distortion.detach()
        teacher_metrics = {name: metrics[name].detach() for name in ("mse", "lpips", "dists", "psnr")}
    with torch.no_grad():
        teacher_rate = estimated_bpp(type("View", (), dict(likelihoods=high["likelihoods"],
                                     side_bits=result.side_bits))(), image) if privileged else metrics["rate_est"]
        ordering = (high["delta"] < low["delta"]).float().mean() if privileged else sf.new_zeros(())
    metrics.update(student_distortion=distortion.detach(), teacher_distortion=teacher_distortion,
                   backend_reconstruction=(1-beta)*distortion.detach()+beta*teacher_distortion,
                   native_objective=native_loss.detach(),
                   backend_objective=backend_loss.detach()+beta*teacher_distortion,
                   sf=sf.detach(), sf_weight=sf.new_tensor(gamma), teacher_q=sf.new_tensor(high_control.value),
                   teacher_rate_est=teacher_rate, high_step_smaller_fraction=ordering,
                   privileged_fraction=sf.new_tensor(float(privileged)),
                   **{f"teacher_{key}": value.detach() for key, value in teacher_metrics.items()})
    # Metrics must not retain either of the two full reconstruction graphs.
    reported = native_loss.detach()-beta*distortion.detach()+beta*teacher_distortion+gamma*sf.detach()
    return result, reported, {key: value.detach() for key, value in metrics.items()}
