"""Bounded full-precision replay of an accumulated optimizer update."""

from contextlib import contextmanager
import json
from pathlib import Path

import torch

from models.sana_self_flow import unwrap_sana_transformer


def can_retry_fp32(device, world, already_retrying, scaler_enabled=False):
    """BF16 needs the same bounded replay as scaled FP16."""
    return world == 1 and not already_retrying and (device.type == "cuda" or scaler_enabled)


def resume_requires_fp32(checkpoint, output):
    """Recognize a legacy norm-only retry now handled by stable clipping."""
    if not checkpoint or not checkpoint.get("full_precision_retry", False):
        return False
    if checkpoint.get("scaler"):
        return True
    diagnostic = Path(output)/f"nonfinite_{checkpoint['step']:07d}_{checkpoint['attempts']:07d}.json"
    try:
        report = json.loads(diagnostic.read_text())
    except (OSError, ValueError):
        return True
    ranks = report.get("ranks", [])
    norm_only = report.get("kind") == "gradient" and len(ranks) == 1 and all(
        item.get("step") == checkpoint["step"] and item.get("rank") == 0
        and item.get("backend") == checkpoint["backend"]
        and item.get("model_state_finite") is True
        and item.get("nonfinite_gradient_count") == 0
        and item.get("nonfinite_gradient_parameters") == [] for item in ranks)
    return not norm_only


def nonfinite_gradient_diagnostics(model):
    bad = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            bad.append(name)
    return {"nonfinite_gradient_count": len(bad), "nonfinite_gradient_parameters": bad[:32]}


@torch.no_grad()
def clip_grad_norm_safe_(parameters, max_norm, error_if_nonfinite=True):
    """Keep the usual clipping path, with FP64 reduction on norm overflow."""
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        norms = [torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients]
        total = torch.linalg.vector_norm(torch.stack([value.to(norms[0].device) for value in norms]))
        if not torch.isfinite(total):
            if error_if_nonfinite:
                raise
            # GradScaler recorded the bad gradients during unscale and will skip.
            return total
        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total)
        return total


@contextmanager
def full_precision_recompute(model, enabled=False):
    """Recompute FP32 activations in blocks so a retry fits the same GPU.

    The caller must include backward in this scope. Parameters, update rules,
    and input batches stay unchanged; temporary checkpoint flags are restored
    even when the forward or backward raises.
    """
    sana = unwrap_sana_transformer(model.DiT) if enabled and hasattr(model, "DiT") else None
    checkpoint_blocks = getattr(model, "checkpoint_blocks", None)
    checkpointing = getattr(sana, "gradient_checkpointing", None)
    try:
        if enabled:
            if checkpoint_blocks is not None:
                model.checkpoint_blocks = True
            if sana is not None:
                sana.enable_gradient_checkpointing()
        yield
    finally:
        if checkpoint_blocks is not None:
            model.checkpoint_blocks = checkpoint_blocks
        if checkpointing is not None:
            sana.gradient_checkpointing = checkpointing


def nonfinite_diagnostics(result, metrics):
    """Small CPU report, only synchronized after a nonfinite forward."""
    tensors = {name: getattr(result, name) for name in ("clean", "coarse", "latent", "image")}
    return {
        "metrics": {name: float(value.detach()) for name, value in metrics.items()},
        "tensors": {name: {"finite": bool(torch.isfinite(value).all()),
                            "max_abs": float(value.detach().abs().max())}
                    for name, value in tensors.items() if value is not None},
    }


def finite_state(model):
    checks = [torch.isfinite(value).all() for values in model.export_state().values()
              for value in values.values() if value.is_floating_point()]
    return not checks or bool(torch.stack(checks).all())
