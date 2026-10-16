"""Bounded full-precision replay of an accumulated optimizer update."""

from contextlib import contextmanager

import torch

from models.sana_self_flow import unwrap_sana_transformer


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
