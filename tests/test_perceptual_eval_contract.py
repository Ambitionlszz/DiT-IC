"""Frozen losses must remain deterministic across parent train/eval switches."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.losses import LPIPSWithDiscriminator


class StatefulDistance(nn.Module):
    """Small stand-in exposing both dropout and mutable normalization state."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Dropout(.5),
        )

    def forward(self, reference, reconstruction):
        difference = self.features(reference) - self.features(reconstruction)
        return difference.square().mean((1, 2, 3), keepdim=True)


@pytest.fixture
def objective():
    # Exercise the production loss with no perceptual-weight downloads or GPU.
    with patch("models.losses.lpips.LPIPS", side_effect=lambda **kwargs: StatefulDistance()), \
         patch("models.losses.pyiqa.create_metric", side_effect=lambda *args, **kwargs: StatefulDistance()):
        return LPIPSWithDiscriminator(device="cpu", disc_start=0, disc_num_layers=1)


def test_parent_mode_switch_keeps_perceptual_eval_and_discriminator_train(objective):
    parent = nn.ModuleDict({"objective": objective})
    for mode in (False, True, False, True):
        assert parent.train(mode) is parent
        assert objective.training is mode
        assert all(module.training is mode for module in objective.discriminator.modules())
        assert all(not module.training for metric in (objective.perceptual_loss, objective.dists)
                   for module in metric.modules())
    assert objective.train() is objective
    assert objective.logvar.requires_grad
    assert all(parameter.requires_grad for parameter in objective.discriminator.parameters())
    assert all(not parameter.requires_grad for metric in (objective.perceptual_loss, objective.dists)
               for parameter in metric.parameters())


@pytest.mark.parametrize("pp_style", [False, True])
def test_training_loss_is_repeatable_and_keeps_image_gradients(objective, pp_style):
    objective.pp_style = pp_style
    objective.train()
    generator = torch.Generator().manual_seed(17)
    reference = torch.rand(2, 3, 8, 8, generator=generator)*2-1
    reconstruction = torch.rand(2, 3, 8, 8, generator=generator).requires_grad_()
    likelihoods = {"y": torch.full((2, 1, 1, 1), .5)}
    buffers = {name: value.clone() for name, value in objective.named_buffers()
               if name.startswith(("perceptual_loss.", "dists."))}

    first, first_metrics = objective(reference, reconstruction, likelihoods, -1, 0)
    second, second_metrics = objective(reference, reconstruction, likelihoods, -1, 0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name in ("train/lpips", "train/dists"):
        torch.testing.assert_close(first_metrics[name], second_metrics[name], rtol=0, atol=0)
    # Isolate perceptual gradients so the pixel term cannot hide a detach/no_grad.
    (first_metrics["train/lpips"] + first_metrics["train/dists"]).backward()
    assert torch.isfinite(reconstruction.grad).all()
    assert reconstruction.grad.abs().sum() > 0
    assert all(parameter.grad is None for metric in (objective.perceptual_loss, objective.dists)
               for parameter in metric.parameters())
    for name, value in objective.named_buffers():
        if name in buffers:
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
