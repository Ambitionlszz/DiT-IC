"""Recomputation must preserve the complete PiT tail and perceptual objective."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch

from models.bt_self_flow.decoder import decode_latent_with_pixel_adapter
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.pixel_adapter import PixelDecoderAdapter
from test_pixel_adapter import SMALL_PIXEL, activate_output, real_small_ae


@pytest.mark.parametrize('amp', [False, True])
def test_complete_rgb_tail_checkpoint_preserves_outputs_and_all_gradients(amp):
    torch.manual_seed(903)
    vae = real_small_ae()
    adapter = PixelDecoderAdapter(8, 32, **SMALL_PIXEL, checkpoint_blocks=True)
    activate_output(adapter)
    checked_vae, checked_adapter = deepcopy(vae), deepcopy(adapter)
    source = torch.randn(1, 4, 2, 4)
    tokens = torch.randn(1, 8, 32)
    delta = torch.exp(torch.randn(1, 1, 1, 2))
    results = []
    for ae, pixel, recompute in ((vae, adapter, False), (checked_vae, checked_adapter, True)):
        inputs = [value.clone().requires_grad_() for value in (source, tokens, delta)]
        context = torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False) if amp else nullcontext()
        with context:
            image, captured = decode_latent_with_pixel_adapter(ae, inputs[0], pixel, inputs[1],
                                                               (2, 4), inputs[2], recompute, True)
            loss = image.float().square().mean() + .01 * captured['pixel_tokens'].float().square().mean()
        loss.backward()
        results.append((image.detach(), *[value.grad for value in inputs]))
        assert pixel.checkpoint_blocks is True
    tolerance = dict(atol=2e-4, rtol=2e-3) if amp else dict(atol=1e-7, rtol=1e-5)
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, **tolerance)
    for plain, checked in ((vae, checked_vae), (adapter, checked_adapter)):
        assert plain.state_dict().keys() == checked.state_dict().keys()
        for name, parameter in plain.named_parameters():
            other = dict(checked.named_parameters())[name]
            if parameter.grad is None:
                assert other.grad is None
            else:
                torch.testing.assert_close(parameter.grad, other.grad, **tolerance)


def test_checkpointed_actual_lpips_dists_preserves_loss_metrics_and_image_gradient():
    torch.manual_seed(903)
    native = ImageLoss(mse_weight=4., lpips_weight=1., dists_weight=1.).eval()
    checked = deepcopy(native)
    checked.checkpoint_features = True
    target = torch.rand(1, 3, 64, 64) * 2 - 1
    predicted = torch.rand_like(target) * 2 - 1
    results = []
    for objective in (native, checked):
        value = predicted.clone().requires_grad_()
        loss, metrics = objective(value, target)
        loss.backward()
        results.append((loss.detach(), metrics, value.grad))
        assert all(parameter.grad is None for parameter in objective.parameters())
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key], results[1][1][key], atol=0, rtol=0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=1e-7, rtol=1e-6)
