"""CPU contracts for LPIPS values, checkpoint compatibility and tiny features."""

from copy import deepcopy
from itertools import product
from unittest.mock import patch

import lpips
import pytest
import torch
from torch import nn

from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.perceptual import StableLPIPS, normalize_features


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def metrics():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(903)
        # The learned LPIPS weights ship with the package; random trunks avoid downloads.
        official = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).requires_grad_(False).eval()
        stable = StableLPIPS(net="alex", pnet_rand=True, verbose=False).requires_grad_(False).eval()
    stable.load_state_dict(official.state_dict(), strict=True)
    return official, stable


@pytest.mark.parametrize("magnitude", [0., 1e-24])
@pytest.mark.parametrize("after_relu", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_and_tiny_normalization_backward(magnitude, after_relu, dtype):
    feature = torch.full((1, 4, 2, 2), magnitude, dtype=dtype, requires_grad=True)
    value = feature.relu() if after_relu else feature
    normalized = normalize_features(value)
    torch.testing.assert_close(normalized, lpips.normalize_tensor(value), rtol=0, atol=0)
    loss = (normalized - .25).square().mean()
    loss.backward()
    assert torch.isfinite(feature.grad).all()
    if after_relu and magnitude == 0:
        assert torch.count_nonzero(feature.grad) == 0
    else:
        assert torch.count_nonzero(feature.grad) == feature.numel()


def test_tiny_positive_feature_exposes_official_sqrt_backward():
    feature = torch.full((1, 4, 1, 1), 1e-24, requires_grad=True)
    (lpips.normalize_tensor(feature.relu()) - .25).square().mean().backward()
    assert not torch.isfinite(feature.grad).all()


def test_normal_feature_values_and_gradients_match_official():
    generator = torch.Generator().manual_seed(37)
    original = torch.randn(2, 17, 4, 5, generator=generator, requires_grad=True)
    stable = original.detach().clone().requires_grad_()
    weights = torch.randn(original.shape, generator=generator)
    expected = lpips.normalize_tensor(original)
    actual = normalize_features(stable)
    expected_grad, = torch.autograd.grad((expected * weights).sum(), original)
    actual_grad, = torch.autograd.grad((actual * weights).sum(), stable)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("spatial,normalize,learned,version", list(product(
    [False, True], [False, True], [False, True], ["0.0", "0.1"])))
def test_official_distance_and_image_gradients(metrics, spatial, normalize, learned, version):
    official, stable = metrics
    for metric in metrics:
        metric.spatial, metric.lpips, metric.version = spatial, learned, version
    generator = torch.Generator().manual_seed(23)
    predicted = torch.rand(1, 3, 64, 64, generator=generator).requires_grad_()
    target = torch.rand(1, 3, 64, 64, generator=generator).requires_grad_()
    if not normalize:
        predicted = predicted.detach().mul(2).sub(1).requires_grad_()
        target = target.detach().mul(2).sub(1).requires_grad_()
    actual_inputs = [value.detach().clone().requires_grad_() for value in (predicted, target)]
    expected, expected_layers = official(predicted, target, retPerLayer=True, normalize=normalize)
    actual, actual_layers = stable(*actual_inputs, retPerLayer=True, normalize=normalize)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)
    assert len(actual_layers) == len(expected_layers) == official.L
    for actual_layer, expected_layer in zip(actual_layers, expected_layers):
        torch.testing.assert_close(actual_layer, expected_layer, rtol=2e-6, atol=2e-8)
    expected_grads = torch.autograd.grad(expected.mean(), (predicted, target))
    actual_grads = torch.autograd.grad(actual.mean(), actual_inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-9)
    with torch.no_grad():
        value_only = stable(*actual_inputs, normalize=normalize)
    torch.testing.assert_close(value_only, actual, rtol=0, atol=0)


def test_state_dict_and_pretrained_initialization_remain_official(metrics):
    official, stable = metrics
    assert StableLPIPS.__init__ is lpips.LPIPS.__init__
    official_state, stable_state = official.state_dict(), stable.state_dict()
    assert official_state.keys() == stable_state.keys()
    for name, value in official_state.items():
        assert torch.equal(stable_state[name], value), name
    official.load_state_dict(stable_state, strict=True)
    stable.load_state_dict(official_state, strict=True)


@pytest.mark.parametrize("net", ["vgg", "squeeze"])
@pytest.mark.parametrize("learned", [False, True])
def test_other_official_backbones_and_baseline_state(net, learned):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        options = dict(net=net, lpips=learned, pnet_rand=True, spatial=True, verbose=False)
        official = lpips.LPIPS(**options).requires_grad_(False).eval()
        stable = StableLPIPS(**options).requires_grad_(False).eval()
        stable.load_state_dict(official.state_dict(), strict=True)
        predicted = torch.rand(1, 3, 64, 80, requires_grad=True)
        target = torch.rand_like(predicted)
    actual_input = predicted.detach().clone().requires_grad_()
    expected, expected_layers = official(predicted, target, retPerLayer=True, normalize=True)
    actual, actual_layers = stable(actual_input, target, retPerLayer=True, normalize=True)
    assert actual.shape == (1, 1, 64, 80)
    assert len(actual_layers) == len(expected_layers) == official.L
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)
    for actual_layer, expected_layer in zip(actual_layers, expected_layers):
        torch.testing.assert_close(actual_layer, expected_layer, rtol=2e-6, atol=2e-8)
    expected_grad, = torch.autograd.grad(expected.mean(), predicted)
    actual_grad, = torch.autograd.grad(actual.mean(), actual_input)
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-9)
    official.load_state_dict(stable.state_dict(), strict=True)


class FeaturePyramid(nn.Module):
    def __init__(self, channels, after_relu):
        super().__init__()
        self.channels, self.after_relu = channels, after_relu

    def forward(self, image):
        feature = image.mean(dim=1, keepdim=True)
        if self.after_relu:
            feature = feature.relu()
        return tuple(feature.expand(-1, channels, -1, -1) for channels in self.channels)


@pytest.mark.parametrize("magnitude,after_relu", [(0., False), (0., True), (1e-24, True)])
def test_image_loss_uses_stable_lpips_for_tiny_features(metrics, magnitude, after_relu):
    metric = deepcopy(metrics[1])
    metric.spatial, metric.lpips, metric.version = False, True, "0.0"
    metric.net = FeaturePyramid(metric.chns, after_relu)
    original_normalizer = lpips.normalize_tensor
    with patch("models.bt_self_flow.perceptual.StableLPIPS", return_value=metric):
        objective = ImageLoss(lpips_weight=1., dists_weight=0., mse_weight=0.)
    assert isinstance(objective.lpips, StableLPIPS)
    predicted = torch.full((1, 3, 2, 2), magnitude, requires_grad=True)
    loss, values = objective(predicted, torch.full_like(predicted, .3))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(values["lpips"])
    assert torch.isfinite(predicted.grad).all()
    assert not any(parameter.requires_grad for parameter in objective.lpips.parameters())
    assert lpips.normalize_tensor is original_normalizer
    if magnitude > 0:
        assert predicted.grad.abs().sum() > 0
