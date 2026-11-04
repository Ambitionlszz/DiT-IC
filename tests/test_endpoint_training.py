"""Actual-output gradient isolation, quality controls, RNG and global means."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from models.bt_self_flow.checkpoint import DeploymentEMA
from models.bt_self_flow.endpoint_self_flow import EndpointEMATeacher, sf_parameters, shared_spatial_l1
from models.bt_self_flow.endpoint_training import average_gradients, endpoint_backward
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl


torch.set_num_threads(1)


def endpoint_config(target="adjacent_ema", enabled=True):
    return {
        "endpoint_self_flow": {"enabled": enabled, "coefficient": 1., "epsilon": 1e-6,
                               "target": target},
        "train": {"accumulation_steps": 1},
        "quality": {"values": [0., .2, .4, .6, .8, 1.],
                    "rate_coefficients": [16., 8., 4., 2., 1., .5]},
        "objective": {"base_weight": .1, "repaired_weight": .1},
    }


class TinyEndpoint(nn.Module):
    """A real differentiable C->DiT->AE->D graph with distinct parameter roles."""

    def __init__(self, enabled=True):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(103)
            self.codec = nn.Conv2d(2, 2, 1)
            self.text_proj = nn.Conv2d(2, 2, 1)
            self.quality_conditioner = nn.Conv2d(2, 2, 1)
            self.DiT = nn.Sequential(nn.Conv2d(2, 4, 1), nn.SiLU(), nn.Conv2d(4, 2, 1))
            self.vae = nn.Sequential(nn.Conv2d(2, 3, 1), nn.SiLU())
            self.pixel_adapter = nn.Conv2d(3, 3, 1)
        self.distortion_scale = SimpleNamespace(s_data=torch.tensor(1.))
        self.sf_enabled = enabled
        self.endpoint_settings = endpoint_config(enabled=enabled)["endpoint_self_flow"]
        self.captures, self.target_grad_modes = [], []

    def forward(self, image, rate_control=None, capture=False):
        self.captures.append(capture)
        clean = image[:, :2].detach()
        coarse = self.codec(clean)
        context = self.text_proj(torch.ones_like(coarse))
        condition = self.quality_conditioner(torch.ones_like(coarse) * rate_control.value)
        latent = coarse - self.DiT(coarse + context + condition)
        feature = self.vae(latent)
        output = feature + self.pixel_adapter(feature)
        return SimpleNamespace(clean=clean, coarse=coarse, latent=latent, image=output,
                               feature=feature if capture else None, side_bits=0,
                               likelihoods={"y": torch.sigmoid(coarse) * .5 + .25})

    def endpoint_feature(self, latent):
        self.target_grad_modes.append(torch.is_grad_enabled())
        return self.vae(latent)


class TinyTeacher(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.17))
        self.stochastic, self.qualities = stochastic, []

    @torch.no_grad()
    def forward(self, clean, control):
        self.qualities.append(control.value)
        noise = torch.rand_like(clean) * .03 if self.stochastic else 0.
        return clean.detach() + self.offset + control.value * .2 + noise


def disabled_scaler():
    return torch.amp.GradScaler("cuda", enabled=False)


def main_loss(model, image, control, config, capture=False):
    result = model(image, rate_control=control, capture=capture)
    loss, _ = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                        ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.),
                                        objective=config["objective"], quality=config["quality"])
    return result, loss


def test_endpoint_sf_changes_whole_sana_only_including_final_output_parameters():
    model, teacher, config = TinyEndpoint(), TinyTeacher(), endpoint_config()
    baseline = deepcopy(model)
    image = torch.randn(4, 3, 3, 5, generator=torch.Generator().manual_seed(42))
    control = RateControl(.2)
    image_loss = ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.)
    result, loss, metrics = endpoint_backward(model, teacher, image, control, image_loss, config,
                                             0, 0, 1, disabled_scaler(), torch.bfloat16)
    _, rd = main_loss(baseline, image, control, config)
    rd.backward()
    allowed = {id(parameter) for parameter in sf_parameters(model)}
    changed = []
    for (name, parameter), (other_name, reference) in zip(model.named_parameters(), baseline.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and reference.grad is not None, name
        if id(parameter) not in allowed:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0, msg=name)
        elif not torch.equal(parameter.grad, reference.grad):
            changed.append(name)
    assert set(changed) == {name for name, _ in model.DiT.named_parameters(prefix="DiT")}
    assert "DiT.2.weight" in changed
    assert metrics["sf_active"] == metrics["sf_weight"] == 1
    assert torch.isfinite(loss) and result.feature.requires_grad
    assert teacher.offset.grad is None and model.target_grad_modes == [False]


@pytest.mark.parametrize("target", ["adjacent_ema", "same_rate_ema", "clean_latent"])
def test_target_controls_use_matched_activation_and_clean_target_needs_no_teacher(target):
    model, config = TinyEndpoint(), endpoint_config(target)
    teacher = None if target == "clean_latent" else TinyTeacher()
    image, control = torch.randn(2, 3, 3, 5), RateControl(.4)
    _, _, metrics = endpoint_backward(model, teacher, image, control,
                                      ImageLoss(lpips_weight=0., dists_weight=0.), config,
                                      0, 0, 1, disabled_scaler(), torch.bfloat16)
    assert model.captures == [True] and metrics["sf_active"] == 1
    if teacher is not None:
        expected = RateControl(.6 if target == "adjacent_ema" else .4).value
        assert teacher.qualities == [expected] and metrics["teacher_q"] == pytest.approx(expected)


@pytest.mark.parametrize("target", ["adjacent_ema", "same_rate_ema", "clean_latent"])
def test_highest_quality_skips_capture_teacher_and_target_decoder_for_every_control(target):
    model, config = TinyEndpoint(), endpoint_config(target)
    image = torch.randn(2, 3, 3, 5)
    teacher = TinyTeacher()
    with patch.object(teacher, "forward", side_effect=AssertionError("highest-q teacher forward")), \
         patch.object(model, "endpoint_feature", side_effect=AssertionError("highest-q target decoder")):
        _, _, metrics = endpoint_backward(model, teacher, image, RateControl(1.),
                                          ImageLoss(lpips_weight=0., dists_weight=0.), config,
                                          0, 0, 1, disabled_scaler(), torch.bfloat16)
    assert model.captures == [False]
    assert metrics["sf_active"] == metrics["sf"] == metrics["sf_weight"] == 0


def test_disabled_sf_matches_main_gradients_and_never_requires_teacher():
    model, config = TinyEndpoint(False), endpoint_config(enabled=False)
    baseline, image, control = deepcopy(model), torch.randn(2, 3, 3, 5), RateControl(.2)
    _, loss, metrics = endpoint_backward(model, None, image, control,
                                         ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.), config,
                                         0, 0, 1, disabled_scaler(), torch.bfloat16)
    _, rd = main_loss(baseline, image, control, config)
    rd.backward()
    assert model.captures == [False] and metrics["sf"] == 0
    torch.testing.assert_close(loss, rd)
    for parameter, reference in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)


def test_teacher_randomness_does_not_advance_main_training_rng():
    model, config = TinyEndpoint(), endpoint_config()
    image = torch.randn(2, 3, 3, 5)
    torch.manual_seed(471)
    before = torch.get_rng_state().clone()
    endpoint_backward(model, TinyTeacher(stochastic=True), image, RateControl(.2),
                      ImageLoss(lpips_weight=0., dists_weight=0.), config,
                      0, 0, 1, disabled_scaler(), torch.bfloat16)
    assert torch.equal(before, torch.get_rng_state())


def test_accumulation_and_scaling_apply_equally_to_rd_and_sf():
    model, reference, teacher = TinyEndpoint(), TinyEndpoint(), TinyTeacher()
    config, image, control = endpoint_config(), torch.randn(2, 3, 3, 5), RateControl(.2)
    config["train"]["accumulation_steps"] = 2
    factor = 8.
    scale = SimpleNamespace(scale=lambda loss: factor * loss)
    endpoint_backward(model, teacher, image, control,
                      ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.), config,
                      0, 0, 1, scale, torch.bfloat16)
    result, rd = main_loss(reference, image, control, config, capture=True)
    with torch.no_grad():
        target = reference.endpoint_feature(teacher(result.clean, RateControl(.4)))
    total, count = shared_spatial_l1(result.feature, target)
    allowed = sf_parameters(reference)
    extra = torch.autograd.grad(factor * total / count / 2, allowed, retain_graph=True)
    (factor * rd / 2).backward()
    for parameter, gradient in zip(allowed, extra):
        parameter.grad.add_(gradient)
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_real_sana_codec_decoder_sf_respects_gradient_whitelist_and_rng(checkpoint_blocks):
    from test_endpoint_decoder import tiny_endpoint_model

    model = tiny_endpoint_model(checkpoint_blocks=checkpoint_blocks).train()
    reference = deepcopy(model)
    config, control = endpoint_config(), RateControl(.2)
    teacher = EndpointEMATeacher(model)
    teacher.load_deployment_ema(DeploymentEMA(model).state)
    image = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(678))
    image_loss = ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.)
    rng = torch.get_rng_state().clone()
    _, loss, metrics = endpoint_backward(model, teacher, image, control, image_loss, config,
                                         0, 0, 1, disabled_scaler(), torch.bfloat16)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    baseline = reference(image, rate_control=control, capture=True)
    rd, _ = reconstruction_objective(baseline, image, control, reference.distortion_scale.s_data,
                                      image_loss, objective=config["objective"], quality=config["quality"])
    rd.backward()
    assert torch.equal(after, torch.get_rng_state())
    allowed = {id(parameter) for parameter in sf_parameters(model)}
    changed = []
    for (name, parameter), (other_name, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if parameter.grad is None or expected.grad is None:
            assert parameter.grad is expected.grad is None, name
        elif id(parameter) in allowed:
            assert torch.isfinite(parameter.grad).all(), name
            if not torch.equal(parameter.grad, expected.grad):
                changed.append(name)
        else:
            torch.testing.assert_close(parameter.grad, expected.grad, atol=0, rtol=0, msg=name)
    assert any("transformer_blocks.0." in name for name in changed)
    assert any("transformer_blocks.3." in name for name in changed)
    assert torch.isfinite(loss) and metrics["sf_active"] == 1
    assert all(parameter.grad is None for parameter in teacher.parameters())


def endpoint_mean_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{Path(directory) / 'rendezvous'}",
                            timeout=timedelta(seconds=90))
    try:
        image = torch.randn(8, 3, 3, 5, generator=torch.Generator().manual_seed(620))
        for quality in (.2, 1.):
            model, config = TinyEndpoint(), endpoint_config()
            endpoint_backward(model, TinyTeacher(), image[rank * 4:(rank + 1) * 4], RateControl(quality),
                              ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.), config,
                              0, rank, 2, disabled_scaler(), torch.bfloat16)
            average_gradients(model.parameters(), 2)
            if rank == 0:
                reference = TinyEndpoint()
                endpoint_backward(reference, TinyTeacher(), image, RateControl(quality),
                                  ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.), config,
                                  0, 0, 1, disabled_scaler(), torch.bfloat16)
                for parameter, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-5)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_batch4_mean_matches_single_global_batch8_with_highest_branch(tmp_path):
    mp.spawn(endpoint_mean_worker, args=(str(tmp_path),), nprocs=2, join=True)
