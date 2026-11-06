"""Independent SF statistics and full RGB-head training/deployment contracts."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, clone_state, evaluation_state, read_checkpoint
from models.bt_self_flow.endpoint_self_flow import independent_spatial_l1, sf_parameters
from models.bt_self_flow.experiment import load_config, optimizer_groups, validate_endpoint_self_flow, validate_resume_contract
from test_endpoint_decoder import tiny_endpoint_model
from test_endpoint_entrypoint import ASSETS, setup_run, tiny_factory, trainer_args
from test_spatial_training import check_state_equal


def test_independent_statistics_use_the_exact_differentiable_student_gradient():
    x = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(871), requires_grad=True)
    teacher = torch.randn(x.shape, generator=torch.Generator().manual_seed(872), requires_grad=True)
    reference = x.detach().clone().requires_grad_()

    def normalize(value):
        centered = value - value.mean((-2, -1), keepdim=True)
        return centered / (centered.square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()

    error, count = independent_spatial_l1(x, teacher)
    (error / count).backward()
    (normalize(reference) - normalize(teacher.detach())).abs().mean().backward()
    torch.testing.assert_close(x.grad, reference.grad, atol=0, rtol=0)
    assert teacher.grad is None and error.dtype == torch.float32
    # A detached denominator misses the mean/variance gradient terms.
    detached = x.detach().clone().requires_grad_()
    mean = detached.detach().mean((-2, -1), keepdim=True)
    std = ((detached.detach() - mean).square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
    ((detached - mean) / std - normalize(teacher.detach())).abs().mean().backward()
    assert not torch.allclose(x.grad, detached.grad, atol=1e-6)
    torch.testing.assert_close(x.grad.sum((-2, -1)), torch.zeros(2, 3), atol=2e-8, rtol=0)


def test_independent_loss_ignores_channel_offsets_and_positive_scale_with_stable_bf16():
    teacher = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(873))
    student = teacher * torch.tensor([1.5, 2., 3.]).reshape(1, 3, 1, 1) + 4
    error, count = independent_spatial_l1(student, teacher)
    assert error / count < 2e-6
    low = student.to(torch.bfloat16).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        error, count = independent_spatial_l1(low, teacher.to(torch.bfloat16))
    assert error.dtype == torch.float32
    (error / count).backward()
    assert torch.isfinite(low.grad).all()
    constant = torch.ones(2, 3, 4, 5, requires_grad=True)
    total, count = independent_spatial_l1(constant, torch.zeros_like(constant))
    (total / count).backward()
    assert total == 0 and torch.isfinite(constant.grad).all()


def test_new_recipe_and_invalid_normalization_are_explicit():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/bt_self_flow/endpoint_sf_hybrid/new_sf_rgb.yaml")
    assert config["model"]["train_rgb_head"] is True
    assert config["endpoint_self_flow"]["normalization"] == "independent_spatial"
    assert config["endpoint_self_flow"]["coefficient"] == .2
    assert config["image_loss"] == dict(mse_weight=4., lpips_weight=1., dists_weight=.5)
    assert (config["train"]["world_size"], config["train"]["batch_size"],
            config["train"]["accumulation_steps"], config["train"]["effective_batch"]) == (2, 4, 1, 8)
    bad = deepcopy(config)
    bad["endpoint_self_flow"]["normalization"] = "detached_independent"
    with pytest.raises(ValueError, match="normalization"):
        validate_endpoint_self_flow(bad)
    previous = deepcopy(config)
    previous["endpoint_self_flow"]["normalization"] = "shared_student"
    with pytest.raises(ValueError, match="resume contract"):
        validate_resume_contract(previous, config)


def test_rgb_head_has_its_own_optimizer_group_and_is_excluded_from_sf():
    model = tiny_endpoint_model(train_rgb_head=True)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/bt_self_flow/endpoint_sf_hybrid/new_sf_rgb.yaml")
    groups, _ = optimizer_groups(model, config)
    rgb = next(group for group in groups if group["name"] == "rgb_head")
    assert {id(p) for p in rgb["params"]} == {id(p) for p in model.rgb_head.parameters()}
    assert all(p.requires_grad for p in model.rgb_head.parameters())
    assert not {id(p) for p in rgb["params"]} & {id(p) for p in sf_parameters(model)}
    assert rgb["lr"] == config["optimizer"]["vae_lr"]


def test_rgb_state_roundtrip_and_ema_averages_actual_full_head():
    model = tiny_endpoint_model(train_rgb_head=True).eval()
    ema = DeploymentEMA(model, decay=.5)
    before = clone_state(model.export_state())
    with torch.no_grad():
        model.rgb_head.weight.add_(.2)
        if model.rgb_head.bias is not None:
            model.rgb_head.bias.add_(.1)
    ema.update(model)
    for name, value in model.rgb_head.state_dict().items():
        torch.testing.assert_close(ema.state["rgb_head"][name], (value + before["rgb_head"][name]) / 2)
    raw = clone_state(model.export_state())
    receiver = tiny_endpoint_model(train_rgb_head=True).eval()
    receiver.load_bt_state(raw)
    check_state_equal(receiver.export_state(), raw)
    with evaluation_state(model, ema.state):
        check_state_equal(model.export_state()["rgb_head"], ema.state["rgb_head"])
    check_state_equal(model.export_state(), raw)
    without_head = deepcopy(raw)
    del without_head["rgb_head"]
    with pytest.raises(ValueError, match="missing rgb_head"):
        receiver.load_bt_state(without_head)
    with pytest.raises(ValueError, match="trained RGB head"):
        tiny_endpoint_model().load_bt_state(raw)


def test_real_new_sf_rgb_training_updates_head_and_resumes_exactly(tmp_path):
    config, source = setup_run(tmp_path)
    config["model"]["train_rgb_head"] = True
    config["endpoint_self_flow"].update(normalization="independent_spatial", coefficient=.2)
    initial = tiny_factory(config).rgb_head.weight.detach().clone()
    with patch.object(trainer, "make_model", side_effect=tiny_factory), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source), 0, 1, torch.device("cpu"))
        config["train"]["output"] = str(tmp_path / "split")
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
        paused = tmp_path / "split/paused.pt"
        trainer.train(config, trainer_args(source, resume=paused), 0, 1, torch.device("cpu"))
    full = read_checkpoint(tmp_path / "run/complete.pt")
    split = read_checkpoint(tmp_path / "split/complete.pt")
    for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "scaler", "rng_by_rank"):
        check_state_equal(full[key], split[key])
    assert not torch.equal(full["raw"]["rgb_head"]["weight"], initial)
    assert "rgb_head" in full["ema"]
