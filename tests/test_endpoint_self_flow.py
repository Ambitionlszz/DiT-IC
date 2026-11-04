"""Endpoint teachers, packet-quality pairing and shared-statistics gradients."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model

from models.bt_self_flow.checkpoint import DeploymentEMA, clone_state
from models.bt_self_flow.endpoint_self_flow import (
    EndpointEMATeacher, adjacent_quality, sf_parameters, shared_spatial_diagnostics,
    shared_spatial_l1)
from models.bt_self_flow.packet import RateControl
from models.sana_self_flow import unwrap_sana_transformer
from test_bt_backend import tiny_sana
from test_precision_mu_pixel_adapter import precision_model


torch.set_num_threads(1)
QUALITY = {"values": [0., .2, .4, .6, .8, 1.]}


def native_teacher_model():
    model = precision_model().eval()
    # Include real PEFT adapters at the final output projection. Reading its
    # base .weight directly would silently discard their trained contribution.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(313)
        model.DiT = get_peft_model(tiny_sana().requires_grad_(False), LoraConfig(
            r=2, lora_alpha=2, target_modules=["to_q", "to_v", "proj_out"], bias="none"))
    return model


def test_adjacent_quality_is_packet_exact_deterministic_and_top_has_no_teacher():
    config = {"quality": QUALITY}
    before = torch.get_rng_state().clone()
    for low, high in zip(QUALITY["values"][:-1], QUALITY["values"][1:]):
        assert adjacent_quality(config, RateControl(low)).low_code == RateControl(high).low_code
    assert adjacent_quality(config, RateControl(1.)) is None
    assert torch.equal(before, torch.get_rng_state())
    for quality in ({"values": [0., .4, .2]}, {"values": [0., 0., 1.]}, {"values": [0.]}):
        with pytest.raises(ValueError, match="quality levels"):
            adjacent_quality({"quality": quality}, RateControl(0.))
    with pytest.raises(ValueError, match="configured"):
        adjacent_quality(config, RateControl(.3))
    with pytest.raises(ValueError, match="whole-image"):
        adjacent_quality(config, RateControl(.2, .8, torch.ones(1, 1, 1, 1, dtype=torch.bool)))


def test_shared_stats_l1_equals_weighted_raw_distance_and_keeps_mean_and_scale_errors():
    teacher = torch.tensor([1., 2., 4., 7.]).reshape(1, 1, 2, 2)
    student = (teacher * 3 + 5).requires_grad_()
    epsilon = 1e-6
    std = torch.sqrt(student.detach().var(dim=(-2, -1), correction=0, keepdim=True) + epsilon)
    with patch("torch.nn.functional.cosine_similarity", side_effect=AssertionError("no cosine loss")):
        error, count = shared_spatial_l1(student, teacher, epsilon)
    torch.testing.assert_close(error / count, ((student - teacher).abs() / std).mean())
    assert error > 0  # Independent SN would make this positive affine case zero.
    same, _ = shared_spatial_l1(student, student.detach(), epsilon)
    assert same == 0
    shifted, _ = shared_spatial_l1(teacher, teacher + 3, epsilon)
    assert shifted > 0


def test_shared_stats_are_detached_teacher_has_no_gradient_and_fp32_is_used():
    student = torch.tensor([.2, .4, .8, 1.6], requires_grad=True).reshape(1, 1, 2, 2)
    student.retain_grad()
    teacher = torch.tensor([1., -.5, 2., -.2], requires_grad=True).reshape(1, 1, 2, 2)
    teacher.retain_grad()
    error, count = shared_spatial_l1(student, teacher)
    (error / count).backward()
    std = torch.sqrt(student.detach().var(dim=(-2, -1), correction=0, keepdim=True) + 1e-6)
    expected = torch.sign(student.detach() - teacher.detach()) / std / count
    torch.testing.assert_close(student.grad, expected, atol=0, rtol=0)
    assert teacher.grad is None and error.dtype == torch.float32
    low = student.detach().to(torch.bfloat16).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        error, count = shared_spatial_l1(low, teacher.detach().to(torch.bfloat16))
    assert error.dtype == torch.float32
    (error / count).backward()
    assert torch.isfinite(low.grad).all()


def test_constant_features_remain_finite_and_diagnostics_do_not_change_loss():
    student = torch.ones(2, 3, 2, 4, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    error, count = shared_spatial_l1(student, teacher)
    stats = shared_spatial_diagnostics(student, teacher)
    assert stats["sf_low_variance_fraction"] == 1
    assert stats["sf_low_variance_contribution"] == 1
    assert not any(value.requires_grad for value in stats.values())
    (error / count).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
    assert teacher.grad is None
    for epsilon in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="epsilon"):
            shared_spatial_l1(student, teacher, epsilon)
    with pytest.raises(ValueError, match="B,C,H,W"):
        shared_spatial_l1(student.flatten(2), teacher.flatten(2))


def test_full_native_teacher_uses_ema_own_analysis_precision_and_output_head_adapters():
    model = native_teacher_model()
    output_head = unwrap_sana_transformer(model.DiT).proj_out
    with torch.no_grad():
        output_head.lora_B["default"].weight.fill_(.2)
    ema = DeploymentEMA(model)
    rng = torch.get_rng_state().clone()
    teacher = EndpointEMATeacher(model)
    assert torch.equal(rng, torch.get_rng_state())
    teacher.load_deployment_ema(ema.state)
    assert not hasattr(teacher, "vae") and not hasattr(teacher, "pixel_adapter")
    assert len(unwrap_sana_transformer(teacher.DiT).transformer_blocks) == 4
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train()
    assert not teacher.training and not teacher.codec.training
    clean = torch.randn(2, 4, 8, 8, requires_grad=True)
    control = RateControl(.6)
    received = []
    handle = teacher.codec.register_forward_pre_hook(lambda module, args: received.append(args))
    endpoint = teacher(clean, control)
    handle.remove()
    assert len(received) == 1 and received[0][1].low_code == control.low_code
    assert not received[0][0].requires_grad
    with torch.no_grad():
        view = model.codec(clean.detach(), control)
        prediction, _ = model.predict(view["mean"], precision=view["delta"])
        expected = model.sched.step(prediction, view["mean"])
    torch.testing.assert_close(endpoint, expected, atol=2e-6, rtol=1e-6)
    assert endpoint.shape == clean.shape and not endpoint.requires_grad
    # A partial/truncated EMA or a direct base-weight projection cannot pass.
    teacher_head = unwrap_sana_transformer(teacher.DiT).proj_out
    teacher_head.enable_adapters(False)
    without_head_adapter = teacher(clean, control)
    assert not torch.equal(endpoint, without_head_adapter)
    teacher_head.enable_adapters(True)


def test_teacher_reloads_single_ema_and_rejects_missing_or_extra_dit_state():
    model = native_teacher_model()
    ema = DeploymentEMA(model, decay=.5)
    teacher = EndpointEMATeacher(model)
    teacher.load_deployment_ema(ema.state)
    key = next(name for name in ema.state["dit"] if "proj_out.lora_B" in name)
    initial = ema.state["dit"][key].clone()
    with torch.no_grad():
        dict(model.DiT.named_parameters())[key].add_(.4)
    ema.update(model)
    teacher.load_deployment_ema(ema.state)
    torch.testing.assert_close(teacher.DiT.state_dict()[key], initial + .2)
    fresh = EndpointEMATeacher(model)
    fresh.load_deployment_ema(clone_state(ema.state))
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name]), name
    missing = clone_state(ema.state)
    missing["dit"].pop(key)
    with pytest.raises(ValueError, match="complete, exact"):
        teacher.load_deployment_ema(missing)
    extra = clone_state(ema.state)
    extra["dit"]["not_a_lora_tensor"] = torch.zeros(1)
    with pytest.raises(ValueError, match="complete, exact"):
        teacher.load_deployment_ema(extra)
    missing_condition = clone_state(ema.state)
    missing_condition.pop("conditioner")
    with pytest.raises(ValueError, match="components"):
        teacher.load_deployment_ema(missing_condition)


def test_sf_whitelist_includes_all_existing_sana_adapters_and_excludes_shared_aliases():
    model = native_teacher_model()
    expected = {id(parameter) for parameter in model.DiT.parameters() if parameter.requires_grad}
    assert {id(parameter) for parameter in sf_parameters(model)} == expected
    assert id(unwrap_sana_transformer(model.DiT).proj_out.lora_B["default"].weight) in expected
    # Explicit alias test protects future wrappers registering a shared module.
    model.DiT.shared_conditioner = model.quality_conditioner
    excluded = {id(parameter) for parameter in model.quality_conditioner.parameters()}
    actual = {id(parameter) for parameter in sf_parameters(model)}
    assert actual == expected and not (actual & excluded)
