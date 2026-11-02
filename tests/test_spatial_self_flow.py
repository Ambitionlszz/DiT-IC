"""Spatial SF: real tiny SANA/codec, gradient roles, packets and EMA targets."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.bt_self_flow.checkpoint import DeploymentEMA, clone_state
from models.bt_self_flow.experiment import make_model, validate_spatial_self_flow
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.spatial_model import SpatialAblationCodec
from models.bt_self_flow.spatial_self_flow import (
    SpatialEMATeacher, SpatialProjector, active_samples, higher_quality,
    sf_parameters, spatial_normalize, spatial_response_error)
from models.bt_self_flow.spatial_training import spatial_backward
from scripts.launch_spatial_sf import ablation_configs, validate_initialization
from test_precision_mu_pixel_adapter import precision_model
from test_pixel_adapter import SMALL_PIXEL, activate_output


torch.set_num_threads(1)


def tiny_spatial_config(variant="c_d_sf"):
    config = ablation_configs()[variant]
    config["spatial_self_flow"].update(student_layer=1, teacher_layer=3, bypass_probability=0.)
    config["train"].update(world_size=1, batch_size=1, effective_batch=1, tensorboard=False)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    if config["decoder"]["method"] != "native":
        config["decoder"]["pixel_adapter"] = dict(SMALL_PIXEL, seed=903, checkpoint_blocks=True)
    return config


def tiny_spatial_model(config=None, device="cpu"):
    config = config or tiny_spatial_config()
    def initialize(self, *args, **kwargs):
        base = precision_model()
        del base.pixel_adapter
        self.__dict__.update(base.__dict__)
    with patch.object(BTCodec, "__init__", initialize):
        return make_model(config, torch.device(device))


def test_six_recipes_only_change_declared_ablation_switches():
    configs = ablation_configs()
    assert set(configs) == {"c", "c_d", "c_sf", "c_d_sf", "c_p", "c_d_p"}
    for config in configs.values():
        t = config["train"]
        assert (t["world_size"], t["batch_size"], t["accumulation_steps"], t["effective_batch"]) == (2, 4, 1, 8)
        assert t["steps"] == 200000
        assert config["spatial_self_flow"]["coefficient"] == 1
    for change in ({"coefficient": .1}, {"enabled": True, "projector_enabled": False}, {"cosine": True}):
        config = deepcopy(configs["c_sf"])
        config["spatial_self_flow"].update(change)
        with pytest.raises(ValueError):
            validate_spatial_self_flow(config)


def test_normalization_uses_space_preserves_gradients_and_excludes_padding():
    value = torch.randn(2, 7, 4, requires_grad=True)
    normalized = spatial_normalize(value)
    torch.testing.assert_close(normalized.mean(1), torch.zeros(2, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(normalized.square().mean(1), torch.ones(2, 4), atol=2e-5, rtol=0)
    shifted = value * torch.tensor([1., 2., 3., 4.]) + torch.tensor([2., -3., 5., 1.])
    torch.testing.assert_close(normalized, spatial_normalize(shifted), atol=2e-5, rtol=0)
    valid = torch.ones(2, 7, dtype=torch.bool)
    padded = torch.cat([value, torch.full((2, 2, 4), 1e5)], 1)
    valid = torch.cat([valid, torch.zeros(2, 2, dtype=torch.bool)], 1)
    torch.testing.assert_close(normalized, spatial_normalize(padded, valid=valid)[:, :7])
    teacher = torch.randn_like(value, requires_grad=True)
    error, count = spatial_response_error(value, teacher)
    (error / count).backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert teacher.grad is None
    constant = torch.ones(2, 7, 4, requires_grad=True)
    assert torch.equal(spatial_normalize(constant), torch.zeros_like(constant))
    spatial_normalize(constant).sum().backward()
    assert torch.isfinite(constant.grad).all()


def test_spatial_l1_penalizes_wrong_positions_and_has_no_cosine_call():
    teacher = torch.tensor([10., 10., 12., 12.]).reshape(1, 4, 1)
    correct = torch.tensor([2., 2., 3., 3.]).reshape(1, 4, 1)
    wrong = torch.tensor([2., 3., 2., 3.]).reshape(1, 4, 1)
    with patch("torch.nn.functional.cosine_similarity", side_effect=AssertionError("cosine is forbidden")):
        a, n = spatial_response_error(correct, teacher)
        b, _ = spatial_response_error(wrong, teacher)
    assert float(a / n) < 3e-6 and float(b / n) > .99


def test_projector_identity_and_random_stream_isolation():
    torch.manual_seed(23)
    before = torch.get_rng_state().clone()
    projector = SpatialProjector(5)
    assert torch.equal(before, torch.get_rng_state())
    tokens = torch.randn(3, 12, 5)
    torch.testing.assert_close(projector(tokens, 3, 4), tokens, atol=0, rtol=0)
    with torch.no_grad():
        projector.conv.bias.add_(.3)
    active = torch.tensor([True, False, True])
    result = projector(tokens, 3, 4, active)
    torch.testing.assert_close(result[1], tokens[1], atol=0, rtol=0)
    assert not torch.equal(result[0], tokens[0])
    settings = dict(bypass_probability=.1)
    before = torch.get_rng_state().clone()
    mask = active_samples(settings, 16, 903, 7, 1, 0, torch.device("cpu"))
    assert torch.equal(mask, active_samples(settings, 16, 903, 7, 1, 0, torch.device("cpu")))
    higher_quality(ablation_configs()["c_sf"], RateControl(.2), 7, 1)
    assert torch.equal(before, torch.get_rng_state())


def test_initial_parity_and_independent_receiver_switches():
    mu, delta = torch.randn(1, 4, 8, 8), torch.ones(1, 1, 4, 4)
    results, shared = [], None
    for variant in ("c", "c_d", "c_sf", "c_d_sf"):
        model = tiny_spatial_model(tiny_spatial_config(variant)).eval()
        state = model.export_state()
        public = {part: values for part, values in state.items() if part not in ("pixel_adapter", "spatial_projector")}
        if shared is not None:
            for part, values in public.items():
                for name, value in values.items():
                    assert torch.equal(value, shared[part][name]), (variant, part, name)
        shared = clone_state(public)
        with torch.no_grad():
            results.append(model.decode_received(mu, delta))
    for output in results[1:]:
        for name in ("prediction", "latent", "image"):
            torch.testing.assert_close(output[name], results[0][name], atol=2e-6, rtol=1e-6)
    activate_output(model.pixel_adapter)
    with torch.no_grad():
        model.spatial_projector.conv.weight.mul_(1.05)
        full = model.decode_received(mu, delta)
        with model.decoder_switches(sf=False, d=True):
            no_sf = model.decode_received(mu, delta)
        with model.decoder_switches(sf=True, d=False):
            no_d = model.decode_received(mu, delta)
    assert not torch.equal(full["latent"], no_sf["latent"])
    assert torch.equal(full["latent"], no_d["latent"])
    assert not torch.equal(full["image"], no_d["image"])
    assert model.p_enabled and model.d_enabled


def test_teacher_uses_ema_codec_precision_and_main_path_projector():
    model = tiny_spatial_model()
    ema = DeploymentEMA(model, decay=.5)
    old = ema.state["spatial_projector"]["conv.bias"].clone()
    with torch.no_grad():
        model.spatial_projector.conv.bias.add_(.2)
    ema.update(model)
    torch.testing.assert_close(ema.state["spatial_projector"]["conv.bias"], old + .1)
    teacher = SpatialEMATeacher(model)
    teacher.load_deployment_ema(ema.state)
    for part in ("codec", "spatial_projector"):
        for name, value in getattr(teacher, part).state_dict().items():
            assert torch.equal(value, ema.state[part][name]), (part, name)
    clean = torch.randn(1, 4, 8, 8)
    calls = []
    hook = teacher.codec.register_forward_pre_hook(lambda module, args: calls.append(args))
    target = teacher(clean, RateControl(.8))
    hook.remove()
    assert calls[0][1].value == pytest.approx(.8, abs=2e-5)
    assert target.shape == (1, 64, 32) and not target.requires_grad
    assert len(teacher.DiT.get_base_model().transformer_blocks) == 3
    assert all(not p.requires_grad for p in teacher.parameters())


def test_sf_gradient_is_added_only_to_p_and_early_sana_not_c_ae_d_or_late_layers():
    config = tiny_spatial_config()
    model = tiny_spatial_model(config).train()
    baseline = deepcopy(model)
    teacher = SpatialEMATeacher(model)
    teacher.load_deployment_ema(DeploymentEMA(model).state)
    image = torch.randn(1, 3, 256, 256)
    image_loss = ImageLoss(**config["image_loss"])
    rng = torch.get_rng_state().clone()
    result, loss, metrics = spatial_backward(model, teacher, image, RateControl(.2), image_loss, config,
                                            0, 0, 1, torch.amp.GradScaler("cuda", enabled=False), torch.bfloat16)
    torch.set_rng_state(rng)
    reference = baseline(image, RateControl(.2))
    main, _ = reconstruction_objective(reference, image, RateControl(.2), baseline.distortion_scale.s_data,
                                        image_loss, objective=config["objective"], quality=config["quality"])
    main.backward()
    allowed = {id(p) for p in sf_parameters(model)}
    changed = []
    for (name, parameter), (other_name, reference_parameter) in zip(model.named_parameters(), baseline.named_parameters()):
        assert name == other_name
        if id(parameter) not in allowed:
            if parameter.grad is None or reference_parameter.grad is None:
                assert parameter.grad is reference_parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0, msg=name)
        elif parameter.grad is not None and reference_parameter.grad is not None:
            if not torch.equal(parameter.grad, reference_parameter.grad):
                changed.append(name)
    assert any(name.startswith("spatial_projector.") for name in changed)
    assert metrics["sf_weight"] == 1 and torch.isfinite(loss)
    assert all(p.grad is None for p in teacher.parameters())


def test_packets_use_received_state_and_exact_state_loading_rejects_missing_modules():
    model = tiny_spatial_model().eval()
    activate_output(model.pixel_adapter)
    with torch.no_grad():
        model.spatial_projector.conv.bias.add_(.02)
    receiver = tiny_spatial_model().eval()
    receiver.load_bt_state(model.export_state())
    image = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    for q in (0., .4, 1.):
        packet = model.compress(image, RateControl(q))
        expected = model(image, RateControl(q)).image
        with patch.object(receiver, "encode_clean_latent", side_effect=AssertionError("sender encoder used")), \
             patch.object(receiver.codec.router, "forward", side_effect=AssertionError("source router used")):
            decoded = receiver.decompress(packet.to_bytes())
        torch.testing.assert_close(expected, decoded, atol=2e-5, rtol=1e-5)
    state = dict(model.export_state())
    state.pop("spatial_projector")
    with pytest.raises(ValueError, match="spatial_projector"):
        receiver.load_bt_state(state)
    with pytest.raises(ValueError, match="pure-C"):
        validate_initialization(dict(phase="backend_init", backend="shared", raw=model.export_state()))


def test_checkpoint_recomputation_keeps_main_path_and_gradients():
    model = tiny_spatial_model(tiny_spatial_config("c_sf")).train()
    mu, delta = torch.randn(1, 4, 8, 8), torch.ones(1, 1, 4, 4)
    results = []
    for checkpointed in (False, True):
        model.zero_grad(set_to_none=True)
        model.checkpoint_blocks = checkpointed
        prediction, feature = model.predict(mu, precision=delta, capture=True)
        (prediction.square().mean() + spatial_normalize(feature).abs().mean()).backward()
        results.append((prediction.detach(), model.spatial_projector.conv.weight.grad.clone()))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
