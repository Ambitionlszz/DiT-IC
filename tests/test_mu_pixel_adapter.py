"""Mu-only LN PiT A/B: attention sources, whole receiver and training contract."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.experiment import (check_stage_model, load_config, make_model, optimizer_groups,
                                            training_contract, validate_decoder)
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.mu_pixel_model import MuQKCPixelCodec, MuPixelQKCPixelCodec, METHOD as MAIN, CONTROL_METHOD
from models.bt_self_flow.pixel_adapter import MuPixelAdapter, MuGuidedAttention, RMSNorm, apply_rotary, rotary_position
from models.sana_self_flow import quality_conditioned_sana_forward
from test_bt_training import small_config
from test_c_precision_v3 import CODEC, activate_conditions
from test_pixel_adapter import QUALITY, SMALL_PIXEL, activate_output
from test_precision_pixel_adapter import precision_pixel_model
from test_received_update_pixel_adapter import same_state


ROOT = Path(__file__).resolve().parents[1]


def mu_pixel_model(config=None, device="cpu", qk_source="mu"):
    model = precision_pixel_model()
    if config:
        qk_source = "mu" if config["decoder"]["method"] == MAIN else "pixels"
    model.__class__ = MuQKCPixelCodec if qk_source == "mu" else MuPixelQKCPixelCodec
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(903)
        model.pixel_adapter = MuPixelAdapter(8, 4, **SMALL_PIXEL, checkpoint_blocks=True, qk_source=qk_source)
    return model.to(device)


def small_adapter(qk_source="mu", **kwargs):
    settings = dict(pixel_dim=4, patch_size=4, context_dim=16, attention_dim=16,
                    heads=2, checkpoint_blocks=False)
    settings.update(kwargs)
    return MuPixelAdapter(8, 4, qk_source=qk_source, **settings)


def recipe(mode):
    return load_config(ROOT / f"configs/bt_self_flow/precision_v3/mu_pit_{mode}_joint.yaml")


def test_a_b_recipes_only_change_qk_source_and_output_directory():
    native, a, b = load_config(ROOT / "configs/bt_self_flow/precision_v3/c_joint.yaml"), recipe("a"), recipe("b")
    for cfg in (a, b):
        for key in ("model", "data", "quality", "objective", "image_loss", "optimizer"):
            assert cfg[key] == native[key]
        check_stage_model(cfg, native)
        assert cfg["train"]["steps"] == 200000
        assert cfg["train"]["batch_size"] == cfg["train"]["effective_batch"] == 8
        assert cfg["decoder"]["pixel_adapter"]["depth"] == 2
    assert a["decoder"]["method"] == CONTROL_METHOD and b["decoder"]["method"] == MAIN
    assert a["train"]["output"] != b["train"]["output"]
    dataset = SimpleNamespace(fingerprint="same_data")
    assert training_contract(a, dataset, 1) != training_contract(b, dataset, 1)
    a["train"]["output"] = b["train"]["output"]
    a["decoder"]["method"] = b["decoder"]["method"]
    assert a == b


@pytest.mark.parametrize("mutation", ["legacy_step", "scale_gate", "qk_source", "width", "sf", "frontend", "backend"])
def test_config_rejects_unsupported_or_incompatible_options(mutation):
    cfg = recipe("b")
    if mutation in ("legacy_step", "scale_gate", "qk_source"):
        key = {"legacy_step": "delta_ref", "scale_gate": "use_quantization", "qk_source": "qk_source"}[mutation]
        cfg["decoder"]["pixel_adapter"][key] = 1
    elif mutation == "width":
        cfg["decoder"]["pixel_adapter"]["attention_dim"] = 128
    elif mutation == "sf":
        cfg["self_flow"]["enabled"] = True
    elif mutation == "frontend":
        cfg["model"]["codec"]["precision_consistent"] = False
    else:
        cfg["model"]["backend"] = "D"
    with pytest.raises(ValueError):
        validate_decoder(cfg)


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_normalization_condition_order_and_initialization(source):
    adapter = small_adapter(source)
    mu = torch.randn(1, 4, 2, 3)
    actual = adapter.condition(mu, (4, 6))
    mapped = adapter.content_projection(mu)
    expected = adapter.content_norm(torch.nn.functional.interpolate(
        mapped, size=(4, 6), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (1, 24, 16)
    assert not any(isinstance(module, RMSNorm) for module in adapter.modules())
    assert not hasattr(adapter, "output_norm")
    norms = [module for module in adapter.modules() if isinstance(module, nn.LayerNorm)]
    assert len(norms) == 5  # shared content plus two pre-AdaLN pixel norms per block
    for norm in norms:
        assert norm.eps == 1e-6
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))
        assert torch.equal(norm.bias, torch.zeros_like(norm.bias))
    assert torch.count_nonzero(adapter.pixel_output.weight) == 0
    assert torch.count_nonzero(adapter.pixel_output.bias) == 0
    for block in adapter.blocks:
        assert block.adaLN_modulation[0].out_features == 16 * 6 * 4
        assert torch.count_nonzero(block.adaLN_modulation[0].weight) > 0
        assert not hasattr(block.attn, "q_norm") and not hasattr(block.attn, "k_norm")


def test_mode_initialization_is_capacity_and_parameter_matched():
    torch.manual_seed(903)
    a = small_adapter("pixels")
    torch.manual_seed(903)
    b = small_adapter("mu")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    for key, value in a.state_dict().items():
        if key != "qk_source_code":
            assert torch.equal(value, b.state_dict()[key]), key
    with pytest.raises(RuntimeError, match="Q/K source"):
        b.load_state_dict(a.state_dict())


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_attention_sources_and_standard_rope_softmax_math(source):
    attention = MuGuidedAttention(16, 2, source)
    pixels, content = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    position = rotary_position(8, 2, 3, "cpu")
    received, handles = {}, []
    for name in ("q", "k", "v"):
        def record(module, args, name=name):
            received[name] = args[0].detach().clone()
        handles.append(getattr(attention, name).register_forward_pre_hook(record))
    actual = attention(pixels, content, position)
    for handle in handles:
        handle.remove()
    for name in ("q", "k"):
        torch.testing.assert_close(received[name], content if source == "mu" else pixels, atol=0, rtol=0)
    torch.testing.assert_close(received["v"], pixels, atol=0, rtol=0)
    # Explicit score matrix verifies head order, RoPE axes and standard scale.
    origin = content if source == "mu" else pixels
    q, k = [getattr(attention, name)(origin).reshape(2, 6, 2, 8) for name in ("q", "k")]
    q, k = apply_rotary(q, k, position)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = attention.v(pixels).reshape(2, 6, 2, 8).transpose(1, 2)
    weights = ((q @ k.transpose(-1, -2)) / (8 ** .5)).softmax(-1)
    expected = attention.proj((weights @ v).transpose(1, 2).reshape(2, 6, 16))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_zero_writeback_opens_all_internal_gradients_after_update(source):
    adapter = small_adapter(source, checkpoint_blocks=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    features = torch.randn(2, 8, 16, 24)
    mu = torch.randn(2, 4, 2, 3, requires_grad=True)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        mu.grad = None
        actual = adapter(features, mu)
        if step == 0:
            assert torch.equal(actual, features)
        actual.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        optimizer.step()
    for name, parameter in adapter.named_parameters():
        assert parameter.grad.abs().sum() > 0, name
    assert mu.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("source", ["mu", "pixels"])
@torch.backends.cudnn.flags(benchmark=False, deterministic=True)
def test_checkpoint_adapter_forward_and_gradients(source, bf16, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(903)
    plain = small_adapter(source).to(device)
    activate_output(plain)
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    inputs = (torch.randn(1, 8, 16, 24, device=device), torch.randn(1, 4, 2, 3, device=device))
    results = []
    for adapter in (plain, checked):
        x, mu = [value.clone().requires_grad_() for value in inputs]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            actual = adapter(x, mu)
        actual.square().mean().backward()
        results.append((actual.detach(), x.grad, mu.grad))
    # CUDA backward accumulation may round differently when the shared content
    # is recomputed. Keep forward bit-exact and allow only small gradient noise;
    # CPU retains the exact comparison for the entire checkpointed path.
    tolerance = dict(atol=1e-8, rtol=1e-4) if device == "cuda" else dict(atol=0, rtol=0)
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for left, right in zip(results[0][1:], results[1][1:]):
        torch.testing.assert_close(left, right, **tolerance)
    for left, right in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
        assert torch.isfinite(left.grad).all()


@pytest.mark.parametrize("mode,cls", [("a", MuPixelQKCPixelCodec), ("b", MuQKCPixelCodec)])
def test_factory_preserves_c_weights_rng_and_freezing(mode, cls):
    def initialize_base(self, *args, **kwargs):
        native = precision_pixel_model()
        del native.pixel_adapter
        self.__dict__.update(native.__dict__)
    expected = BTCodec.__new__(BTCodec)
    initialize_base(expected)
    expected_state = clone_state(expected.export_state())
    expected_rng = torch.get_rng_state().clone()
    cfg = recipe(mode)
    cfg["decoder"]["pixel_adapter"] = deepcopy(SMALL_PIXEL)
    with patch.object(BTCodec, "__init__", initialize_base):
        model = make_model(cfg, torch.device("cpu"))
    assert isinstance(model, cls)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for part, values in expected_state.items():
        for name, value in values.items():
            assert torch.equal(value, model.export_state()[part][name]), (part, name)
    assert all(not p.requires_grad for p in model.vae.encoder.parameters())


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_initial_native_c_parity_no_hidden_capture_and_one_sana_call(source):
    model = mu_pixel_model(qk_source=source).eval()
    activate_conditions(model)
    mu = torch.randn(1, 4, 8, 12)
    delta = torch.exp(torch.randn(1, 1, 4, 6))
    with torch.no_grad():
        expected_latent, expected_image, _ = BTCodec.repair(model, mu, precision=delta)
        calls = model.prediction_calls
        with patch("models.bt_self_flow.model.quality_conditioned_sana_forward",
                   wraps=quality_conditioned_sana_forward) as run:
            actual = model.decode_received(mu, delta, return_features=True)
    assert run.call_count == 1 and run.call_args.kwargs.get("capture_layer") is None
    assert model.prediction_calls - calls == 1
    assert "sana_tokens" not in actual
    torch.testing.assert_close(actual["latent"], expected_latent, atol=0, rtol=0)
    torch.testing.assert_close(actual["image"], expected_image, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_whole_decoder_recompute_and_end_to_end_condition_gradients(source, bf16):
    plain = mu_pixel_model(qk_source=source).train()
    activate_conditions(plain)
    activate_output(plain.pixel_adapter)
    plain.checkpoint_blocks = plain.pixel_adapter.checkpoint_blocks = False
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    inputs = (torch.randn(1, 4, 8, 12), torch.exp(torch.randn(1, 1, 4, 6)))
    results = []
    for model in (plain, checked):
        mu, delta = [value.clone().requires_grad_() for value in inputs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            result = model.decode_received(mu, delta, return_features=True)
        result["image"].square().mean().backward()
        results.append((result["image"].detach(), mu.grad, delta.grad))
        assert mu.grad.abs().sum() > 0 and delta.grad.abs().sum() > 0
        for parameter in model.pixel_adapter.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    tolerance = dict(atol=2e-4, rtol=2e-3) if bf16 else dict(atol=1e-6, rtol=1e-6)
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, **tolerance)
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, **tolerance, msg=name)


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_joint_image_loss_keeps_existing_c_and_d_gradients(source):
    model = mu_pixel_model(qk_source=source).train()
    activate_conditions(model)
    activate_output(model.pixel_adapter)
    image = torch.rand(1, 3, 256, 256) * 2 - 1
    result = model(image, RateControl(.4), capture=True)
    loss, _ = reconstruction_objective(result, image, RateControl(.4), model.distortion_scale.s_data,
                                       ImageLoss(mse_weight=4., lpips_weight=0., dists_weight=0.),
                                       objective=dict(base_weight=.1, repaired_weight=0.), quality=QUALITY)
    loss.backward()
    groups, aux = optimizer_groups(model, small_config(Path("unused"), "joint"))
    assert {g["name"] for g in groups} == {"codec", "backend", "vae", "pixel_adapter"} and aux
    parameters = [model.codec.router.head[-1].weight, model.text_proj.weight,
                  model.quality_conditioner.block_projection.weight, *model.pixel_adapter.parameters(),
                  *model.codec.precision_embedding.parameters()]
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert result.pixel_features["mu"].requires_grad and result.pixel_features["pixel_tokens"].requires_grad


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_six_qualities_and_independent_receiver_use_original_packets(source, tmp_path):
    model = mu_pixel_model(qk_source=source).eval()
    activate_conditions(model)
    activate_output(model.pixel_adapter)
    receiver = mu_pixel_model(qk_source=source).eval()
    receiver.load_bt_state(model.export_state())
    native = deepcopy(model)
    native.__class__ = BTCodec
    del native.pixel_adapter
    image = torch.randn(1, 3, 233, 377).clamp(-1, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("receiver used source-only information")
    for q in QUALITY["values"]:
        with torch.no_grad():
            expected = model(pad_image(image), RateControl(q)).image[..., :233, :377]
            packet = model.compress(image, RateControl(q))
            assert packet.to_bytes() == native.compress(image, RateControl(q)).to_bytes()
        assert packet.version == 3 and not packet.mask and not packet.scales
        assert len(packet.to_bytes()) == sum(packet.byte_breakdown().values())
        calls = receiver.prediction_calls
        with patch.object(receiver, "encode_clean_latent", side_effect=forbidden), \
             patch.object(receiver.codec.router, "forward", side_effect=forbidden):
            actual = receiver.decompress(packet.to_bytes())
        assert receiver.prediction_calls - calls == 1
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.save(model.export_state(), tmp_path / "state.pt")
    (tmp_path / "packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_mu_pixel_adapter import mu_pixel_model
root = Path(sys.argv[1])
model = mu_pixel_model(qk_source=sys.argv[2]).eval()
model.load_bt_state(torch.load(root / 'state.pt', weights_only=True))
model.codec.update(force=True)
def forbidden(*args, **kwargs):
    raise AssertionError('independent receiver used sender')
model.encode_clean_latent = forbidden
model.codec.router.forward = forbidden
torch.save(model.decompress((root / 'packet.btic').read_bytes()), root / 'decoded.pt')
assert model.prediction_calls == 1
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path), source], cwd=ROOT, timeout=120, check=True)
    torch.testing.assert_close(torch.load(tmp_path / "decoded.pt", weights_only=True), actual, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_ema_strict_modes_and_explicit_old_shared_initialization(source):
    model = mu_pixel_model(qk_source=source)
    state = clone_state(model.export_state())
    receiver = mu_pixel_model(qk_source=source)
    receiver.load_bt_state(state)
    ema = DeploymentEMA(model, decay=.5)
    with torch.no_grad():
        model.pixel_adapter.blocks[0].attn.q.weight.add_(1.)
    ema.update(model)
    torch.testing.assert_close(ema.state["pixel_adapter"]["blocks.0.attn.q.weight"],
                               state["pixel_adapter"]["blocks.0.attn.q.weight"] + .5)
    other = mu_pixel_model(qk_source="pixels" if source == "mu" else "mu")
    for wrong in (other.export_state(), precision_pixel_model().export_state()):
        with pytest.raises(ValueError, match="incompatible mu PiT"):
            receiver.load_bt_state(wrong)
    fresh = deepcopy(receiver.pixel_adapter.state_dict())
    receiver.load_bt_state(precision_pixel_model().export_state(), initialize_pixel_adapter=True)
    same_state(fresh, receiver.pixel_adapter.state_dict())
    del state["pixel_adapter"]
    with pytest.raises(ValueError, match="missing pixel_adapter"):
        receiver.load_bt_state(state)
    receiver.load_bt_state(state, initialize_pixel_adapter=True)


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_condition_diagnosis_fixes_formal_sana_and_ae_features(source):
    from scripts.diagnose_mu_pit import capture_packet_tail, decode_cached_tail
    model = mu_pixel_model(qk_source=source).eval()
    activate_conditions(model)
    activate_output(model.pixel_adapter)
    image = torch.rand(1, 3, 233, 377) * 2 - 1
    with torch.no_grad():
        packet = model.compress(image, RateControl(.2))
        normal, features, mu = capture_packet_tail(model, packet.to_bytes())
        original_features, original_mu = features.clone(), mu.clone()
        calls = model.prediction_calls
        interventions = [mu, torch.zeros_like(mu), mu.roll(1, dims=-1), torch.randn_like(mu)]
        outputs = [decode_cached_tail(model, features, condition, image.shape[-2:]) for condition in interventions]
        bypass = decode_cached_tail(model, features, mu, image.shape[-2:], bypass=True)
    torch.testing.assert_close(normal, outputs[0], atol=0, rtol=0)
    assert model.prediction_calls == calls
    assert torch.equal(features, original_features) and torch.equal(mu, original_mu)
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], bypass)


@pytest.mark.parametrize("mode", ["a", "b"])
def test_exact_trainer_resume_with_complete_state(mode, tmp_path):
    (tmp_path / "images").mkdir()
    Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path / "images/a.png")
    config = small_config(tmp_path, "joint")
    config["model"].update(backend="C", codec=deepcopy(CODEC))
    config["self_flow"] = dict(enabled=False, mixed_sampling=False)
    config["decoder"] = dict(method=MAIN if mode == "b" else CONTROL_METHOD, pixel_adapter=deepcopy(SMALL_PIXEL))
    config["quality"] = deepcopy(QUALITY)
    config["objective"] = dict(base_weight=.1, repaired_weight=0.)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    config["execution"] = dict(saved_tensors="stride_preserving_cpu")
    config["train"].update(batch_size=1, accumulation_steps=1, effective_batch=1, router_ramp_steps=0)
    initial = precision_pixel_model().export_state()  # shared handoff can include the old D
    source = tmp_path / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                             raw=initial, assets={"test": "mu-pit"}, config=deepcopy(config), calibration={}))
    def run(name, resume=None, max_updates=None):
        cfg = deepcopy(config)
        cfg["train"]["output"] = str(tmp_path / name)
        args = SimpleNamespace(resume=str(resume) if resume else None,
                               init_checkpoint=None if resume else str(source), max_updates=max_updates)
        with patch.object(trainer, "make_model", side_effect=mu_pixel_model), \
             patch.object(trainer, "asset_identity", return_value={"test": "mu-pit"}), redirect_stdout(io.StringIO()):
            trainer.train(cfg, args, 0, 1, torch.device("cpu"))
    run("whole")
    run("resumed", max_updates=1)
    paused = read_checkpoint(tmp_path / "resumed/paused.pt")
    assert "pixel_adapter" in paused["ema"] and paused["step"] == 1
    config["execution"]["saved_tensors"] = "device"
    run("resumed", resume=tmp_path / "resumed/paused.pt")
    whole, resumed = [read_checkpoint(tmp_path / name / "complete.pt") for name in ("whole", "resumed")]
    for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "rng_by_rank", "scaler"):
        same_state(whole[key], resumed[key])


def ddp_worker(rank, directory, source):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{directory}/rendezvous",
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(903)
        adapter = small_adapter(source, checkpoint_blocks=True)
        wrapped = DistributedDataParallel(adapter)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        for step in range(2):
            torch.manual_seed(903 + rank + 2 * step)
            optimizer.zero_grad(set_to_none=True)
            wrapped(torch.randn(1, 8, 16, 24), torch.randn(1, 4, 2, 3)).square().mean().backward()
            optimizer.step()
        for parameter in adapter.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(parameter, expected)
            assert parameter.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("source", ["mu", "pixels"])
def test_two_rank_zero_initialized_checkpointed_ddp(source, tmp_path):
    mp.spawn(ddp_worker, args=(str(tmp_path), source), nprocs=2, join=True)
