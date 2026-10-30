"""Receiver-only PiT v2: amplitude, official math, packets and exact recovery."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import os
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
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.pixel_adapter import ReceivedUpdatePixelAdapter, PiTBlock, rotary_position
from models.bt_self_flow.received_update_pixel_model import ReceivedUpdateCPixelCodec, METHOD as PIXEL_METHOD
from models.sana_self_flow import quality_conditioned_sana_forward
from test_bt_training import small_config
from test_c_precision_v3 import CODEC, activate_conditions
from test_pixel_adapter import QUALITY, SMALL_PIXEL, activate_output
from test_precision_pixel_adapter import precision_pixel_model


ROOT = Path(__file__).resolve().parents[1]


def received_update_model(config=None, device="cpu"):
    model = precision_pixel_model()
    model.__class__ = ReceivedUpdateCPixelCodec
    model.pixel_adapter = ReceivedUpdatePixelAdapter(8, 4, **SMALL_PIXEL, checkpoint_blocks=True)
    return model.to(device)


def small_adapter(**kwargs):
    settings = dict(precision_dim=8, pixel_dim=4, patch_size=4, context_dim=16,
                    attention_dim=16, heads=2, checkpoint_blocks=False)
    settings.update(kwargs)
    return ReceivedUpdatePixelAdapter(8, 4, **settings)


def activate_conditions_and_output(adapter):
    activate_output(adapter)
    with torch.no_grad():
        for block in adapter.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=.01)
        nn.init.normal_(adapter.gate_mlp[-1].weight, std=.01)


def test_recipe_preserves_c_and_rejects_old_conditions():
    config = load_config(ROOT / "configs/bt_self_flow/precision_v3/pixel_v2_joint.yaml")
    native = load_config(ROOT / "configs/bt_self_flow/precision_v3/c_joint.yaml")
    for key in ("model", "quality", "objective", "image_loss", "optimizer", "data"):
        assert config[key] == native[key]
    check_stage_model(config, native)
    assert config["decoder"]["method"] == PIXEL_METHOD
    assert config["train"]["steps"] == 200000
    assert (config["train"]["batch_size"], config["train"]["effective_batch"]) == (8, 8)
    dataset = SimpleNamespace(fingerprint="same_data")
    old = load_config(ROOT / "configs/bt_self_flow/precision_v3/pixel_joint.yaml")
    assert training_contract(config, dataset, 1) != training_contract(old, dataset, 1)
    for key in ("delta_ref", "use_quantization"):
        bad = deepcopy(config)
        bad["decoder"]["pixel_adapter"][key] = 1
        with pytest.raises(ValueError, match="unknown pixel_adapter"):
            validate_decoder(bad)
    for section, key, value in (("model", "backend", "D"), ("self_flow", "enabled", True)):
        bad = deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            validate_decoder(bad)
    bad = deepcopy(config)
    bad["model"]["codec"]["precision_consistent"] = False
    with pytest.raises(ValueError, match="frontend"):
        validate_decoder(bad)


def test_real_constructor_factory_preserves_base_rng_and_initialization():
    def initialize_base(self, *args, **kwargs):
        native = precision_pixel_model()
        del native.pixel_adapter
        self.__dict__.update(native.__dict__)
    expected = BTCodec.__new__(BTCodec)
    initialize_base(expected)
    expected_state = clone_state(expected.export_state())
    expected_rng = torch.get_rng_state().clone()
    config = load_config(ROOT / "configs/bt_self_flow/precision_v3/pixel_v2_joint.yaml")
    config["decoder"]["pixel_adapter"] = deepcopy(SMALL_PIXEL)
    with patch.object(BTCodec, "__init__", initialize_base):
        model = make_model(config, torch.device("cpu"))
    assert isinstance(model, ReceivedUpdateCPixelCodec)
    assert isinstance(model.pixel_adapter, ReceivedUpdatePixelAdapter)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for part, values in expected_state.items():
        for name, value in values.items():
            assert torch.equal(model.export_state()[part][name], value), (part, name)
    assert all(block.post_modulation for block in model.pixel_adapter.blocks)


def test_condition_preserves_amplitude_channels_and_precision_cell_boundaries():
    adapter = small_adapter()
    mu, update = torch.randn(2, 4, 2, 3), torch.randn(2, 4, 2, 3)
    embedding = torch.randn(2, 8, 2, 3)
    scale = torch.tensor(2.)
    base, residual, gate = adapter.condition_parts(mu, update, embedding, scale, (4, 6))
    assert torch.equal(gate, torch.ones_like(gate))
    for factor in (0., .5, 2.):
        b, r, g = adapter.condition_parts(mu, factor * update, embedding, scale, (4, 6))
        torch.testing.assert_close(b, base, atol=0, rtol=0)
        torch.testing.assert_close(r, factor * residual, atol=0, rtol=0)
        torch.testing.assert_close(g, gate, atol=0, rtol=0)
    condition = adapter.condition(mu, update, embedding, scale, (4, 6))
    expected = torch.nn.functional.silu(torch.cat((base, residual), 1)).flatten(2).transpose(1, 2)
    torch.testing.assert_close(condition, expected, atol=0, rtol=0)
    activate_conditions_and_output(adapter)
    _, _, gate = adapter.condition_parts(mu, update, embedding, scale, (4, 6))
    assert (gate > .5).all() and (gate < 1.5).all()
    torch.testing.assert_close(gate[..., ::2, ::2], gate[..., 1::2, 1::2], atol=0, rtol=0)
    assert not torch.equal(gate, torch.ones_like(gate))
    with pytest.raises(ValueError, match="whole PiT regions"):
        adapter.condition(mu, update, embedding, scale, (3, 6))
    with pytest.raises(ValueError, match="fixed s_data"):
        adapter.condition(mu, update, embedding, torch.tensor(0.), (4, 6))


def test_zero_writeback_and_gradients_activate_through_normal_updates():
    adapter = small_adapter(checkpoint_blocks=True)
    features = torch.randn(2, 8, 16, 24)
    mu, update = torch.randn(2, 4, 2, 3), torch.randn(2, 4, 2, 3)
    embedding = torch.randn(2, 8, 2, 3)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=0)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        output = adapter(features, mu, update, embedding, torch.tensor(2.))
        if step == 0:
            assert torch.equal(output, features)
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
        optimizer.step()
    for parameter in (adapter.mu_projection.weight, adapter.r_projection.weight,
                      adapter.gate_mlp[0].weight, adapter.gate_mlp[-1].weight,
                      adapter.blocks[0].attn.qkv.weight, adapter.blocks[1].attn.qkv.weight,
                      adapter.output_norm.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bf16", [False, True])
def test_checkpointed_adapter_matches_values_and_input_parameter_gradients(bf16, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    plain = small_adapter().to(device)
    activate_conditions_and_output(plain)
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    inputs = (torch.randn(1, 8, 16, 24, device=device), torch.randn(1, 4, 2, 3, device=device),
              torch.randn(1, 4, 2, 3, device=device), torch.randn(1, 8, 2, 3, device=device))
    results = []
    for adapter in (plain, checked):
        xs = [x.clone().requires_grad_() for x in inputs]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16):
            output = adapter(*xs, torch.tensor(2., device=device))
        output.square().mean().backward()
        results.append((output.detach(), *(x.grad for x in xs)))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert torch.isfinite(left).all()
    for left, right in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_post_modulation_matches_official_forward_and_backward():
    reference = os.environ.get("PIXELDIT_REFERENCE")
    if reference is None:
        pytest.skip("set PIXELDIT_REFERENCE to upstream revision f5ed1f6")
    sys.path.insert(0, reference)
    from pixdit_core.pixeldit_c2i import PiTBlock as OfficialPiT
    torch.manual_seed(903)
    official = OfficialPiT(4, 16, 2, 2, attn_hidden_size=16, adaln_post_modulation=True)
    local = PiTBlock(4, 16, 2, 16, 2, post_modulation=True)
    def rename(name):
        return name.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")
    local.load_state_dict({rename(k): v for k, v in official.state_dict().items()}, strict=True)
    pixels, condition = torch.randn(2, 6, 4, 4), torch.randn(2, 6, 16)
    x1, x2 = [pixels.clone().requires_grad_() for _ in range(2)]
    c1, c2 = [condition.clone().requires_grad_() for _ in range(2)]
    expected = official(x1.reshape(12, 4, 4), c1.reshape(12, 16), 4, 6, 2).reshape_as(pixels)
    actual = local(x2, c2, rotary_position(8, 2, 3, "cpu"))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in ((x1.grad, x2.grad), (c1.grad, c2.grad)):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
    params = dict(local.named_parameters())
    for name, parameter in official.named_parameters():
        torch.testing.assert_close(params[rename(name)].grad, parameter.grad, atol=1e-6, rtol=1e-6)


def test_zero_pit_matches_native_c_one_prediction_and_actual_scheduler_update():
    model = received_update_model().eval()
    activate_conditions(model)
    mu = torch.randn(1, 4, 8, 12)
    delta = torch.exp(torch.randn(1, 1, 4, 6))
    with torch.no_grad():
        expected_latent, expected_image, _ = BTCodec.repair(model, mu, precision=delta)
        before = model.prediction_calls
        with patch("models.bt_self_flow.model.quality_conditioned_sana_forward",
                   wraps=quality_conditioned_sana_forward) as call:
            output = model.decode_received(mu, delta, return_features=True)
    assert call.call_count == 1 and call.call_args.kwargs.get("capture_layer") is None
    assert model.prediction_calls - before == 1
    assert "sana_tokens" not in output
    torch.testing.assert_close(output["latent"], expected_latent, atol=0, rtol=0)
    torch.testing.assert_close(output["update"], expected_latent.float() - mu.float(), atol=0, rtol=0)
    torch.testing.assert_close(output["image"], expected_image, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bf16", [False, True])
def test_full_decoder_checkpoint_gradient_parity_and_shared_embedding_ownership(bf16):
    plain = received_update_model().train()
    activate_conditions(plain)
    activate_conditions_and_output(plain.pixel_adapter)
    plain.checkpoint_blocks = False
    plain.pixel_adapter.checkpoint_blocks = False
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    inputs = (torch.randn(1, 4, 8, 12), torch.exp(torch.randn(1, 1, 4, 6)))
    results = []
    for model in (plain, checked):
        mu, delta = [value.clone().requires_grad_() for value in inputs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            output = model.decode_received(mu, delta, return_features=True)
        output["image"].square().mean().backward()
        results.append((output["image"].detach(), mu.grad, delta.grad))
        assert output["update"].requires_grad
        assert not any("precision_embedding" in name for name in model.pixel_adapter.state_dict())
        assert not any("s_data" in name for name in model.pixel_adapter.state_dict())
        groups, aux = optimizer_groups(model, small_config(Path("unused"), "joint"))
        assert groups and aux
        parameters = [model.pixel_adapter.mu_projection.weight, model.pixel_adapter.r_projection.weight,
                      model.pixel_adapter.gate_mlp[0].weight,
                      model.quality_conditioner.block_projection.weight,
                      *model.codec.precision_embedding.parameters()]
        for parameter in parameters:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    tolerance = dict(atol=2e-4, rtol=2e-3) if bf16 else dict(atol=1e-6, rtol=1e-6)
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, **tolerance)
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, **tolerance, msg=name)


def test_six_quality_packets_match_c_bytes_and_decode_without_sender(tmp_path):
    model = received_update_model().eval()
    activate_conditions(model)
    activate_conditions_and_output(model.pixel_adapter)
    native = deepcopy(model)
    native.__class__ = BTCodec
    del native.pixel_adapter
    receiver = received_update_model().eval()
    receiver.load_bt_state(model.export_state())
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
        before = receiver.prediction_calls
        with patch.object(receiver, "encode_clean_latent", side_effect=forbidden), \
             patch.object(receiver.codec.router, "forward", side_effect=forbidden):
            actual = receiver.decompress(packet.to_bytes())
        assert receiver.prediction_calls - before == 1
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.save(model.export_state(), tmp_path / "state.pt")
    (tmp_path / "packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_received_update_pixel_adapter import received_update_model
root = Path(sys.argv[1])
model = received_update_model().eval()
model.load_bt_state(torch.load(root / 'state.pt', weights_only=True))
model.codec.update(force=True)
def forbidden(*args, **kwargs):
    raise AssertionError('independent receiver used sender')
model.encode_clean_latent = forbidden
model.codec.router.forward = forbidden
torch.save(model.decompress((root / 'packet.btic').read_bytes()), root / 'decoded.pt')
assert model.prediction_calls == 1
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT, timeout=120, check=True)
    torch.testing.assert_close(torch.load(tmp_path / "decoded.pt", weights_only=True), actual, atol=2e-5, rtol=2e-5)


def test_ema_strict_version_load_and_shared_c_handoff():
    model = received_update_model()
    state = clone_state(model.export_state())
    receiver = received_update_model()
    receiver.load_bt_state(state)
    ema = DeploymentEMA(model, decay=.5)
    with torch.no_grad():
        model.pixel_adapter.r_projection.weight.add_(1.)
    ema.update(model)
    torch.testing.assert_close(ema.state["pixel_adapter"]["r_projection.weight"],
                               state["pixel_adapter"]["r_projection.weight"] + .5)
    with pytest.raises(RuntimeError):
        receiver.load_bt_state(precision_pixel_model().export_state())
    initial_adapter = deepcopy(receiver.pixel_adapter.state_dict())
    receiver.load_bt_state(precision_pixel_model().export_state(), initialize_pixel_adapter=True)
    same_state(initial_adapter, receiver.pixel_adapter.state_dict())
    del state["pixel_adapter"]
    with pytest.raises(ValueError, match="missing pixel_adapter"):
        receiver.load_bt_state(state)
    receiver.load_bt_state(state, initialize_pixel_adapter=True)
    assert torch.count_nonzero(receiver.pixel_adapter.pixel_output.weight) == 0


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_state(a, b)
    else:
        assert left == right


def test_exact_trainer_resume_preserves_all_state(tmp_path):
    (tmp_path / "images").mkdir()
    Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path / "images/a.png")
    config = small_config(tmp_path, "joint")
    config["model"].update(backend="C", codec=deepcopy(CODEC))
    config["self_flow"] = dict(enabled=False, mixed_sampling=False)
    config["decoder"] = dict(method=PIXEL_METHOD, pixel_adapter=deepcopy(SMALL_PIXEL))
    config["quality"] = deepcopy(QUALITY)
    config["objective"] = dict(base_weight=.1, repaired_weight=0.)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    config["execution"] = dict(saved_tensors="stride_preserving_cpu")
    config["train"].update(batch_size=1, accumulation_steps=1, effective_batch=1, router_ramp_steps=0)
    initial = received_update_model().export_state()
    del initial["pixel_adapter"]
    source = tmp_path / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                             raw=initial, assets={"test": "received-update-pixel"}, config=deepcopy(config), calibration={}))
    def run(name, resume=None, max_updates=None):
        cfg = deepcopy(config)
        cfg["train"]["output"] = str(tmp_path / name)
        args = SimpleNamespace(resume=str(resume) if resume else None,
                               init_checkpoint=None if resume else str(source), max_updates=max_updates)
        with patch.object(trainer, "make_model", side_effect=received_update_model), \
             patch.object(trainer, "asset_identity", return_value={"test": "received-update-pixel"}), redirect_stdout(io.StringIO()):
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


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{directory}/rendezvous",
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(903)
        adapter = small_adapter(checkpoint_blocks=True)
        wrapped = DistributedDataParallel(adapter)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        for step in range(4):
            torch.manual_seed(903 + rank + 2 * step)
            optimizer.zero_grad(set_to_none=True)
            wrapped(torch.randn(1, 8, 16, 24), torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3),
                    torch.randn(1, 8, 2, 3), torch.tensor(2.)).square().mean().backward()
            optimizer.step()
        for parameter in adapter.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(parameter, expected)
        assert adapter.gate_mlp[0].weight.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpointed_zero_initialization_has_no_unused_parameters(tmp_path):
    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
