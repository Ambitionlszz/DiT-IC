"""B + additive precision AdaLN: physical input, affine math and deployment."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

import test_mu_pixel_adapter as base
from models.bt_self_flow.checkpoint import DeploymentEMA, clone_state
from models.bt_self_flow.experiment import load_config, make_model, check_stage_model
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.mu_pixel_model import PrecisionMuQKCPixelCodec, PRECISION_METHOD
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.pixel_adapter import PrecisionMuPixelAdapter, rotary_position
from test_pixel_adapter import activate_output, SMALL_PIXEL, QUALITY


ROOT = Path(__file__).resolve().parents[1]
_old_model = base.mu_pixel_model


def precision_model(config=None, device="cpu", **kwargs):
    model = _old_model(qk_source="mu")
    model.__class__ = PrecisionMuQKCPixelCodec
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(903)
        model.pixel_adapter = PrecisionMuPixelAdapter(8, 4, **SMALL_PIXEL, checkpoint_blocks=True)
    return model.to(device)


def adapter(**kwargs):
    settings = dict(pixel_dim=4, patch_size=4, context_dim=16, attention_dim=16, heads=2,
                    checkpoint_blocks=False)
    settings.update(kwargs)
    return PrecisionMuPixelAdapter(8, 4, **settings)


def recipe():
    return load_config(ROOT / "configs/bt_self_flow/precision_v3/mu_pit_b_precision_joint.yaml")


def test_recipe_and_factory_keep_native_c_contract():
    config, old = recipe(), base.recipe("b")
    assert config["decoder"]["method"] == PRECISION_METHOD
    config["decoder"]["method"] = old["decoder"]["method"]
    config["train"]["output"] = old["train"]["output"]
    assert config == old
    config = recipe()
    native = load_config(ROOT / "configs/bt_self_flow/precision_v3/c_joint.yaml")
    check_stage_model(config, native)
    for key in ("model", "data", "quality", "objective", "image_loss", "optimizer", "self_flow"):
        assert config[key] == native[key]
    def initialize(self, *args, **kwargs):
        model = _old_model()
        del model.pixel_adapter
        self.__dict__.update(model.__dict__)
    config["decoder"]["pixel_adapter"] = deepcopy(SMALL_PIXEL)
    with patch.object(BTCodec, "__init__", initialize):
        model = make_model(config, torch.device("cpu"))
    assert isinstance(model, PrecisionMuQKCPixelCodec)
    assert all(block.precision_modulation.in_features == 128 for block in model.pixel_adapter.blocks)


def test_zero_precision_heads_preserve_all_seeded_b_weights_and_nonzero_output():
    torch.manual_seed(903)
    old = base.small_adapter()
    torch.manual_seed(903)
    new = adapter()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    for block in new.blocks:
        assert block.precision_modulation.out_features == 16
        assert not block.precision_modulation.weight.count_nonzero()
        assert not block.precision_modulation.bias.count_nonzero()
    activate_output(old)
    new.pixel_output.load_state_dict(old.pixel_output.state_dict())
    features, mu = torch.randn(1, 8, 16, 24), torch.randn(1, 4, 2, 3)
    precision = torch.randn(1, 128, 2, 3)
    torch.testing.assert_close(new(features, mu, precision), old(features, mu), atol=0, rtol=0)


def test_precision_affine_math_broadcasting_and_unchanged_qk_and_gates():
    model = adapter()
    block = model.blocks[0]
    nn.init.normal_(block.precision_modulation.weight, std=.03)
    nn.init.normal_(block.precision_modulation.bias, std=.03)
    pixels, content, precision = torch.randn(1, 6, 16, 4), torch.randn(1, 6, 16), torch.randn(1, 6, 128)
    position = rotary_position(8, 2, 3, "cpu")
    sa, ca, ga, sm, cm, gm = block.adaLN_modulation(content).reshape(1, 6, 16, 24).chunk(6, -1)
    spa, cpa, spm, cpm = block.precision_modulation(precision).unsqueeze(2).chunk(4, -1)
    compact = block.compress_to_attn((block.norm1(pixels) * (1 + ca + cpa) + sa + spa).flatten(-2))
    updated = pixels + ga * block.expand_from_attn(block.attn(compact, content, position)).reshape_as(pixels)
    expected = updated + gm * block.mlp(block.norm2(updated) * (1 + cm + cpm) + sm + spm)
    received = []
    handle = block.attn.q.register_forward_pre_hook(lambda module, args: received.append(args[0]))
    actual = block(pixels, (content, precision), position)
    handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(received[0], content)
    field = torch.randn(1, 128, 2, 3)
    aligned = model.precision_condition(field, (4, 6)).transpose(1, 2).reshape(1, 128, 4, 6)
    assert torch.equal(aligned, field.repeat_interleave(2, -2).repeat_interleave(2, -1))


def test_normal_updates_open_head_and_shared_embedding_gradients():
    model = adapter(checkpoint_blocks=True)
    embedding = nn.Conv2d(1, 128, 1)
    optimizer = torch.optim.AdamW([*model.parameters(), *embedding.parameters()], lr=1e-3)
    x, mu, delta = torch.randn(1, 8, 16, 24), torch.randn(1, 4, 2, 3), torch.randn(1, 1, 2, 3)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = model(x, mu, embedding(delta))
        if step == 0:
            assert torch.equal(result, x)
        result.square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step >= 1:
                assert parameter.grad.abs().sum() > 0, name
        if step == 2:
            assert embedding.weight.grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bf16", [False, True])
@torch.backends.cudnn.flags(benchmark=False, deterministic=True)
def test_precision_tuple_checkpoint_forward_and_all_gradients(device, bf16):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(903)
    plain = adapter().to(device)
    activate_output(plain)
    for block in plain.blocks:
        nn.init.normal_(block.precision_modulation.weight, std=.02)
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    inputs = [torch.randn(*shape, device=device) for shape in ((1, 8, 16, 24), (1, 4, 2, 3), (1, 128, 2, 3))]
    results = []
    for model in (plain, checked):
        values = [x.clone().requires_grad_() for x in inputs]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            output = model(*values)
        output.square().mean().backward()
        results.append((output.detach(), *[x.grad for x in values]))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    tolerance = dict(atol=1e-8, rtol=1e-4) if device == "cuda" else dict(atol=0, rtol=0)
    for a, b in zip(results[0][1:], results[1][1:]):
        torch.testing.assert_close(a, b, **tolerance)
    for a, b in zip(plain.parameters(), checked.parameters()):
        assert torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, **tolerance)


def test_original_precision_input_and_native_c_initial_parity():
    model = precision_model().eval()
    mu, delta = torch.randn(1, 4, 8, 12), torch.exp(torch.randn(1, 1, 4, 6))
    with torch.no_grad():
        _, native, _ = BTCodec.repair(model, mu, precision=delta)
        calls = model.prediction_calls
        result = model.decode_received(mu, delta, return_features=True)
        expected = model.codec.precision_embedding(delta.log())
    assert model.prediction_calls - calls == 1
    torch.testing.assert_close(result["image"], native, atol=1e-6, rtol=1e-6)
    assert torch.equal(result["precision_features"], expected)


@pytest.mark.parametrize("bf16", [False, True])
def test_full_decoder_recomputation_with_precision_and_joint_loss(bf16):
    with patch.object(base, "mu_pixel_model", precision_model):
        base.test_whole_decoder_recompute_and_end_to_end_condition_gradients("mu", bf16)
        if not bf16:
            base.test_joint_image_loss_keeps_existing_c_and_d_gradients("mu")


def test_real_packet_and_fresh_process_receiver(tmp_path):
    model = precision_model().eval()
    base.activate_conditions(model)
    activate_output(model.pixel_adapter)
    for block in model.pixel_adapter.blocks:
        nn.init.normal_(block.precision_modulation.weight, std=.02)
    receiver = precision_model().eval()
    receiver.load_bt_state(model.export_state())
    native = deepcopy(model)
    native.__class__ = BTCodec
    del native.pixel_adapter
    image = torch.rand(1, 3, 233, 377) * 2 - 1
    for quality in QUALITY["values"]:
        with torch.no_grad():
            packet = model.compress(image, RateControl(quality))
            assert packet.to_bytes() == native.compress(image, RateControl(quality)).to_bytes()
            expected = model(pad_image(image), RateControl(quality)).image[..., :233, :377]
            calls = receiver.prediction_calls
            with patch.object(receiver, "encode_clean_latent", side_effect=AssertionError("sender access")), \
                 patch.object(receiver.codec.router, "forward", side_effect=AssertionError("sender access")):
                actual = receiver.decompress(packet.to_bytes())
        assert receiver.prediction_calls - calls == 1
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.save(model.export_state(), tmp_path / "state.pt")
    (tmp_path / "packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_precision_mu_pixel_adapter import precision_model
root=Path(sys.argv[1]); model=precision_model().eval()
model.load_bt_state(torch.load(root/'state.pt',weights_only=True))
model.codec.update(force=True)
torch.save(model.decompress((root/'packet.btic').read_bytes()),root/'decoded.pt')
assert model.prediction_calls==1
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT, check=True, timeout=120)
    torch.testing.assert_close(torch.load(tmp_path / "decoded.pt", weights_only=True), actual, atol=2e-5, rtol=2e-5)


def test_strict_version_ema_and_fresh_shared_initialization():
    model = precision_model()
    old = _old_model()
    with pytest.raises(ValueError, match="precision AdaLN"):
        model.load_bt_state(old.export_state())
    original = clone_state(model.export_state())
    model.load_bt_state(old.export_state(), initialize_pixel_adapter=True)
    base.same_state(original["pixel_adapter"], model.pixel_adapter.state_dict())
    ema = DeploymentEMA(model, decay=.5)
    with torch.no_grad(): model.pixel_adapter.blocks[0].precision_modulation.weight.fill_(1.)
    ema.update(model)
    assert torch.equal(ema.state["pixel_adapter"]["blocks.0.precision_modulation.weight"],
                       torch.full_like(model.pixel_adapter.blocks[0].precision_modulation.weight, .5))
    precision_model().load_bt_state(ema.state)


def test_exact_training_resume_including_precision_heads(tmp_path):
    with patch.object(base, "mu_pixel_model", precision_model), patch.object(base, "MAIN", PRECISION_METHOD):
        base.test_exact_trainer_resume_with_complete_state("b", tmp_path)


def test_cached_interventions_keep_actual_precision_and_sana_fixed():
    from scripts.diagnose_mu_pit import capture_packet_tail, decode_cached_tail
    model = precision_model().eval()
    image = torch.rand(1, 3, 233, 377) * 2 - 1
    with torch.no_grad():
        packet = model.compress(image, RateControl(.4))
        normal, features, mu, precision = capture_packet_tail(model, packet, return_precision=True)
        calls = model.prediction_calls
        replay = decode_cached_tail(model, features, mu, image.shape[-2:], precision_features=precision)
        decode_cached_tail(model, features, torch.zeros_like(mu), image.shape[-2:], precision_features=precision)
    assert model.prediction_calls == calls
    assert torch.equal(replay, normal)
