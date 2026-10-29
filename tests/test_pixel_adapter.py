"""PiT math, real-library codec integration, deployment and continuation."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
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
from diffusers import AutoencoderDC
from peft import LoraConfig, get_peft_model

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.decoder import decode_latent, decode_latent_with_pixel_adapter
from models.bt_self_flow.experiment import load_config, optimizer_groups, validate_decoder
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.pixel_adapter import PixelDecoderAdapter, PiTBlock, pixel_position, rotary_position
from models.bt_self_flow.pixel_model import CPixelAdapterCodec, METHOD as PIXEL_METHOD, native_sana_features
from models.sana_self_flow import unwrap_sana_transformer
from test_bt_backend import tiny_model, tiny_sana
from test_bt_training import small_config


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/bt_self_flow/pixel_adapter"
CODEC = dict(ch_emd=4, channel=16, channel_out=4, precision_conditioning=True, rate_likelihood="noise")
QUALITY = dict(values=[0., .2, .4, .6, .8, 1.], rate_coefficients=[16., 8., 4., 2., 1., .5])
SMALL_PIXEL = dict(depth=2, pixel_dim=4, patch_size=16, context_dim=16, attention_dim=16, heads=2)


def small_adapter(**kwargs):
    settings = dict(depth=2, pixel_dim=4, patch_size=4, context_dim=16, attention_dim=16,
                    heads=2, checkpoint_blocks=False)
    settings.update(kwargs)
    return PixelDecoderAdapter(8, 32, **settings)


def real_small_ae():
    channels, layers, kinds, scales = (8,) * 6, (1,) * 6, ("ResBlock",) * 6, ((),) * 6
    ae = AutoencoderDC(latent_channels=4, encoder_block_out_channels=channels, decoder_block_out_channels=channels,
                       encoder_layers_per_block=layers, decoder_layers_per_block=layers, attention_head_dim=4,
                       encoder_block_types=kinds, decoder_block_types=kinds,
                       encoder_qkv_multiscales=scales, decoder_qkv_multiscales=scales)
    ae = get_peft_model(ae.requires_grad_(False), LoraConfig(r=2, lora_alpha=2,
                                                          target_modules=["conv_out"], bias="none"))
    for name, parameter in ae.named_parameters():
        parameter.requires_grad_("decoder." in name and "lora_" in name)
    return ae


def pixel_model(config=None, device="cpu"):
    model = tiny_model("C")
    model.__class__ = CPixelAdapterCodec
    torch.manual_seed(903)
    model.codec = BTLatentCodec(**CODEC)
    model.vae = real_small_ae()
    model.pixel_adapter = PixelDecoderAdapter(8, 32, **SMALL_PIXEL, checkpoint_blocks=True)
    model.pixel_seed, model.self_flow_enabled = 903, False
    model.codec.update(force=True)
    return model.to(device)


def activate_output(adapter):
    with torch.no_grad():
        nn.init.normal_(adapter.pixel_output.weight, std=.01)


def test_two_block_recipe_and_matched_c_contract():
    warm, native, full, control = [load_config(RECIPE / f"{name}.yaml")
                                   for name in ("common", "native", "joint", "content_only")]
    assert warm["train"]["steps"] == 50000 and full["train"]["steps"] == 200000
    for cfg in (warm, native, full, control):
        assert (cfg["train"]["batch_size"], cfg["train"]["accumulation_steps"], cfg["train"]["effective_batch"]) == (8, 1, 8)
        assert cfg["model"]["backend"] == "C" and not cfg["self_flow"]["enabled"]
        assert cfg["quality"] == QUALITY
    for key in ("model", "quality", "objective", "image_loss", "optimizer", "data"):
        assert native[key] == full[key] == control[key]
    assert full["decoder"]["pixel_adapter"]["depth"] == 2
    control["train"]["output"] = full["train"]["output"]
    control["decoder"]["pixel_adapter"]["use_quantization"] = True
    assert control == full


@pytest.mark.parametrize("mutation", ["D", "self_flow", "unknown", "heads", "v3", "phase"])
def test_new_method_does_not_relax_legacy_validation(mutation):
    config = load_config(RECIPE / "joint.yaml")
    if mutation == "D":
        config["model"]["backend"] = "D"
    elif mutation == "self_flow":
        config["self_flow"]["enabled"] = True
    elif mutation == "unknown":
        config["decoder"]["pixel_adapter"]["typo"] = 1
    elif mutation == "heads":
        config["decoder"]["pixel_adapter"]["heads"] = 3
    elif mutation == "v3":
        config["model"]["codec"]["precision_consistent"] = True
    else:
        config["train"]["phase"] = "backend"
    with pytest.raises(ValueError):
        validate_decoder(config)


def test_official_pit_forward_backward_and_position_parity():
    reference = os.environ.get("PIXELDIT_REFERENCE")
    if reference is None:
        pytest.skip("set PIXELDIT_REFERENCE to the pinned official repository")
    import sys
    sys.path.insert(0, reference)
    from pixdit_core.pixeldit_c2i import PiTBlock as OfficialPiT
    from pixdit_core.modules import get_2d_sincos_pos_embed_from_grid, precompute_freqs_cis_2d
    torch.manual_seed(1029)
    official = OfficialPiT(4, 16, 2, 2, attn_hidden_size=16)
    ours = PiTBlock(4, 16, 2, 16, 2)
    def rename(name):
        return name.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")
    ours.load_state_dict({rename(k): v for k, v in official.state_dict().items()}, strict=True)
    a, c = torch.randn(2, 6, 4, 4), torch.randn(2, 6, 16)
    x1, x2, c1, c2 = [value.clone().requires_grad_() for value in (a, a, c, c)]
    expected = official(x1.reshape(12, 4, 4), c1.reshape(12, 16), 4, 6, 2).reshape_as(a)
    actual = ours(x2, c2, rotary_position(8, 2, 3, "cpu"))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward(); actual.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(c1.grad, c2.grad, atol=1e-6, rtol=1e-6)
    ours_parameters = dict(ours.named_parameters())
    for name, parameter in official.named_parameters():
        torch.testing.assert_close(ours_parameters[rename(name)].grad, parameter.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rotary_position(8, 2, 3, "cpu"), precompute_freqs_cis_2d(8, 2, 3))
    grid = np.stack(np.meshgrid(np.arange(6, dtype=np.float32), np.arange(4, dtype=np.float32)))
    expected_pos = torch.from_numpy(get_2d_sincos_pos_embed_from_grid(4, grid)).reshape(4, 6, 4).float()
    torch.testing.assert_close(pixel_position(4, 6, 4, "cpu", torch.float32), expected_pos)


def test_native_sana_output_and_checkpoint_gradients():
    torch.manual_seed(903)
    sana = tiny_sana()
    coarse, text, time = torch.randn(2, 4, 8, 12), torch.randn(2, 2, 16), torch.tensor([700., 700.])
    expected = sana(coarse, text, time, return_dict=False)[0]
    results = []
    for checkpointed in (False, True):
        sana.zero_grad(set_to_none=True)
        source = coarse.clone().requires_grad_()
        prediction, feature, hw = native_sana_features(sana, source, text, time, checkpointed)
        torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)
        assert feature.shape == (2, 96, 32) and hw == (8, 12) and feature.requires_grad
        (prediction.square().mean() + .01 * feature.square().mean()).backward()
        results.append((prediction.detach(), source.grad.clone(), sana.proj_out.weight.grad.clone()))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


def test_pixel_initialization_preserves_native_parameters_trainability_and_rng():
    def initialize_base(self, *args, **kwargs):
        native = tiny_model("C")
        self.__dict__.update(native.__dict__)
        self.codec = BTLatentCodec(**CODEC)
        self.vae = real_small_ae()
    expected = BTCodec.__new__(BTCodec)
    initialize_base(expected)
    expected_state = clone_state(expected.export_state())
    expected_trainable = {name for name, value in expected.named_parameters() if value.requires_grad}
    expected_rng = torch.get_rng_state().clone()
    with patch.object(BTCodec, "__init__", initialize_base):
        actual = CPixelAdapterCodec("unused", "unused", pixel_config={**SMALL_PIXEL, "seed": 903})
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for part, values in expected_state.items():
        for name, value in values.items():
            assert torch.equal(value, actual.export_state()[part][name]), (part, name)
    actual_trainable = {name for name, value in actual.named_parameters()
                        if value.requires_grad and not name.startswith("pixel_adapter.")}
    assert actual_trainable == expected_trainable
    assert len(actual.pixel_adapter.blocks) == 2 and actual.self_flow_enabled is False


def test_identity_then_both_blocks_and_conditions_receive_gradients():
    torch.manual_seed(903)
    adapter = small_adapter()
    features = torch.randn(2, 8, 16, 24)
    tokens = torch.randn(2, 6, 32, requires_grad=True)
    delta = torch.full((2, 1, 2, 3), 1.5, requires_grad=True)
    first = adapter(features, tokens, (2, 3), delta)
    assert torch.equal(first, features)
    first.square().mean().backward()
    assert adapter.pixel_output.weight.grad.abs().sum() > 0
    assert adapter.blocks[0].attn.qkv.weight.grad.abs().sum() == 0
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    output, captured = adapter(features, tokens, (2, 3), delta, return_features=True)
    output.square().mean().backward()
    for parameter in (adapter.blocks[0].attn.qkv.weight, adapter.blocks[1].attn.qkv.weight,
                      adapter.content_projection.weight, adapter.scale_embedding.mlp[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert tokens.grad.abs().sum() > 0 and delta.grad.abs().sum() > 0
    assert captured["pixel_tokens"].shape == (2, 24, 16, 4)
    assert captured["pixel_tokens"].requires_grad and captured["token_hw"] == (4, 6)
    assert adapter.blocks[0].attn.qkv.weight is not adapter.blocks[1].attn.qkv.weight


def test_scale_condition_and_capacity_matched_constant_control():
    torch.manual_seed(903)
    full, control = small_adapter(), small_adapter(use_quantization=False)
    control.load_state_dict(full.state_dict())
    activate_output(full); control.pixel_output.load_state_dict(full.pixel_output.state_dict())
    features, tokens = torch.randn(1, 8, 16, 24), torch.randn(1, 6, 32)
    one, two = torch.ones(1, 1, 2, 3), torch.full((1, 1, 2, 3), 2.)
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in control.parameters())
    assert torch.equal(control(features, tokens, (2, 3), one), control(features, tokens, (2, 3), two))
    assert not torch.allclose(full(features, tokens, (2, 3), one), full(features, tokens, (2, 3), two), atol=1e-7)
    with pytest.raises(ValueError, match="whole PiT regions"):
        full(features, tokens, (2, 3), torch.ones(1, 1, 3, 3))


def test_actual_step_is_repeated_by_coding_cell_without_spatial_averaging():
    adapter = small_adapter(delta_ref=2.)
    tokens = torch.randn(1, 6, 32)
    delta = torch.tensor([[[[.5, 1., 2.], [4., 8., 16.]]]], requires_grad=True)
    received = []
    original = adapter.scale_embedding.forward
    def record(value):
        received.append(value.detach())
        return original(value)
    with patch.object(adapter.scale_embedding, "forward", side_effect=record):
        condition = adapter.condition(tokens, (2, 3), delta, (4, 6))
    expected = torch.nn.functional.interpolate(torch.log2(delta.detach() / 2.), size=(4, 6), mode="nearest")
    assert torch.equal(received[0], expected.flatten(1))
    condition.square().mean().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_adapter_checkpointing_matches_output_and_gradients():
    plain = small_adapter(); activate_output(plain)
    checked = deepcopy(plain); checked.checkpoint_blocks = True
    source, tokens, delta = torch.randn(1, 8, 16, 24), torch.randn(1, 6, 32), torch.ones(1, 1, 2, 3)
    output = []
    for adapter in (plain, checked):
        x = source.clone().requires_grad_()
        y = adapter(x, tokens, (2, 3), delta)
        y.square().mean().backward()
        output.append((y.detach(), x.grad.clone()))
    for a, b in zip(*output):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    for a, b in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)


def test_cpu_bf16_pixel_attention_and_two_updates():
    adapter = small_adapter(checkpoint_blocks=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    features, tokens, delta = torch.randn(1, 8, 16, 24), torch.randn(1, 6, 32), torch.ones(1, 1, 2, 3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = adapter(features, tokens, (2, 3), delta)
            if step == 0:
                assert torch.equal(output, features)
            loss = output.square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
        optimizer.step()
    assert adapter.blocks[1].attn.qkv.weight.grad.abs().sum() > 0


def pixel_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{directory}/rendezvous",
                            timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(903)
        adapter = small_adapter(checkpoint_blocks=True)
        wrapped = DistributedDataParallel(adapter)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        for step in range(2):
            torch.manual_seed(903 + rank + step * 2)
            optimizer.zero_grad(set_to_none=True)
            features, tokens = torch.randn(1, 8, 16, 24), torch.randn(1, 6, 32)
            wrapped(features, tokens, (2, 3), torch.ones(1, 1, 2, 3)).square().mean().backward()
            optimizer.step()
        for parameter in adapter.parameters():
            expected = parameter.detach().clone()
            dist.broadcast(expected, 0)
            assert torch.equal(parameter, expected)
        assert adapter.blocks[1].attn.qkv.weight.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpointed_zero_initialization_has_no_unused_parameters(tmp_path):
    mp.spawn(pixel_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_real_ae_zero_update_preserves_native_head_and_peft_names():
    model = pixel_model()
    latent, tokens, delta = torch.randn(1, 4, 8, 16), torch.randn(1, 128, 32), torch.ones(1, 1, 4, 8)
    names = tuple(model.vae.state_dict())
    expected = decode_latent(model.vae, latent)
    for checkpointed in (False, True):
        actual, features = decode_latent_with_pixel_adapter(model.vae, latent, model.pixel_adapter,
                                                           tokens, (8, 16), delta, checkpointed, True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert actual.shape == (1, 3, 256, 512)
        assert features["pixel_tokens"].requires_grad
    assert tuple(model.vae.state_dict()) == names


def test_real_packet_all_qualities_crop_no_sender_and_one_prediction():
    model = pixel_model().eval()
    activate_output(model.pixel_adapter)
    model.codec.update(force=True)
    native = deepcopy(model)
    native.__class__ = BTCodec
    del native.pixel_adapter
    image = torch.rand(1, 3, 233, 400) * 2 - 1
    for q in QUALITY["values"]:
        control = RateControl(q)
        with torch.no_grad():
            packet = model.compress(image, control)
            assert packet.to_bytes() == native.compress(image, control).to_bytes()
            assert not packet.scales and not packet.mask
            expected = model(pad_image(image), control).image[..., :233, :400]
            with patch.object(model, "encode_clean_latent", side_effect=AssertionError("sender latent")), \
                 patch.object(model.codec.router, "forward", side_effect=AssertionError("sender router")):
                calls = model.prediction_calls
                actual = model.decompress(packet.to_bytes())
            assert model.prediction_calls - calls == 1
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


def test_independent_process_decodes_pixel_checkpoint_without_source(tmp_path):
    model = pixel_model().eval()
    activate_output(model.pixel_adapter)
    image = torch.rand(1, 3, 217, 379) * 2 - 1
    with torch.no_grad():
        packet = model.compress(image, RateControl(.6))
        expected = model.decompress(packet.to_bytes())
    torch.save(model.export_state(), tmp_path / "state.pt")
    (tmp_path / "packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_pixel_adapter import pixel_model
root = Path(sys.argv[1])
model = pixel_model().eval()
model.load_bt_state(torch.load(root / 'state.pt', weights_only=True))
model.codec.update(force=True)
def forbidden(*args, **kwargs):
    raise AssertionError('independent decoder used sender-only input')
model.encode_clean_latent = forbidden
model.codec.router.forward = forbidden
torch.save(model.decompress((root / 'packet.btic').read_bytes()), root / 'decoded.pt')
assert model.prediction_calls == 1
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=120, cwd=ROOT)
    actual = torch.load(tmp_path / "decoded.pt", weights_only=True)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)


def test_image_loss_updates_existing_c_and_pixel_parameters():
    model = pixel_model().train(); activate_output(model.pixel_adapter)
    image = torch.rand(1, 3, 256, 256) * 2 - 1
    result = model(image, RateControl(.4), capture=True)
    loss, _ = reconstruction_objective(result, image, RateControl(.4), model.distortion_scale.s_data,
                                       ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0),
                                       objective=dict(base_weight=.1, repaired_weight=0), quality=QUALITY)
    loss.backward()
    parameters = dict(model.named_parameters())
    checks = [model.codec.router.head[-1].weight, model.codec.analysis_precision.projection.weight,
              model.codec.synthesis_precision.projection.weight, model.pixel_adapter.content_projection.weight,
              model.pixel_adapter.scale_embedding.mlp[0].weight,
              model.pixel_adapter.blocks[0].attn.qkv.weight, model.pixel_adapter.blocks[1].attn.qkv.weight,
              model.text_proj.weight]
    checks += [p for name, p in parameters.items() if "lora_B" in name and p.requires_grad]
    for parameter in checks:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.vae.encoder.parameters())
    assert result.pixel_features["pixel_tokens"].requires_grad and result.pixel_features["sana_tokens"].requires_grad


def test_optimizer_ema_strict_restore_and_shared_initialization():
    model = pixel_model()
    config = load_config(RECIPE / "joint.yaml")
    groups, aux = optimizer_groups(model, config)
    ids = {id(p) for group in groups for p in group["params"]}
    assert ids | {id(p) for p in aux} == {id(p) for p in model.parameters() if p.requires_grad}
    assert next(group for group in groups if group["name"] == "pixel_adapter")["lr"] == config["optimizer"]["backend_lr"]
    ema = DeploymentEMA(model, decay=.5)
    before = ema.state["pixel_adapter"]["pixel_output.weight"].clone()
    with torch.no_grad():
        model.pixel_adapter.pixel_output.weight.add_(1)
    ema.update(model)
    assert torch.equal(ema.state["pixel_adapter"]["pixel_output.weight"], before + .5)
    state = clone_state(model.export_state())
    restored = pixel_model(); restored.load_bt_state(state)
    for name, value in model.pixel_adapter.state_dict().items():
        assert torch.equal(value, restored.pixel_adapter.state_dict()[name])
    del state["pixel_adapter"]
    with pytest.raises(ValueError, match="missing pixel_adapter"):
        restored.load_bt_state(state)
    restored.load_bt_state(state, initialize_pixel_adapter=True)


def test_two_updates_and_exact_trainer_resume(tmp_path):
    (tmp_path / "images").mkdir()
    Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path / "images/a.png")
    config = small_config(tmp_path, "joint")
    config["model"].update(backend="C", codec=deepcopy(CODEC))
    config["self_flow"] = dict(enabled=False, mixed_sampling=False)
    config["decoder"] = dict(method=PIXEL_METHOD, pixel_adapter=deepcopy(SMALL_PIXEL))
    config["quality"] = deepcopy(QUALITY)
    config["objective"] = dict(base_weight=.1, repaired_weight=0.)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    config["train"].update(batch_size=1, accumulation_steps=1, effective_batch=1, router_ramp_steps=0)
    initial = pixel_model().export_state()
    del initial["pixel_adapter"]
    source = tmp_path / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                             raw=initial, assets={"test": "pixel"}, config=deepcopy(config), calibration={}))
    def run(output, resume=None, max_updates=None):
        cfg = deepcopy(config); cfg["train"]["output"] = str(output)
        args = SimpleNamespace(resume=str(resume) if resume else None,
                               init_checkpoint=None if resume else str(source), max_updates=max_updates)
        with patch.object(trainer, "make_model", side_effect=pixel_model), \
             patch.object(trainer, "asset_identity", return_value={"test": "pixel"}), redirect_stdout(io.StringIO()):
            trainer.train(cfg, args, 0, 1, torch.device("cpu"))
    run(tmp_path / "whole")
    run(tmp_path / "resumed", max_updates=1)
    paused = read_checkpoint(tmp_path / "resumed/paused.pt")
    assert paused["step"] == 1 and "pixel_adapter" in paused["raw"] and "teacher" not in paused
    run(tmp_path / "resumed", resume=tmp_path / "resumed/paused.pt")
    whole, resumed = [read_checkpoint(tmp_path / name / "complete.pt") for name in ("whole", "resumed")]
    assert whole["step"] == resumed["step"] == 2
    for part in ("raw", "ema"):
        for component, values in whole[part].items():
            for name, value in values.items():
                assert torch.equal(value, resumed[part][component][name]), (part, component, name)
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
    for key in ("optimizer", "aux_optimizer", "scheduler", "rng_by_rank", "scaler"):
        same_state(whole[key], resumed[key])
