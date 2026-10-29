"""New-C/PiT integration: conditioned tokens, real packets and exact continuation."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import (check_stage_model, load_config, optimizer_groups,
                                            training_contract, validate_decoder)
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.precision_conditioning import PrecisionGeneratorConditioner
from models.bt_self_flow.precision_pixel_model import PrecisionCPixelAdapterCodec, METHOD as PIXEL_METHOD
from models.bt_self_flow.pixel_model import native_sana_features
from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from test_bt_training import small_config
from test_c_precision_v3 import CODEC, activate_conditions
from test_pixel_adapter import QUALITY, SMALL_PIXEL, activate_output, pixel_model


ROOT = Path(__file__).resolve().parents[1]


def precision_pixel_model(config=None, device="cpu"):
    model = pixel_model()
    model.__class__ = PrecisionCPixelAdapterCodec
    model.codec = BTLatentCodec(**CODEC)
    model.quality_conditioner = PrecisionGeneratorConditioner(32, 4)
    model.codec.update(force=True)
    return model.to(device)


def test_recipe_preserves_new_c_and_enforces_architecture():
    native = load_config(ROOT / "configs/bt_self_flow/precision_v3/c_joint.yaml")
    pixel = load_config(ROOT / "configs/bt_self_flow/precision_v3/pixel_joint.yaml")
    for key in ("model", "quality", "objective", "image_loss", "optimizer", "data"):
        assert pixel[key] == native[key]
    check_stage_model(pixel, native)
    assert pixel["decoder"]["pixel_adapter"]["depth"] == 2
    assert pixel["train"]["batch_size"] == 8 and pixel["train"]["accumulation_steps"] == 1
    pixel["model"]["codec"]["precision_consistent"] = False
    with pytest.raises(ValueError, match="frontend"):
        validate_decoder(pixel)
    with pytest.raises(ValueError, match="codec differs"):
        check_stage_model(pixel, native)


def test_zero_pit_preserves_active_new_c_conditioning_and_final_tokens():
    model = precision_pixel_model().eval()
    activate_conditions(model)
    mu = torch.randn(1, 4, 8, 12)
    delta = torch.exp(torch.randn(1, 1, 4, 6))
    with torch.no_grad():
        expected_prediction, _ = BTCodec.predict(model, mu, precision=delta)
        _, expected_image, _ = BTCodec.repair(model, mu, precision=delta)
        before = model.prediction_calls
        with patch("models.bt_self_flow.precision_pixel_model.quality_conditioned_sana_forward",
                   wraps=quality_conditioned_sana_forward) as conditioned_forward:
            actual = model.decode_received(mu, delta, return_features=True)
        assert conditioned_forward.call_count == 1
        assert conditioned_forward.call_args.kwargs["capture_layer"] == 3
        native_prediction, native_tokens, _ = native_sana_features(
            unwrap_sana_transformer(model.DiT), mu, model._fixed_text_condition(1, mu.device),
            model.self_flow_timestep(1, mu.device))
    assert model.prediction_calls - before == 1
    assert actual["sana_token_hw"] == (8, 12)
    assert actual["sana_tokens"].shape == native_tokens.shape
    assert not torch.allclose(actual["sana_tokens"], native_tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual["prediction"], expected_prediction, atol=0, rtol=0)
    torch.testing.assert_close(actual["image"], expected_image, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(expected_prediction, native_prediction, atol=1e-5, rtol=1e-5)


def test_gradients_cover_both_pit_blocks_and_new_c_precision_path():
    model = precision_pixel_model().train()
    activate_conditions(model)
    activate_output(model.pixel_adapter)
    result = model(torch.randn(1, 3, 256, 256), RateControl(.4))
    result.image.square().mean().backward()
    parameters = [model.pixel_adapter.blocks[i].attn.qkv.weight for i in range(2)]
    parameters += [model.pixel_adapter.content_projection.weight, model.pixel_adapter.scale_embedding.mlp[0].weight,
                   model.quality_conditioner.block_projection.weight, model.quality_conditioner.output_projection.weight]
    parameters += [getattr(model.codec, name).projection.weight for name in
                   ("analysis_precision", "analysis_precision_low", "synthesis_precision_low", "synthesis_precision")]
    parameters += list(model.codec.precision_embedding.parameters())
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    groups, auxiliary = optimizer_groups(model, small_config(Path("unused"), "joint"))
    assert "pixel_adapter" in {group["name"] for group in groups} and auxiliary
    ema = DeploymentEMA(model)
    old = ema.state["pixel_adapter"]["pixel_output.weight"].clone()
    with torch.no_grad():
        model.pixel_adapter.pixel_output.weight.add_(1.)
    ema.update(model)
    torch.testing.assert_close(ema.state["pixel_adapter"]["pixel_output.weight"], old + .001)


def test_v3_packets_decode_six_qualities_without_sender_or_extra_bits():
    model = precision_pixel_model().eval()
    activate_conditions(model)
    activate_output(model.pixel_adapter)
    image = torch.randn(1, 3, 256, 512).clamp(-1, 1)
    receiver = precision_pixel_model().eval()
    receiver.load_bt_state(model.export_state())
    def forbidden(*args, **kwargs):
        raise AssertionError("receiver used source-only information")
    for q in QUALITY["values"]:
        with torch.no_grad():
            expected = model(image, RateControl(q)).image
            packet = model.compress(image, RateControl(q))
        assert packet.version == 3 and not packet.scales
        assert len(packet.to_bytes()) == sum(packet.byte_breakdown().values())
        before = receiver.prediction_calls
        with patch.object(receiver, "encode_clean_latent", side_effect=forbidden), \
             patch.object(receiver.codec.router, "forward", side_effect=forbidden):
            actual = receiver.decompress(packet.to_bytes())
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert receiver.prediction_calls - before == 1
    incomplete = deepcopy(model.export_state())
    del incomplete["pixel_adapter"]
    with pytest.raises(ValueError, match="pixel_adapter"):
        receiver.load_bt_state(incomplete)
    receiver.load_bt_state(incomplete, initialize_pixel_adapter=True)
    del incomplete["conditioner"]
    with pytest.raises(ValueError, match="conditioner"):
        receiver.load_bt_state(incomplete, initialize_pixel_adapter=True)


def test_asset_relocation_requires_matching_hashes_and_preserves_codec_guard():
    source = {"model": {"sana_path": "/old/sana", "text_path": "/old/text", "codec": deepcopy(CODEC)},
              "quality": deepcopy(QUALITY)}
    target = deepcopy(source)
    target["model"].update(sana_path="/new/sana", text_path="/new/text")
    with patch("models.bt_self_flow.experiment.asset_identity", return_value={"sha": "same"}):
        with pytest.raises(ValueError, match="sana_path"):
            check_stage_model(target, source)
        with pytest.raises(ValueError, match="sana_path"):
            check_stage_model(target, source, {"sha": "different"})
        check_stage_model(target, source, {"sha": "same"})
        target["model"]["codec"].pop("precision_consistent")
        with pytest.raises(ValueError, match="codec differs"):
            check_stage_model(target, source, {"sha": "same"})


@pytest.mark.parametrize("resume_storage", ["stride_preserving_cpu", "device"])
def test_exact_resume_preserves_conditioner_pit_ema_optimizer_and_rng(tmp_path, resume_storage):
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
    initial = precision_pixel_model().export_state()
    del initial["pixel_adapter"]
    source = tmp_path / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                             raw=initial, assets={"test": "precision-pixel"}, config=deepcopy(config), calibration={}))
    def run(name, resume=None, max_updates=None):
        cfg = deepcopy(config)
        cfg["train"]["output"] = str(tmp_path / name)
        args = SimpleNamespace(resume=str(resume) if resume else None,
                               init_checkpoint=None if resume else str(source), max_updates=max_updates)
        with patch.object(trainer, "make_model", side_effect=precision_pixel_model), \
             patch.object(trainer, "asset_identity", return_value={"test": "precision-pixel"}), redirect_stdout(io.StringIO()):
            trainer.train(cfg, args, 0, 1, torch.device("cpu"))
    run("whole")
    run("resumed", max_updates=1)
    paused = read_checkpoint(tmp_path / "resumed/paused.pt")
    assert paused["step"] == 1 and {"conditioner", "pixel_adapter"} <= paused["ema"].keys()
    config["execution"]["saved_tensors"] = resume_storage
    run("resumed", resume=tmp_path / "resumed/paused.pt")
    whole, resumed = [read_checkpoint(tmp_path / name / "complete.pt") for name in ("whole", "resumed")]
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
    for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "rng_by_rank", "scaler"):
        same_state(whole[key], resumed[key])


def test_storage_placement_may_change_on_resume_but_model_batch_and_objective_may_not():
    config = load_config(ROOT / "configs/bt_self_flow/precision_v3/pixel_joint.yaml")
    dataset = SimpleNamespace(fingerprint="same_data")
    source = training_contract(config, dataset, 1)
    changed = deepcopy(config)
    changed["execution"]["saved_tensors"] = "stride_preserving_cpu"
    assert training_contract(changed, dataset, 1) == source
    for section, key, value in (("train", "batch_size", 4), ("objective", "base_weight", .2),
                                ("model", "dit_rank", 64)):
        bad = deepcopy(changed)
        bad[section][key] = value
        assert training_contract(bad, dataset, 1) != source
    bad = deepcopy(changed)
    bad["model"]["codec"]["precision_consistent"] = False
    assert training_contract(bad, dataset, 1) != source
