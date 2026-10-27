"""Precision-consistent C: paid hard fields, shared conditions and real decoding."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderDC

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, read_checkpoint
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import check_stage_model, load_config, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import BTImagePacket, RateControl
from models.bt_self_flow.precision_conditioning import PrecisionGeneratorConditioner
from models.bt_self_flow.router import SourcePrecisionPlanner, allocation_from_codes
from models.sana_self_flow import quality_conditioned_sana_forward, unwrap_sana_transformer
from test_bt_backend import tiny_model
from test_bt_training import make_tiny_stage
from test_c_precision_v2 import config_for as v2_config, write_image


torch.set_num_threads(1)
CODEC = dict(ch_emd=4, channel=16, channel_out=4, precision_conditioning=True,
             precision_consistent=True, rate_likelihood="noise")
RECIPE = Path(__file__).resolve().parents[1] / "configs/bt_self_flow/precision_v3"


def new_codec():
    return BTLatentCodec(**CODEC)


def activate_conditions(model):
    codec = model.codec if hasattr(model, "codec") else model
    with torch.no_grad():
        for name in ("analysis_precision", "analysis_precision_low", "synthesis_precision_low", "synthesis_precision"):
            affine = getattr(codec, name)
            nn.init.normal_(affine.projection.weight, std=.015)
            nn.init.normal_(affine.projection.bias, std=.01)
        for projection in codec.router.precision_projections:
            nn.init.normal_(projection.weight, std=.02)
        if hasattr(model, "quality_conditioner"):
            for projection in (model.quality_conditioner.block_projection, model.quality_conditioner.output_projection):
                nn.init.normal_(projection.weight, std=.003)


def full_factory(config=None, device="cpu"):
    model = tiny_model("C")
    model.vae.requires_grad_(False)
    torch.manual_seed(903)
    model.codec = new_codec()
    model.quality_conditioner = PrecisionGeneratorConditioner(32, 4)
    return model.to(device)


def stage_factory(config, device):
    model = make_tiny_stage(config, device)
    torch.manual_seed(903)
    model.codec = new_codec().to(device)
    return model


def config_for(root, phase="codec"):
    config = v2_config(root, phase)
    config["model"].update(backend="C", codec=deepcopy(CODEC))
    return config


def packet_from(codec, clean, q=.4):
    control = RateControl(q)
    encoded = codec.compress(clean, control, trace=True)
    packet = BTImagePacket("C", (clean.shape[-2] * 32, clean.shape[-1] * 32),
                           encoded["y_shape"], encoded["hyper_shape"], control.low_code,
                           control.high_code, encoded["hyper_stream"], encoded["main_stream"],
                           encoded["allocation"], version=codec.packet_version)
    return encoded, packet


def test_recipe_identity_and_stage_contract():
    warm, joint = [load_config(RECIPE / f"{name}.yaml") for name in ("common", "c_joint")]
    assert warm["model"]["codec"]["precision_consistent"]
    assert warm["model"]["codec"] == joint["model"]["codec"]
    assert warm["model"]["backend"] == joint["model"]["backend"] == "C"
    for config in (warm, joint):
        assert config["train"]["batch_size"] == 8
        assert config["train"]["accumulation_steps"] == 1
        assert config["train"]["effective_batch"] == 8
    assert "c_precision_v3" in warm["train"]["output"] and "c_precision_v3" in joint["train"]["output"]
    check_stage_model(joint, warm)
    old = deepcopy(warm)
    old["model"]["codec"].pop("precision_consistent")
    with pytest.raises(ValueError, match="codec differs"):
        check_stage_model(joint, old)
    with pytest.raises(ValueError, match="requires precision_conditioning"):
        BTLatentCodec(precision_consistent=True)


def test_planner_precedes_analysis_and_never_reads_reference_cost():
    codec = new_codec().eval()
    assert isinstance(codec.router, SourcePrecisionPlanner)
    assert not hasattr(codec.router, "feature") and not hasattr(codec.router, "cost")
    order = []
    hooks = [codec.router.register_forward_hook(lambda *a: order.append("plan")),
             codec.g_a.pre1.register_forward_hook(lambda *a: order.append("analyze"))]
    try:
        with patch.object(codec, "_prior", side_effect=AssertionError("planner used coding cost")):
            source = codec.analyze(torch.randn(1, 4, 8, 16), RateControl(.4))
    finally:
        for hook in hooks:
            hook.remove()
    assert order == ["plan", "analyze"]
    assert source.y.shape[-2:] == source.allocation_codes.shape[-2:] == (4, 8)
    assert torch.equal(source.allocation, allocation_from_codes(source.allocation_codes))


def test_all_transform_scales_receive_the_same_hard_field():
    codec = new_codec().eval()
    activate_conditions(codec)
    # Force a nonuniform field so global-only conditioning cannot pass.
    codes = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4) % 8
    allocation = allocation_from_codes(codes)
    conditions = {}
    hooks = []
    for name in ("analysis_precision", "analysis_precision_low", "synthesis_precision_low", "synthesis_precision"):
        hooks.append(getattr(codec, name).register_forward_pre_hook(
            lambda module, inputs, name=name: conditions.update({name: inputs[1].detach().clone()})))
    try:
        with patch.object(codec.router, "forward", return_value=(allocation, codes)):
            view = codec(torch.randn(1, 4, 8, 8), RateControl(.4))
    finally:
        for hook in hooks:
            hook.remove()
    local = codec.precision_embedding(view["delta"].log()).detach()
    assert torch.equal(conditions["analysis_precision_low"], local)
    assert torch.equal(conditions["synthesis_precision_low"], local)
    high = F.interpolate(local, size=(8, 8), mode="nearest")
    assert torch.equal(conditions["analysis_precision"], high)
    assert torch.equal(conditions["synthesis_precision"], high)
    expected = codec.steps(RateControl(.4), allocation)
    assert torch.equal(view["delta"], expected)


def test_six_quality_real_rans_fresh_decoder_and_architecture_rejection(tmp_path):
    torch.manual_seed(903)
    codec = new_codec().eval()
    activate_conditions(codec)
    codec.update(force=True)
    clean = torch.randn(1, 4, 16, 24)
    for q in (0., .2, .4, .6, .8, 1.):
        encoded, packet = packet_from(codec, clean, q)
        received = BTImagePacket.from_bytes(packet.to_bytes())
        assert received.version == 3 and not received.scales
        assert len(packet.to_bytes()) == sum(packet.byte_breakdown().values())
        assert len(packet.allocation) == 36
        decoder = new_codec().eval()
        decoder.load_state_dict(codec.state_dict(), strict=True)
        with patch.object(decoder.router, "forward", side_effect=AssertionError("source planner at receiver")), \
             patch.object(decoder, "analyze", side_effect=AssertionError("source analysis at receiver")):
            decoded = decoder.decompress(received, trace=True)
        assert torch.equal(encoded["mean"], decoded["mean"])
        for a, b in zip(encoded["trace"], decoded["trace"]):
            assert torch.equal(a["symbols"], b["symbols"])
            assert torch.equal(a["indexes"], b["indexes"])
        with pytest.raises(ValueError, match="version"):
            decoder.decompress(replace(received, version=1))
        old = BTLatentCodec(ch_emd=4, channel=16, channel_out=4, precision_conditioning=True).eval()
        with pytest.raises(ValueError, match="version"):
            old.decompress(received)
    torch.save(codec.state_dict(), tmp_path / "codec.pt")
    (tmp_path / "packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.packet import BTImagePacket
torch.set_num_threads(1)
p = Path(sys.argv[1])
m = BTLatentCodec(ch_emd=4,channel=16,channel_out=4,precision_conditioning=True,precision_consistent=True,rate_likelihood='noise').eval()
m.load_state_dict(torch.load(p/'codec.pt',weights_only=True),strict=True)
def forbidden(*a, **kw): raise AssertionError('source information at receiver')
m.router.forward = forbidden
m.analyze = forbidden
torch.save(m.decompress(BTImagePacket.from_bytes((p/'packet.btic').read_bytes()))['mean'],p/'decoded.pt')
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=120)
    assert torch.equal(decoded["mean"], torch.load(tmp_path / "decoded.pt", weights_only=True))


def test_quality_cache_and_mixed_quality_are_rejected():
    codec = new_codec()
    clean = torch.randn(1, 4, 8, 8)
    source = codec.analyze(clean, RateControl(.4))
    with pytest.raises(ValueError, match="reused across qualities"):
        codec.quantize_analysis(source, RateControl(.6))
    with pytest.raises(ValueError, match="whole-image"):
        codec.analyze(clean, RateControl(.2, 1., torch.zeros(1, 1, 4, 4, dtype=torch.bool)))


def test_zero_generator_condition_matches_native_and_keeps_scheduler():
    model = full_factory().eval()
    coarse = torch.randn(1, 4, 8, 8)
    delta = torch.rand(1, 1, 4, 4) + .5
    time = model.self_flow_timestep(1, coarse.device)
    text = model._fixed_text_condition(1, coarse.device)
    expected = model.DiT(coarse, encoder_hidden_states=text, timestep=time, return_dict=False)[0]
    actual, _ = model.predict(coarse, precision=delta)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    latent, _, _ = model.repair(coarse, precision=delta)
    assert torch.allclose(latent, model.sched.step(expected, coarse), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="quantization steps"):
        model.predict(coarse)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_image_gradients_shared_embedding_and_independent_receive(checkpointed):
    model = full_factory().train()
    model.checkpoint_blocks = checkpointed
    if checkpointed:
        # Exercise the production DC-AE checkpoint API rather than TinyAE's stub decoder.
        model.vae = AutoencoderDC(
            latent_channels=4, encoder_block_out_channels=(16,) * 6,
            decoder_block_out_channels=(16,) * 6,
            encoder_layers_per_block=(1,) * 6, decoder_layers_per_block=(1,) * 6,
            encoder_block_types=("ResBlock",) * 6, decoder_block_types=("ResBlock",) * 6,
            encoder_qkv_multiscales=((),) * 6, decoder_qkv_multiscales=((),) * 6,
            attention_head_dim=8, scaling_factor=1.,
        ).requires_grad_(False).eval()
    activate_conditions(model)
    image = torch.rand(1, 3, 256, 256) * 2 - 1
    config = config_for(Path("unused"), "joint")
    result = model(image, RateControl(.4))
    loss, _ = reconstruction_objective(result, image, RateControl(.4), model.distortion_scale.s_data,
                                       ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0),
                                       objective=config["objective"], quality=config["quality"])
    loss.backward()
    parameters = [model.codec.router.head[-1].weight,
                  model.codec.precision_embedding.network[0].weight,
                  model.quality_conditioner.block_projection.weight,
                  model.quality_conditioner.output_projection.weight,
                  model.quality_conditioner.layer_gain]
    parameters += [getattr(model.codec, name).projection.weight for name in
                   ("analysis_precision", "analysis_precision_low", "synthesis_precision_low", "synthesis_precision")]
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.vae.encoder.parameters())
    groups, auxiliary = optimizer_groups(model, config)
    assert sum(id(p) == id(model.codec.precision_embedding.network[0].weight)
               for group in groups for p in group["params"]) == 1
    assert any(model.quality_conditioner.block_projection.weight is p for group in groups for p in group["params"])
    model.eval(); model.codec.update(force=True)
    with torch.no_grad():
        direct = model(image, RateControl(.4)).image
        packet = model.compress(image, RateControl(.4))
    decoder = full_factory().eval()
    if checkpointed:
        decoder.vae = deepcopy(model.vae)
    decoder.load_bt_state(model.export_state())
    before = decoder.prediction_calls
    with patch.object(decoder, "encode_clean_latent", side_effect=AssertionError("receiver used source")), \
         patch.object(decoder.codec.router, "forward", side_effect=AssertionError("receiver used planner")):
        decoded = decoder.decompress(packet.to_bytes())
    assert decoder.prediction_calls - before == 1
    assert torch.allclose(decoded, direct, atol=1e-5, rtol=1e-5)


def test_sana_precision_uses_shared_phi_and_checkpointing_preserves_gradients():
    model = full_factory().train()
    activate_conditions(model)
    sana = unwrap_sana_transformer(model.DiT)
    latent = torch.randn(1, 4, 8, 8, requires_grad=True)
    time = model.self_flow_timestep(1, "cpu")
    log_step = torch.randn(1, 8, 8, requires_grad=True)
    results = []
    for checkpointed in (False, True):
        model.zero_grad(set_to_none=True)
        latent.grad = None
        log_step.grad = None
        text = model._fixed_text_condition(1, "cpu")
        with patch.object(model.codec.precision_embedding, "forward", wraps=model.codec.precision_embedding.forward) as phi:
            output, _ = quality_conditioned_sana_forward(
                sana, latent, text, time, log_step, model.quality_conditioner,
                continuous_conditioning=True, checkpoint_blocks=checkpointed,
                precision_embedding=model.codec.precision_embedding,
            )
            output.square().mean().backward()
            assert phi.call_count == 1
        results.append((output.detach(), latent.grad.clone(), log_step.grad.clone(),
                        model.codec.precision_embedding.network[0].weight.grad.clone()))
    for expected, actual in zip(*results):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_production_codec_channels_and_512_by_768_geometry():
    codec = BTLatentCodec(precision_conditioning=True, precision_consistent=True, rate_likelihood="noise").eval()
    activate_conditions(codec)
    codec.update(force=True)
    clean = torch.randn(1, 32, 16, 24)
    encoded, packet = packet_from(codec, clean)
    decoded = codec.decompress(BTImagePacket.from_bytes(packet.to_bytes()), trace=True)
    assert encoded["mean"].shape == clean.shape
    assert packet.y_shape == (8, 12) and packet.hyper_shape == (2, 3)
    assert torch.equal(encoded["mean"], decoded["mean"])


def test_generator_conditioner_ema_and_strict_restore():
    model = full_factory()
    ema = DeploymentEMA(model, decay=.9)
    before = model.quality_conditioner.block_projection.weight.detach().clone()
    with torch.no_grad():
        model.quality_conditioner.block_projection.weight.add_(1)
        model.codec.precision_embedding.network[0].weight.add_(.5)
    ema.update(model)
    expected = before.lerp(model.quality_conditioner.block_projection.weight.detach(), .1)
    assert torch.equal(ema.state["conditioner"]["block_projection.weight"], expected)
    receiver = full_factory()
    receiver.load_bt_state(ema.state)
    assert torch.equal(receiver.quality_conditioner.block_projection.weight, expected)
    missing = deepcopy(ema.state)
    missing.pop("conditioner")
    with pytest.raises(ValueError, match="missing its conditioner"):
        receiver.load_bt_state(missing)


def test_warmup_resume_ema_handoff_and_joint_exact_resume(tmp_path):
    write_image(tmp_path)
    args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
    config = config_for(tmp_path)
    with patch.object(trainer, "make_model", side_effect=stage_factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v3"}), redirect_stdout(io.StringIO()):
        trainer.train(deepcopy(config), args, 0, 1, torch.device("cpu"))
        split = deepcopy(config); split["train"]["output"] = str(tmp_path / "split")
        args.max_updates = 1
        trainer.train(split, args, 0, 1, torch.device("cpu"))
        args.resume, args.max_updates = str(tmp_path / "split/paused.pt"), None
        trainer.train(split, args, 0, 1, torch.device("cpu"))
    full, resumed = [read_checkpoint(tmp_path / name / "complete.pt") for name in ("codec", "split")]
    assert full["contract"]["format"] == "bt_packet_v3_q16_a3_precision_consistent"
    for weights in ("raw", "ema"):
        for part, values in full[weights].items():
            for name, tensor in values.items():
                assert torch.equal(tensor, resumed[weights][part][name]), (weights, part, name)
    # This fixture checks the stage protocol, not research rate-span performance.
    full["calibration"].update(rate_span_pass=True, quality_span_pass=True)
    torch.save(full, tmp_path / "common.pt")
    joint = config_for(tmp_path, "joint")
    with patch.object(trainer, "make_model", side_effect=full_factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v3"}), redirect_stdout(io.StringIO()):
        trainer.prepare_backend(joint, tmp_path / "common.pt", tmp_path / "backend_init.pt", "cpu")
    shared = read_checkpoint(tmp_path / "backend_init.pt")
    assert "conditioner" in shared["raw"] and shared["codec_weights"] == "ema"
    for name, value in full["ema"]["codec"].items():
        assert torch.equal(value, shared["raw"]["codec"][name])
    with patch.object(trainer, "make_model", side_effect=full_factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v3"}), \
         patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0)), \
         redirect_stdout(io.StringIO()):
        args.resume, args.init_checkpoint, args.max_updates = None, str(tmp_path / "backend_init.pt"), None
        trainer.train(deepcopy(joint), args, 0, 1, torch.device("cpu"))
        joint["train"]["output"] = str(tmp_path / "joint_split")
        args.max_updates = 1
        trainer.train(joint, args, 0, 1, torch.device("cpu"))
        args.resume, args.init_checkpoint, args.max_updates = str(tmp_path / "joint_split/paused.pt"), None, None
        trainer.train(joint, args, 0, 1, torch.device("cpu"))
    full, resumed = [read_checkpoint(tmp_path / name / "complete.pt") for name in ("joint", "joint_split")]
    for weights in ("raw", "ema"):
        for part, values in full[weights].items():
            for name, tensor in values.items():
                assert torch.equal(tensor, resumed[weights][part][name]), (weights, part, name)


@pytest.mark.skipif(os.environ.get("C_V3_CUDA_CHECK") != "1" or not torch.cuda.is_available(),
                    reason="opt-in CUDA check; select a device with available memory")
def test_v3_cuda_bf16_update_and_real_packet():
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(1024 ** 3 / torch.cuda.get_device_properties(device).total_memory, device)
    model = full_factory(device=device).train()
    activate_conditions(model)
    image = torch.rand(1, 3, 256, 256, device=device) * 2 - 1
    config = config_for(Path("unused"), "joint")
    groups, _ = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(image, RateControl(.4))
        loss, _ = reconstruction_objective(result, image, RateControl(.4), model.distortion_scale.s_data,
                                           ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0).to(device),
                                           objective=config["objective"], quality=config["quality"])
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (model.codec.router.head[-1].weight,
                      model.codec.precision_embedding.network[0].weight,
                      model.quality_conditioner.block_projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    model.eval(); model.codec.update(force=True)
    with torch.no_grad():
        direct = model(image, RateControl(.4)).image
        packet = model.compress(image, RateControl(.4))
        with patch.object(model.codec.router, "forward", side_effect=AssertionError("source at receiver")):
            decoded = model.decompress(packet.to_bytes())
    assert packet.version == 3 and (direct - decoded).abs().max() <= 2e-4
