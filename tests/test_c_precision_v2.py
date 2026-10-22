"""Production v2 paths: actual rANS, non-identity conditions, RD and resume."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import check_stage_model, load_config, schedule_factor
from models.bt_self_flow.losses import ImageLoss, rate_beta, reconstruction_objective
from models.bt_self_flow.packet import BTImagePacket, RateControl
from models.bt_self_flow.validation import validate
from test_bt_backend import tiny_model
from test_bt_training import make_tiny_stage, small_config


torch.set_num_threads(1)
RECIPE = Path(__file__).resolve().parents[1]/"configs/bt_self_flow/precision_v2"
CODEC = dict(ch_emd=4, channel=16, channel_out=4, precision_conditioning=True, rate_likelihood="noise")
QUALITY = dict(values=[0., .2, .4, .6, .8, 1.], rate_coefficients=[32., 16., 8., 4., 2., 1.])


def new_codec():
    return BTLatentCodec(**CODEC)


def stage_factory(config, device):
    model = make_tiny_stage(config, device)
    torch.manual_seed(903)
    model.codec = new_codec().to(device)
    return model


def full_factory(config, device):
    model = tiny_model(config["model"]["backend"])
    model.vae.requires_grad_(False)
    torch.manual_seed(903)
    model.codec = new_codec()
    return model.to(device)


def config_for(root, phase="codec"):
    config = small_config(root, phase)
    config["model"]["codec"] = deepcopy(CODEC)
    config["quality"] = deepcopy(QUALITY)
    config["objective"] = dict(base_weight=1. if phase == "codec" else .1, repaired_weight=0.)
    config["optimizer"].update(schedule="constant", warmup=0, joint_codec_lr=1e-4, vae_lr=1e-4)
    config["train"].update(router_uniform_steps=0, router_ramp_steps=0, require_quality_span=False)
    return config


def activate_conditions(codec):
    # Identity initial adapters would fail to reveal a missing decoder condition.
    with torch.no_grad():
        for affine in (codec.analysis_precision, codec.synthesis_precision):
            nn.init.normal_(affine.projection.weight, std=.015)
            nn.init.normal_(affine.projection.bias, std=.01)
        for projection in codec.router.precision_projections:
            nn.init.normal_(projection.weight, std=.03)


def ddp_v2_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root/'rendezvous'}", timeout=timedelta(seconds=180))
    try:
        config = config_for(root)
        config["train"].update(accumulation_steps=1, effective_batch=2)
        models = []
        def factory(cfg, device):
            model = stage_factory(cfg, device)
            models.append(model)
            return model
        with patch.object(trainer, "make_model", side_effect=factory), \
             patch.object(trainer, "asset_identity", return_value={"test": "v2"}), redirect_stdout(io.StringIO()):
            trainer.train(config, type("Args", (), dict(resume=None, init_checkpoint=None, max_updates=None))(),
                          rank, 2, torch.device("cpu"))
        for parameter in models[0].parameters():
            if parameter.requires_grad:
                expected = parameter.detach().clone()
                dist.broadcast(expected, 0)
                assert torch.equal(parameter, expected), "v2 DDP parameters diverged"
    finally:
        dist.destroy_process_group()


def write_image(root):
    (root/"images").mkdir()
    Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/a.png")


def test_complete_recipe_pair_and_lr_boundaries():
    warm, c, d = [load_config(RECIPE/f"{name}.yaml") for name in ("common", "c_joint", "d_joint")]
    assert (warm["train"]["batch_size"], warm["train"]["accumulation_steps"], warm["train"]["effective_batch"]) == (8, 1, 16)
    assert warm["train"]["steps"] == 50000 and c["train"]["steps"] == 200000
    assert c["train"]["batch_size"]*c["train"]["accumulation_steps"] == c["train"]["effective_batch"] == 8
    d["model"]["backend"], d["train"]["output"] = c["model"]["backend"], c["train"]["output"]
    assert c == d
    assert warm["quality"] == c["quality"] == QUALITY
    assert warm["model"]["codec"] == c["model"]["codec"]
    assert c["image_loss"] == dict(mse_weight=4., lpips_weight=1., dists_weight=1.)
    for step, factor in ((0, 1), (99999, 1), (100000, .5), (159999, .5), (160000, .1), (190000, .01)):
        assert schedule_factor(step, 200000, c["optimizer"]) == pytest.approx(factor)
    for q, coefficient in zip(QUALITY["values"], QUALITY["rate_coefficients"]):
        assert rate_beta(RateControl(q), QUALITY) == pytest.approx(coefficient)
    for rank in (0, 1):
        assert [trainer.sample_control(c, step, rank, (2, 8, 8), "cpu").value for step in range(30)] == [
            trainer.sample_control(d, step, 0, (2, 8, 8), "cpu").value for step in range(30)]


def test_q_changes_analysis_and_content_decisions_without_cross_q_cache():
    torch.manual_seed(903)
    codec = new_codec().eval()
    activate_conditions(codec)
    clean = torch.randn(1, 4, 16, 24)
    a, b = codec.analyze(clean, RateControl(0)), codec.analyze(clean, RateControl(1))
    assert not torch.allclose(a.y, b.y)
    with pytest.raises(ValueError, match="cannot be reused"):
        codec.quantize_analysis(a, RateControl(1))
    mixed = RateControl(0., 1., torch.zeros(1, 1, 8, 12, dtype=torch.bool))
    with pytest.raises(ValueError, match="whole-image"):
        codec.steps(mixed, a.allocation)
    # Isolate the Router precision input, holding every content input fixed.
    captured = []
    hook = codec.router.head.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
    for q in (0, 1):
        embedding = codec.precision_embedding(codec.global_log_step(RateControl(q), 1, "cpu"))
        codec.router(clean, a.y, torch.rand(1, 1, 8, 12, generator=torch.Generator().manual_seed(9)), embedding)
    hook.remove()
    centered = [u-u.mean(1, keepdim=True) for u in captured]
    assert (centered[0]-centered[1]).abs().max() > 1e-7  # q is not merely a cancelling scalar bias.


def test_initialization_is_inside_a_bin_and_rd_reaches_all_router_content_branches():
    torch.manual_seed(903)
    codec = new_codec().train()
    clean = torch.randn(2, 4, 8, 8)
    result = codec(clean, RateControl(.4))
    assert result["allocation_codes"].unique().tolist() == [4]
    loss = (result["mean"]-clean).square().mean() + sum(-v.log().mean() for v in result["likelihoods"].values())
    loss.backward()
    parameters = dict(codec.named_parameters())
    names = ("g_a.pre1.0.weight", "router.source.weight", "router.feature.weight", "router.fusion.0.weight",
             "router.precision_projections.0.weight", "analysis_precision.projection.weight",
             "synthesis_precision.projection.weight", "router.head.2.weight")
    for name in names:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    assert all(torch.isfinite(p.grad).all() for p in parameters.values() if p.grad is not None)
    # Zero-initialized condition heads block their shared input MLP only on
    # the first update. Check that the embedding itself starts learning next.
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-4)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    result = codec(clean, RateControl(.4))
    ((result["mean"]-clean).square().mean() + sum(-v.log().mean() for v in result["likelihoods"].values())).backward()
    embedding_gradient = codec.precision_embedding.network[0].weight.grad
    assert embedding_gradient is not None and torch.isfinite(embedding_gradient).all() and embedding_gradient.abs().sum() > 0


def test_noisy_rate_does_not_change_hard_context_or_reconstruction():
    torch.manual_seed(903)
    codec = new_codec().train()
    clean = torch.randn(1, 4, 8, 8)
    one, two = codec(clean, RateControl(.4), trace=True), codec(clean, RateControl(.4), trace=True)
    assert torch.equal(one["mean"], two["mean"])
    assert not torch.equal(one["likelihoods"]["h"], two["likelihoods"]["h"])
    assert not torch.equal(one["likelihoods"]["y"], two["likelihoods"]["y"])
    for a, b in zip(one["trace"], two["trace"]):
        assert torch.equal(a["symbols"], b["symbols"]) and torch.equal(a["restored"], b["restored"])
    codec.eval()
    one, two = codec(clean, RateControl(.4)), codec(clean, RateControl(.4))
    assert all(torch.equal(one["likelihoods"][key], two["likelihoods"][key]) for key in ("y", "h"))


def test_six_quality_real_packets_nonidentity_conditions_and_fresh_decoder(tmp_path):
    torch.manual_seed(903)
    codec = new_codec().eval()
    activate_conditions(codec)
    codec.update(force=True)
    clean = torch.randn(1, 4, 16, 24)
    for q in QUALITY["values"]:
        control = RateControl(q)
        with torch.no_grad():
            direct, encoded = codec(clean, control, trace=True), codec.compress(clean, control, trace=True)
        packet = BTImagePacket("C", (512, 768), encoded["y_shape"], encoded["hyper_shape"], control.low_code,
                               control.high_code, encoded["hyper_stream"], encoded["main_stream"], encoded["allocation"])
        decoder = new_codec().eval()
        decoder.load_state_dict(codec.state_dict(), strict=True)
        with patch.object(decoder.router, "forward", side_effect=AssertionError("decoder ran Router")), \
             patch.object(decoder, "analyze", side_effect=AssertionError("decoder used source")):
            decoded = decoder.decompress(BTImagePacket.from_bytes(packet.to_bytes()), trace=True)
        assert torch.equal(encoded["mean"], decoded["mean"])
        assert torch.allclose(direct["mean"], decoded["mean"], atol=1e-6, rtol=1e-6)
        assert direct["delta"].log().mean().exp().item() == pytest.approx(4*8**(-q), rel=2e-6)
        assert len(packet.allocation) == 36 and len(packet.to_bytes()) == sum(packet.byte_breakdown().values())
        for a, b in zip(encoded["trace"], decoded["trace"]):
            assert torch.equal(a["symbols"], b["symbols"]) and torch.equal(a["indexes"], b["indexes"])
    torch.save(codec.state_dict(), tmp_path/"codec.pt")
    (tmp_path/"packet.btic").write_bytes(packet.to_bytes())
    script = """
import sys, torch
from pathlib import Path
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.packet import BTImagePacket
torch.set_num_threads(1)
p=Path(sys.argv[1]); m=BTLatentCodec(ch_emd=4,channel=16,channel_out=4,precision_conditioning=True,rate_likelihood='noise').eval()
m.load_state_dict(torch.load(p/'codec.pt',weights_only=True),strict=True)
def forbidden(*a,**kw): raise AssertionError('source used by decoder')
m.analyze=forbidden; m.router.forward=forbidden
torch.save(m.decompress(BTImagePacket.from_bytes((p/'packet.btic').read_bytes()))['mean'],p/'decoded.pt')
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=120)
    assert torch.equal(decoded["mean"], torch.load(tmp_path/"decoded.pt", weights_only=True))


@pytest.mark.parametrize("backend", ["C", "D"])
def test_full_backend_packet_path_one_nfe_and_finite_image_gradient(backend):
    config = {"model": {"backend": backend}}
    model = full_factory(config, "cpu")
    activate_conditions(model.codec)
    model.distortion_scale.calibrate(torch.tensor(1.), [torch.tensor([.1, .5, 1.])])
    image = torch.rand(1, 3, 256, 256)*2-1
    model.train()
    result = model(image, RateControl(.4))
    loss, metrics = reconstruction_objective(result, image, RateControl(.4), model.distortion_scale.s_data,
                                             ImageLoss(lpips_weight=0, dists_weight=0, mse_weight=4),
                                             objective=dict(base_weight=.1, repaired_weight=0), quality=QUALITY)
    loss.backward()
    assert not metrics["latent"].requires_grad
    for parameter in (model.codec.router.head[-1].weight, model.codec.synthesis_precision.projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.vae.encoder.parameters())
    model.eval(); model.codec.update(force=True)
    for q in (0, .4, 1):
        packet = model.compress(image, RateControl(q))
        with torch.no_grad():
            expected = model(image, RateControl(q)).image
        with patch.object(model.codec.router, "forward", side_effect=AssertionError("decoder Router")), \
             patch.object(model, "encode_clean_latent", side_effect=AssertionError("decoder clean image")):
            before = model.prediction_calls
            decoded = model.decompress(packet.to_bytes())
        assert model.prediction_calls-before == 1
        assert bool(packet.scales) == (backend == "D")
        assert torch.allclose(expected, decoded, atol=1e-5, rtol=1e-5)


def test_validation_reuses_only_frozen_ae_and_applies_configured_rd():
    model = stage_factory({"train": {"seed": 903}}, "cpu")
    images = [torch.rand(3, 256, 256)*2-1]
    with patch.object(model, "encode_clean_latent", wraps=model.encode_clean_latent) as encoder, \
         patch.object(model.codec, "analyze", wraps=model.codec.analyze) as analysis:
        report = validate(model, images, actual_rate=False, quality=QUALITY, objective=dict(base_weight=1.))
    assert encoder.call_count == 1 and analysis.call_count == 6
    for q in QUALITY["values"]:
        row = report[str(q)]
        assert row["loss"] == pytest.approx(row["base"]+rate_beta(RateControl(q), QUALITY)*row["rate_est"], rel=1e-6)


def test_noise_training_resume_ema_handoff_and_both_e2e_arms(tmp_path):
    write_image(tmp_path)
    args = type("Args", (), dict(resume=None, init_checkpoint=None, max_updates=None))()
    config = config_for(tmp_path)
    with patch.object(trainer, "make_model", side_effect=stage_factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v2"}), redirect_stdout(io.StringIO()):
        trainer.train(deepcopy(config), args, 0, 1, torch.device("cpu"))
        split = deepcopy(config); split["train"]["output"] = str(tmp_path/"split")
        args.max_updates = 1
        trainer.train(split, args, 0, 1, torch.device("cpu"))
        args.resume, args.max_updates = str(tmp_path/"split/paused.pt"), None
        trainer.train(split, args, 0, 1, torch.device("cpu"))
    full, resumed = [read_checkpoint(tmp_path/name/"complete.pt") for name in ("codec", "split")]
    for state in ("raw", "ema"):
        for part, values in full[state].items():
            for name, tensor in values.items():
                assert torch.equal(tensor, resumed[state][part][name]), (state, part, name)
    assert full["scheduler"] == resumed["scheduler"]
    assert full["calibration"]["weights"] == "deployment_ema"
    assert full["raw"]["distortion_scale"]["s_data"] == full["ema"]["distortion_scale"]["s_data"]
    # Untrained tiny codec is not a research rate-span result; allow the fixture handoff.
    full["calibration"].update(rate_span_pass=True, quality_span_pass=True)
    torch.save(full, tmp_path/"common.pt")
    joint = config_for(tmp_path, "joint")
    with patch.object(trainer, "make_model", side_effect=full_factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v2"}), redirect_stdout(io.StringIO()):
        trainer.prepare_backend(joint, tmp_path/"common.pt", tmp_path/"backend_init.pt", "cpu")
    shared = read_checkpoint(tmp_path/"backend_init.pt")
    assert shared["codec_weights"] == "ema"
    for name, tensor in full["ema"]["codec"].items():
        assert torch.equal(tensor, shared["raw"]["codec"][name]), name
    for backend in ("C", "D"):
        joint["model"]["backend"] = backend
        joint["train"].update(output=str(tmp_path/backend), tensorboard=True)
        args.resume, args.init_checkpoint, args.max_updates = None, str(tmp_path/"backend_init.pt"), 1
        with patch.object(trainer, "make_model", side_effect=full_factory), \
             patch.object(trainer, "asset_identity", return_value={"test": "v2"}), \
             patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0)), \
             redirect_stdout(io.StringIO()):
            trainer.train(deepcopy(joint), args, 0, 1, torch.device("cpu"))
        record = json.loads((tmp_path/backend/"train.jsonl").read_text().splitlines()[0])
        assert record["q"] in QUALITY["values"] and record["weighted_repaired"] == 0
        assert record["gradient_norms_after_clip"]["router"] > 0
        assert list((tmp_path/backend/"tensorboard").glob("events.out.tfevents.*"))
        assert read_checkpoint(tmp_path/backend/"paused.pt")["step"] == 1


def test_v2_two_rank_main_and_auxiliary_updates(tmp_path):
    write_image(tmp_path)
    mp.spawn(ddp_v2_worker, args=(str(tmp_path),), nprocs=2, join=True)
    state = read_checkpoint(tmp_path/"codec/complete.pt")
    assert state["step"] == 2 and state["contract"]["world_size"] == 2


def test_stage_handoff_rejects_different_rate_training(tmp_path):
    source, target = config_for(tmp_path), config_for(tmp_path, "joint")
    check_stage_model(target, source)
    target["quality"]["rate_coefficients"][0] = 64.
    with pytest.raises(ValueError, match="must match across stages"):
        check_stage_model(target, source)


def test_ema_legacy_aliases_do_not_overwrite_averaged_weights_on_load():
    model = stage_factory({"train": {"seed": 903}}, "cpu")
    ema = DeploymentEMA(model, decay=.9)
    parameter = model.codec.g_a.analysis_transform[2].down.weight
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.)
    ema.update(model)
    canonical = "g_a.analysis_transform.2.down.weight"
    alias = "g_a.analysis_transform.2.branch1.weight"
    assert torch.equal(ema.state["codec"][canonical], ema.state["codec"][alias])
    expected = before.lerp(parameter.detach(), .1)
    model.load_bt_state(ema.state)
    assert torch.equal(parameter, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is inaccessible")
@pytest.mark.parametrize("backend", ["C", "D"])
def test_v2_production_cuda_amp_updates_and_real_packet(tmp_path, backend):
    """Native BF16 on 5090 / scaled FP16 on V100, through the real trainer."""
    write_image(tmp_path)
    config = config_for(tmp_path, "joint")
    config["model"]["backend"] = backend
    config["train"].update(output=str(tmp_path/backend), tensorboard=True)
    source = full_factory(config, "cpu").eval()
    activate_conditions(source.codec)
    source.distortion_scale.calibrate(torch.tensor(1.), [torch.tensor([.1, .5, 1.])])
    source_file = tmp_path/"shared.pt"
    save_atomic(source_file, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared",
                                 step=0, config=config, assets={"test": "v2"}, raw=source.export_state()))
    del source
    models = []
    def factory(cfg, device):
        model = full_factory(cfg, device)
        models.append(model)
        return model
    args = type("Args", (), dict(resume=None, init_checkpoint=str(source_file), max_updates=None))()
    with patch.object(trainer, "make_model", side_effect=factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "v2"}), \
         patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0)), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device("cuda:0"))
    saved = read_checkpoint(tmp_path/backend/"complete.pt")
    assert saved["step"] == 2 and saved["micro_seen"] == 4
    for row in (json.loads(line) for line in (tmp_path/backend/"train.jsonl").read_text().splitlines()):
        assert row["gradient_norms_after_clip"]["router"] > 0 and row["weighted_repaired"] == 0
    model = models.pop().eval()
    model.load_bt_state(saved["ema"])
    model.codec.update(force=True)
    with torch.no_grad():
        image = torch.rand(1, 3, 256, 256, device="cuda")*2-1
        expected = model(image, RateControl(.4)).image
        packet = model.compress(image, RateControl(.4))
        with patch.object(model.codec.router, "forward", side_effect=AssertionError("decoder Router")):
            decoded = model.decompress(packet.to_bytes())
    assert (decoded-expected).abs().max() <= 2e-4
