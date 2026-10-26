"""Real codec/SANA contracts for C quantization-residual Self-Flow."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.bt_self_flow.checkpoint import DeploymentEMA, METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.packet import BTImagePacket, RateControl
from models.bt_self_flow.quantization_model import CQuantizationResidualCodec
from models.bt_self_flow.quantization_residual import (CResidualTeacher, QuantizationResidualHead,
                                                      c_sana_forward, erasure_mask)
from models.bt_self_flow.residual_training import balanced_quality, calibrate_residual, training_mask
from models.sana_self_flow import unwrap_sana_transformer
from test_bt_backend import tiny_model, tiny_sana
from test_bt_training import small_config
import train_bt_self_flow as trainer

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def tiny_residual():
    model = tiny_model("C")
    model.__class__ = CQuantizationResidualCodec
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
                               precision_conditioning=True, rate_likelihood="noise")
    model.residual_head = QuantizationResidualHead(32)
    model.residual_head.register_buffer("active", torch.tensor(False))
    model.residual_intervention = "normal"
    return model


def test_native_sana_and_checkpointed_residual_match():
    sana = tiny_sana()
    latent, text, time = torch.randn(2, 4, 8, 8), torch.randn(2, 2, 16), torch.full((2,), 700.)
    expected = sana(latent, text, time, return_dict=False)[0]
    head = QuantizationResidualHead(32)
    for checkpointed in (False, True):
        output = c_sana_forward(sana, latent, text, time, head=head,
                                log_step=torch.zeros(2, 1, 4, 4), student_layer=0,
                                injection_layer=2, checkpoint_blocks=checkpointed)[0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        assert head.mean.weight.grad.norm() > 0
        head.zero_grad(set_to_none=True)


def test_zero_init_preserves_c_and_head_is_used_in_image():
    model = tiny_residual().eval()
    image = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    with torch.no_grad():
        before = model(image).image
        model.activate_residual()
        after = model(image).image
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        model.residual_head.mean.bias.fill_(.1)
        changed = model(image).image
        assert (changed-after).abs().max() > 1e-5
        model.residual_intervention = "zero"
        zeroed = model(image).image
        torch.testing.assert_close(zeroed, before, rtol=0, atol=0)


def test_sf_detaches_codec_but_image_rd_trains_it():
    model = tiny_residual().train()
    model.activate_residual()
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
    image = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    mask[0, :, :2, :2] = True
    result = model(image, RateControl(.4), sf_teacher=teacher, sf_mask=mask)
    result.auxiliary["sf_loss"].backward()
    assert all(p.grad is None for p in model.codec.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert model.residual_head.mean.weight.grad.norm() > 0
    assert model.residual_head.variance.weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    result = model(image, RateControl(.4), sf_mask=mask)
    (result.image-image).square().mean().backward()
    assert model.codec.g_a.pre1[0].weight.grad.norm() > 0
    assert model.residual_head.mean.weight.grad.norm() > 0
    assert model.residual_head.variance.weight.grad is None


@pytest.mark.parametrize("q", [0., .2, .4, .6, .8, 1.])
def test_perturbed_native_packet_and_teacher_rng(q):
    torch.manual_seed(29)
    codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
                           precision_conditioning=True, rate_likelihood="noise").eval()
    codec.update(force=True)
    clean = torch.randn(1, 4, 16, 16)
    control = RateControl(q)
    source = codec.analyze(clean, control)
    mask = erasure_mask(source.allocation_codes, 1., .25, torch.Generator().manual_seed(11))
    received = codec.quantize_analysis(source, control, erase_mask=mask)
    encoded = codec.compress(clean, control, allocation_codes=received["allocation_codes"])
    packet = BTImagePacket("C", (512, 512), encoded["y_shape"], encoded["hyper_shape"],
                           control.low_code, control.high_code, encoded["hyper_stream"],
                           encoded["main_stream"], encoded["allocation"])
    fresh = deepcopy(codec)
    decoded = fresh.decompress(BTImagePacket.from_bytes(packet.to_bytes()))
    torch.testing.assert_close(decoded["mean"], received["mean"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(decoded["delta"], received["delta"], rtol=0, atol=0)
    codec.train()
    fine_delta = torch.minimum(codec.steps(control, source.allocation), received["delta"])
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        fine = codec.quantize_analysis(source, control, delta_override=fine_delta, compute_likelihood=False)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.all(fine["delta"] <= received["delta"])
    with pytest.raises(ValueError, match="no_grad"):
        codec.quantize_analysis(source, control, delta_override=fine_delta, compute_likelihood=False)


def test_head_teacher_and_deployment_ema_roundtrip():
    model = tiny_residual()
    ema = DeploymentEMA(model)
    model.activate_residual()
    with torch.no_grad():
        model.residual_head.mean.bias.add_(1.)
    ema.update(model)
    assert ema.state["residual"]["active"].item()
    torch.testing.assert_close(ema.state["residual"]["mean.bias"], torch.full((32,), .001))
    state = clone_state(model.export_state())
    fresh = tiny_residual()
    fresh.load_bt_state(state)
    assert fresh.residual_head.active.item()
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
    with torch.no_grad():
        model.text_proj.bias.add_(1.)
    before = teacher.text_projection.bias.clone()
    teacher.update_from_student(model.DiT, model.text_proj, decay=.999)
    torch.testing.assert_close(teacher.text_projection.bias, before+.001)
    newer = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
    newer.load_ema_state_dict(teacher.get_ema_state_dict())
    assert all(not p.requires_grad for p in newer.parameters())


def test_recipe_balanced_q_and_batch_fallback():
    cfg = load_config(ROOT/"configs/bt_self_flow/quantization_residual/c_sf.json")
    fallback = load_config(ROOT/"configs/bt_self_flow/quantization_residual/c_sf_b4a2.json")
    assert 2*cfg["train"]["batch_size"]*cfg["train"]["accumulation_steps"] == 16
    assert 2*fallback["train"]["batch_size"]*fallback["train"]["accumulation_steps"] == 16
    for cycle in range(5):
        assert sorted(balanced_quality(cfg, i) for i in range(cycle*6, (cycle+1)*6)) == cfg["quality"]["values"]
    mask, ramp = training_mask(cfg, 12000, 0, 0, 8, "cpu")
    assert ramp == 1. and mask.flatten(1).any(1).sum() == 4
    assert torch.all(mask.flatten(1).sum(1)[mask.flatten(1).any(1)] == 16)


def ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{directory}/rendezvous", timeout=timedelta(seconds=120))
    try:
        model = tiny_residual()
        model._ddp_params_and_buffers_to_ignore = [n for n, _ in model.named_parameters() if n.endswith(".quantiles")]
        wrapped = DDP(model, broadcast_buffers=False, find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        teacher = None
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            if step == 1:
                model.activate_residual()
                teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
            image = torch.rand(2, 3, 256, 256)*2-1
            mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
            if rank == 0:
                mask[0, :, :2, :2] = True
            result = wrapped(image, RateControl(.4), sf_teacher=teacher,
                             sf_mask=mask if teacher is not None else None)
            loss = (result.image-image).square().mean() + .1*(result.coarse-result.clean).square().mean()
            loss = loss + sum(-likelihood.log().mean() for likelihood in result.likelihoods.values())
            if teacher is not None:
                loss = loss+.01*result.auxiliary["sf_loss"]
            loss.backward()
            optimizer.step()
        for name, p in model.named_parameters():
            if p.requires_grad and not name.endswith(".quantiles"):
                expected = p.detach().clone()
                dist.broadcast(expected, 0)
                torch.testing.assert_close(p, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_dormant_to_active_and_zero_valid_rank(tmp_path):
    mp.start_processes(ddp_worker, args=(str(tmp_path),), nprocs=2, start_method="fork", join=True)


def make_images(root):
    import numpy as np
    from PIL import Image
    (root/"images").mkdir()
    Image.fromarray(np.random.default_rng(99).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/a.png")


def test_real_mu_calibration_preserves_rng_and_gradients(tmp_path):
    make_images(tmp_path)
    model = tiny_residual()
    model.activate_residual()
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
    config = load_config(ROOT/"configs/bt_self_flow/quantization_residual/c_sf.json")
    config["self_flow"]["calibration_samples_per_q"] = 1
    rng = torch.get_rng_state().clone()
    report = calibrate_residual(model, teacher, ImageDataset([tmp_path/"images"], 256),
                                 ImageLoss(mse_weight=4., lpips_weight=0, dists_weight=0),
                                 config, 0, 1, torch.bfloat16)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert report["samples"] == [1.]*6
    assert 0 < report["alpha"] <= 10
    assert len(report["privileged_d411"]) == 6


@pytest.mark.parametrize("privileged_gate_passed", [True, False])
@pytest.mark.parametrize("fixed_weight", [None, 1.])
def test_real_trainer_transition_and_active_resume_match(tmp_path, privileged_gate_passed, fixed_weight):
    make_images(tmp_path)
    config = small_config(tmp_path, "joint")
    config["model"]["backend"] = "C"
    config["self_flow"] = dict(method="quantization_residual_v1", enabled=True, mixed_sampling=False,
                                warmup=1, ramp=1, cell_fraction=.25, gradient_target=.05)
    if fixed_weight is not None:
        config["self_flow"]["weight"] = fixed_weight
    config["quality"] = dict(values=[0., .2, .4, .6, .8, 1.], rate_coefficients=[16., 8., 4., 2., 1., .5])
    config["train"].update(steps=3, accumulation_steps=1, batch_size=2, router_ramp_steps=0)
    config["objective"] = dict(base_weight=.1, repaired_weight=0.)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    initial = tiny_residual()
    state = clone_state(initial.export_state(), cpu=True)
    del state["residual"]
    init = tmp_path/"backend_init.pt"
    save_atomic(init, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                           raw=state, config=config, assets={"test": "residual"}))
    def factory(cfg, device):
        model = tiny_residual().to(device)
        model.vae.requires_grad_(False)
        return model
    calibration = dict(alpha=.001, privileged_gate_passed=privileged_gate_passed, protocol="test_fixture")
    args = SimpleNamespace(resume=None, init_checkpoint=str(init), max_updates=None)
    with patch.object(trainer, "make_model", side_effect=factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "residual"}), \
         patch.object(trainer, "calibrate_residual", return_value=calibration), redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device("cpu"))
        split = deepcopy(config)
        split["train"]["output"] = str(tmp_path/"split")
        args.max_updates = 2
        trainer.train(split, args, 0, 1, torch.device("cpu"))
        prefix = read_checkpoint(tmp_path/"split/pre_sf.pt")
        assert prefix["step"] == 1 and "teacher" not in prefix
        assert not prefix["raw"]["residual"]["active"].item()
        active = read_checkpoint(tmp_path/"split/paused.pt")
        assert active["step"] == 2 and "teacher" in active
        assert active["raw"]["residual"]["active"].item()
        assert active["residual_sf"]["privileged_gate_passed"] is privileged_gate_passed
        assert active["rep_weight"] == (fixed_weight if fixed_weight is not None else calibration["alpha"])
        args.resume, args.init_checkpoint, args.max_updates = str(tmp_path/"split/paused.pt"), None, None
        trainer.train(split, args, 0, 1, torch.device("cpu"))
    full = read_checkpoint(tmp_path/"joint/complete.pt")
    resumed = read_checkpoint(tmp_path/"split/complete.pt")
    for part in ("raw", "ema"):
        for component, values in full[part].items():
            for name, tensor in values.items():
                assert torch.equal(tensor, resumed[part][component][name]), (part, component, name)
    for name, tensor in full["teacher"].items():
        assert torch.equal(tensor, resumed["teacher"][name]), name
    assert resumed["residual_sf"] == full["residual_sf"]
    if fixed_weight is not None:
        records = [json.loads(line) for line in (tmp_path/"split/train.jsonl").read_text().splitlines()]
        assert records[-1]["sf_weight"] == fixed_weight
        assert records[-1]["sf_loss_multiplier"] == fixed_weight/.5
