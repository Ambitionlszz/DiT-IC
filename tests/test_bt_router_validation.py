from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import DeploymentEMA, evaluation_state, read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.validation import merge_validation_reports, validate, validate_distributed
from test_bt_training import make_tiny_stage, small_config


def validation_worker(rank, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root/'rendezvous'}", timeout=timedelta(seconds=180))
    try:
        model = make_tiny_stage(small_config(root), torch.device("cpu"))
        model.codec.router_strength.fill_(.5)
        dataset = ImageDataset([root/"images"], 256, False)
        expected = validate(model, dataset, actual_rate=True)
        actual = validate_distributed(model, dataset, actual_rate=True)
        for quality in expected:
            assert actual[quality] == pytest.approx(expected[quality], rel=2e-7)
        assert float(model.codec.router_strength) == .5
        if rank == 1:
            def broken(*args, **kwargs):
                raise RuntimeError("injected coding error")
            model.codec.compress = broken
        with pytest.raises(RuntimeError, match="rank 1: RuntimeError: injected coding error"):
            validate_distributed(model, dataset, actual_rate=True)
        if rank == 0:
            (root/"parallel.json").write_text(json.dumps(actual))
    finally:
        dist.destroy_process_group()


def test_parallel_validation_matches_sequential_for_uneven_images_and_shares_failures(tmp_path):
    (tmp_path/"images").mkdir()
    rng = np.random.default_rng(92)
    for index in range(3):
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path/f"images/{index}.png")
    mp.spawn(validation_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert len(json.loads((tmp_path/"parallel.json").read_text())) == 6


def test_parallel_packet_metrics_use_checked_packet_counts_and_global_max():
    parts = [dict(images=2, packets=2, report={"0.0": dict(loss=3., packet_bpp=2., packet_forward_max_error=1e-5)}),
             dict(images=1, packets=1, report={"0.0": dict(loss=6., packet_bpp=8., packet_forward_max_error=2e-5)})]
    assert merge_validation_reports(parts)["0.0"] == dict(loss=4., packet_bpp=4., packet_forward_max_error=2e-5)


@pytest.mark.parametrize("strength", [0., .5, 1.])
@pytest.mark.parametrize("use_ema", [False, True])
def test_real_packet_validation_during_router_ramp_restores_training(tmp_path, strength, use_ema):
    images = tmp_path/"images"
    images.mkdir()
    Image.fromarray(np.random.default_rng(91).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(images/"one.png")
    model = make_tiny_stage(small_config(tmp_path), torch.device("cpu")).train()
    model.codec.router_strength.fill_(strength)
    state = DeploymentEMA(model).state if use_ema else None
    clean = model.encode_clean_latent(torch.zeros(1, 3, 256, 256))
    model.codec.eval().update(force=True)
    if strength != 1:
        with pytest.raises(ValueError, match="unfinished Router warmup"):
            model.codec.compress(clean, RateControl(.5))
    model.train()
    original_compress = model.codec.compress
    calls = []

    def compress(*args, **kwargs):
        calls.append(float(model.codec.router_strength))
        return original_compress(*args, **kwargs)

    with patch.object(model.codec, "compress", side_effect=compress):
        report = validate(model, ImageDataset([images], 256, False), state=state, actual_rate=True)
    assert calls == [1.]*6
    assert all(point["rate_actual"] > 0 for point in report.values())
    assert model.training
    assert float(model.codec.router_strength) == strength
    if state is not None:
        assert float(state["codec"]["router_strength"]) == strength


@pytest.mark.parametrize("use_ema", [False, True])
def test_failed_deployment_validation_also_restores_router(tmp_path, use_ema):
    model = make_tiny_stage(small_config(tmp_path), torch.device("cpu")).train()
    model.codec.router_strength.fill_(.25)
    state = DeploymentEMA(model).state if use_ema else None
    with pytest.raises(RuntimeError, match="injected"):
        with evaluation_state(model, state, deployment_router=True):
            assert not model.training
            assert float(model.codec.router_strength) == 1.
            raise RuntimeError("injected coding failure")
    assert model.training
    assert float(model.codec.router_strength) == .25


def test_resume_across_first_validation_and_router_ramp_matches_uninterrupted(tmp_path):
    images = tmp_path/"images"
    images.mkdir()
    Image.fromarray(np.random.default_rng(93).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(images/"one.png")
    config = small_config(tmp_path)
    config["train"].update(steps=6, eval_every=2, save_every=2, validation_actual_bpp=True,
                           router_uniform_steps=2, router_ramp_steps=2)
    args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
    with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
         patch.object(trainer, "asset_identity", return_value={"test": "router-ramp"}), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device("cpu"))
        split = deepcopy(config)
        split["train"]["output"] = str(tmp_path/"split")
        args.max_updates = 2
        trainer.train(split, args, 0, 1, torch.device("cpu"))
        paused = read_checkpoint(tmp_path/"split/paused.pt")
        assert paused["step"] == 2
        assert float(paused["raw"]["codec"]["router_strength"]) == 0.
        assert float(paused["ema"]["codec"]["router_strength"]) == 0.
        args.resume, args.max_updates = str(tmp_path/"split/paused.pt"), None
        trainer.train(split, args, 0, 1, torch.device("cpu"))
    full, resumed = (read_checkpoint(tmp_path/f"{name}/complete.pt") for name in ("codec", "split"))
    assert resumed["step"] == 6
    assert resumed["micro_seen"] == full["micro_seen"]
    assert resumed["calibration"] == full["calibration"]
    for kind in ("raw", "ema"):
        for part, values in full[kind].items():
            for key, value in values.items():
                assert torch.equal(value, resumed[kind][part][key]), f"{kind}.{part}.{key}"
