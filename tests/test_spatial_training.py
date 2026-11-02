"""Real trainer continuation and two-rank, batch4/accum1 spatial SF checks."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
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
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.experiment import optimizer_groups
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.spatial_self_flow import SpatialEMATeacher
from models.bt_self_flow.spatial_training import average_gradients, spatial_backward
from test_spatial_self_flow import tiny_spatial_config, tiny_spatial_model


ASSETS = {"test": "real-tiny-SANA-codec-AE"}


def setup_run(root, variant="c_d_sf", world=1, batch=1):
    root = Path(root)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(19)
    for index in range(2):
        Image.fromarray(generator.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(images / f"{index}.png")
    config = tiny_spatial_config(variant)
    config["data"].update(train=[str(images)], valid=[str(images)], crop=256, valid_crop=256, valid_limit=1, workers=0)
    config["train"].update(world_size=world, batch_size=batch, effective_batch=world * batch,
                           steps=2, eval_every=100, save_every=100, full_decode_every=0,
                           packet_samples=0, log_every=1, output=str(root / "run"))
    pure = tiny_spatial_config("c")
    model = tiny_spatial_model(pure)
    state = clone_state(model.export_state(), cpu=True)
    source = root / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0, raw=state, ema=state,
                             config=pure, assets=ASSETS, calibration={"test": True}))
    return config, source


def trainer_args(source, resume=None, stop=None):
    return SimpleNamespace(init_checkpoint=str(source) if resume is None else None,
                           resume=str(resume) if resume is not None else None, max_updates=stop)


def check_state_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            check_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            check_state_equal(x, y)
    else:
        assert a == b


def test_real_training_resumes_exact_raw_ema_optimizer_and_scheduler(tmp_path):
    config, source = setup_run(tmp_path)
    with patch.object(trainer, "make_model", side_effect=tiny_spatial_model), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source), 0, 1, torch.device("cpu"))
        config["train"]["output"] = str(tmp_path / "split")
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
        paused = tmp_path / "split" / "paused.pt"
        trainer.train(config, trainer_args(source, resume=paused), 0, 1, torch.device("cpu"))
    full = read_checkpoint(tmp_path / "run" / "complete.pt")
    split = read_checkpoint(tmp_path / "split" / "complete.pt")
    for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "scaler", "rng_by_rank"):
        check_state_equal(full[key], split[key])
    assert full["step"] == split["step"] == 2
    assert "spatial_projector" in full["raw"] and "pixel_adapter" in full["raw"]


@pytest.mark.parametrize("variant", ["c", "c_d", "c_p", "c_d_p"])
def test_no_sf_control_never_constructs_teacher_and_updates_through_real_trainer(tmp_path, variant):
    config, source = setup_run(tmp_path, variant)
    with patch.object(trainer, "make_model", side_effect=tiny_spatial_model), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), \
         patch("models.bt_self_flow.spatial_self_flow.SpatialEMATeacher", side_effect=AssertionError("teacher in no-SF control")), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
    state = read_checkpoint(tmp_path / "run" / "paused.pt")
    assert state["step"] == 1
    assert ("spatial_projector" in state["raw"]) == variant.endswith("p")
    assert ("pixel_adapter" in state["raw"]) == ("_d" in variant)


def spatial_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root / 'rendezvous'}", timeout=timedelta(seconds=120))
    models = []
    def factory(config, device):
        model = tiny_spatial_model(config, device)
        models.append(model)
        return model
    def activity(settings, batch, seed, step, local_rank, micro, device):
        # First update has no active P globally. Second update has four
        # active samples on rank 1 and none on rank 0.
        return torch.full((batch,), step == 1 and local_rank == 1, dtype=torch.bool, device=device)
    try:
        config = __import__("json").loads((root / "ddp.json").read_text())
        with patch.object(trainer, "make_model", side_effect=factory), \
             patch.object(trainer, "asset_identity", return_value=ASSETS), \
             patch("models.bt_self_flow.spatial_training.active_samples", side_effect=activity), \
             redirect_stdout(io.StringIO()):
            trainer.train(config, trainer_args(root / "shared.pt"), rank, 2, torch.device("cpu"))
        assert config["train"]["batch_size"] == 4 and config["train"]["accumulation_steps"] == 1
        for parameter in models[0].parameters():
            if parameter.requires_grad:
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(reference, parameter), "parameters diverged across ranks"
        # Also compare SF's uneven-count distributed gradient with a single
        # global batch. This isolates the global active-element denominator.
        config["model"]["codec"]["rate_likelihood"] = "hard"
        model = tiny_spatial_model(config).train()
        model.codec.rate_likelihood = "hard"
        teacher = SpatialEMATeacher(model)
        teacher.load_deployment_ema(trainer.DeploymentEMA(model).state)
        image = torch.randn(8, 3, 256, 256, generator=torch.Generator().manual_seed(29))
        image_loss = ImageLoss(**config["image_loss"])
        scale = torch.amp.GradScaler("cuda", enabled=False)
        with patch("models.bt_self_flow.spatial_training.active_samples", side_effect=activity), \
             patch("models.bt_self_flow.spatial_training.higher_quality", return_value=RateControl(1.)):
            spatial_backward(model, teacher, image[rank * 4:(rank + 1) * 4], RateControl(.2), image_loss,
                             config, 1, rank, 2, scale, torch.bfloat16)
        groups, _ = optimizer_groups(model, config)
        average_gradients([p for group in groups for p in group["params"]], 2)
        actual = model.spatial_projector.conv.weight.grad.clone()
        if rank == 0:
            reference = tiny_spatial_model(config).train()
            reference.codec.rate_likelihood = "hard"
            target = SpatialEMATeacher(reference)
            target.load_deployment_ema(trainer.DeploymentEMA(reference).state)
            mask = torch.tensor([False] * 4 + [True] * 4)
            with patch("models.bt_self_flow.spatial_training.active_samples", return_value=mask), \
                 patch("models.bt_self_flow.spatial_training.higher_quality", return_value=RateControl(1.)):
                spatial_backward(reference, target, image, RateControl(.2), image_loss,
                                 config, 1, 0, 1, scale, torch.bfloat16)
            torch.testing.assert_close(actual, reference.spatial_projector.conv.weight.grad, atol=2e-5, rtol=2e-4)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_process_batch4_accum1_handles_empty_rank_and_global_mean(tmp_path):
    config, _ = setup_run(tmp_path, world=2, batch=4)
    (tmp_path / "ddp.json").write_text(__import__("json").dumps(config))
    mp.spawn(spatial_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_wrong_world_size_fails_before_model_allocation(tmp_path):
    config = tiny_spatial_config()
    config["train"]["world_size"] = 2
    with patch.object(trainer, "make_model", side_effect=AssertionError("model allocated")):
        with pytest.raises(ValueError, match="world size"):
            trainer.train(config, trainer_args(tmp_path / "unused"), 0, 1, torch.device("cpu"))
