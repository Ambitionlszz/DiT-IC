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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.losses import ImageLoss
from scripts.launch_bt_lowrate_c import SingleCRun, resolved_configs
from test_bt_backend import tiny_model
from test_bt_training import small_config


def joint_worker(rank, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root/'rendezvous'}", timeout=timedelta(seconds=180))
    models = []

    def factory(config, device):
        model = tiny_model("C").to(device)
        model.vae.requires_grad_(False)
        models.append(model)
        return model

    try:
        config = small_config(root, "joint")
        config["model"]["backend"] = "C"
        config["image_loss"] = dict(base_weight=0., latent_weight=0.)
        config["train"].update(steps=3, batch_size=2, accumulation_steps=2, effective_batch=8,
                               eval_every=2, save_every=2, validation_actual_bpp=True, rate_weight=3.,
                               parallel_validation=True)
        args = SimpleNamespace(resume=None, init_checkpoint=str(root/"source.pt"), max_updates=None)
        with patch.object(trainer, "make_model", side_effect=factory), \
             patch.object(trainer, "asset_identity", return_value={"test": "ddp-C"}), \
             patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0, **kw)), \
             redirect_stdout(io.StringIO()):
            trainer.train(config, args, rank, 2, torch.device("cpu"))
            split = deepcopy(config)
            split["train"]["output"] = str(root/"split")
            args.max_updates = 2
            trainer.train(split, args, rank, 2, torch.device("cpu"))
            args.resume, args.max_updates = str(root/"split/paused.pt"), None
            args.init_checkpoint = None
            trainer.train(split, args, rank, 2, torch.device("cpu"))
        for parameter in models[-1].parameters():
            if parameter.requires_grad:
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(parameter, reference), "C DDP parameters diverged"
    finally:
        dist.destroy_process_group()


def test_no_aux_c_two_rank_accumulation_and_exact_resume(tmp_path):
    (tmp_path/"images").mkdir()
    Image.fromarray(np.random.default_rng(94).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(tmp_path/"images/one.png")
    config = small_config(tmp_path, "joint")
    config["model"]["backend"] = "C"
    source = tiny_model("C").eval()
    save_atomic(tmp_path/"source.pt", dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared",
                                          step=0, raw=source.export_state(), config=config, assets={"test": "ddp-C"}))
    mp.spawn(joint_worker, args=(str(tmp_path),), nprocs=2, join=True)
    full, split = (read_checkpoint(tmp_path/name/"complete.pt") for name in ("joint", "split"))
    assert split["step"] == 3
    assert split["micro_seen"] == full["micro_seen"] == 6
    assert len(split["rng_by_rank"]) == split["contract"]["world_size"] == 2
    assert split["scheduler"] == full["scheduler"]
    for kind in ("raw", "ema"):
        for part, values in full[kind].items():
            for key, value in values.items():
                assert torch.equal(value, split[kind][part][key]), f"{kind}.{part}.{key}"


def test_production_configs_and_both_stages_use_two_gpus(tmp_path):
    configs = resolved_configs(tmp_path)
    common, joint = configs["common"], configs["c_joint"]
    assert common["train"]["steps"] == 50000
    assert joint["train"]["steps"] == 200000
    assert common["model"]["codec"]["delta_coarse"] == 8
    assert common["model"]["codec"]["delta_fine"] == 1
    assert common["train"]["rate_weight"] == joint["train"]["rate_weight"] == 3
    for config in (common, joint):
        assert not config["self_flow"]["enabled"]
        assert config["model"]["backend"] == "C"
    run = SingleCRun(tmp_path, ["GPU-a", "GPU-b"], "python")
    for name, config in configs.items():
        command = run.command(name, Path(config["train"]["output"]), distributed=True)
        assert "--nproc_per_node=2" in command
    assert joint["train"]["effective_batch"] == 2*joint["train"]["batch_size"]*joint["train"]["accumulation_steps"] == 8


def test_single_c_supervisor_handoff_and_resume_commands(tmp_path):
    (tmp_path/"logs").mkdir()
    (tmp_path/"configs").mkdir()
    run = SingleCRun(tmp_path, ["GPU-a", "GPU-b"], "python")
    commands = []

    def start(name, command, gpus):
        commands.append((name, command, gpus))
        directory = run.directory(name)
        directory.mkdir(parents=True, exist_ok=True)
        (tmp_path/"backend_init.pt" if name == "prepare" else directory/"complete.pt").touch()

    with patch.object(run, "start", side_effect=start), patch.object(run, "wait"):
        run.execute()
    assert [name for name, _, _ in commands] == ["common", "prepare", "c_joint"]
    assert commands[0][2] == commands[2][2] == ["GPU-a", "GPU-b"]
    assert "--nproc_per_node=2" in commands[0][1] and "--nproc_per_node=2" in commands[2][1]
    assert commands[2][1][-2:] == ["--init-checkpoint", str(tmp_path/"backend_init.pt")]
    assert json.loads((tmp_path/"pipeline.json").read_text())["status"] == "completed"
    (tmp_path/"C/joint/complete.pt").unlink()
    (tmp_path/"C/joint/paused.pt").touch()
    commands.clear()
    with patch.object(run, "start", side_effect=start), patch.object(run, "wait"):
        run.execute()
    assert len(commands) == 1
    assert commands[0][1][-2:] == ["--resume", str(tmp_path/"C/joint/paused.pt")]
