from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import itertools
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.data import CounterSampler
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.model import BTCodec, BTCodecStage
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.scale import DistortionScale
from test_bt_backend import TinyAE, tiny_model, tiny_sana


torch.set_num_threads(1)


def make_tiny_stage(config, device):
    torch.manual_seed(config["train"]["seed"])
    model = BTCodecStage.__new__(BTCodecStage)
    nn.Module.__init__(model)
    model.vae = TinyAE().requires_grad_(False).eval()
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4)
    model.distortion_scale = DistortionScale()
    model.backend = "C"
    return model.to(device)


def small_config(root, phase="codec"):
    return {"data": {"train": [str(root/"images")], "valid": [str(root/"images")],
                     "crop": 256, "workers": 0, "valid_limit": 1},
            "model": {"backend": "C" if phase == "codec" else "D"},
            "self_flow": {"enabled": phase == "self_flow", "mixed_sampling": phase == "self_flow",
                          "warmup": 0, "ramp": 1, "calibrate_every": 1},
            "optimizer": {"codec_lr": 1e-4, "joint_codec_lr": 5e-5, "backend_lr": 1e-4,
                          "vae_lr": 1e-5, "aux_lr": 1e-3, "warmup": 1, "max_grad_norm": 4},
            "train": {"phase": phase, "steps": 2, "batch_size": 1, "accumulation_steps": 2,
                      "seed": 903, "eval_every": 100, "save_every": 100, "log_every": 1,
                      "data_scale_samples": 1, "calibration_samples": 1,
                      "router_uniform_steps": 0, "router_ramp_steps": 1,
                      "output": str(root/phase)}}


class OneOverflowScaler:
    """Inject one skipped optimizer update to check data replay on CPU."""
    def __init__(self, *args, **kwargs):
        self.value, self.attempt = 64., 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if self.attempt:
            optimizer.step()

    def update(self):
        if not self.attempt:
            self.value /= 2
        self.attempt += 1

    def state_dict(self):
        return {"scale": self.value, "attempt": self.attempt}


def ddp_worker(rank, directory, slow_checkpoint=False):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{root/'rendezvous'}",
                            timeout=timedelta(seconds=3 if slow_checkpoint else 300))
    models = []
    def factory(config, device):
        model = make_tiny_stage(config, device)
        models.append(model)
        return model
    def slow_save(path, state):
        if slow_checkpoint:
            time.sleep(4)  # Longer than the training collective timeout.
        return save_atomic(path, state)
    try:
        with patch.object(trainer, "make_model", side_effect=factory), \
             patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
             patch.object(trainer, "save_atomic", side_effect=slow_save), \
             redirect_stdout(io.StringIO()):
            trainer.train(small_config(root), SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None),
                          rank, 2, torch.device("cpu"))
        # Both the main reducer and the manually synchronized quantile optimizer
        # must leave every trainable parameter identical on the two ranks.
        for parameter in models[0].parameters():
            if parameter.requires_grad:
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                if not torch.equal(reference, parameter):
                    raise AssertionError("DDP parameters diverged after accumulated main/aux updates")
    finally:
        dist.destroy_process_group()


class TrainingTest(unittest.TestCase):
    def test_amp_retry_preserves_paired_data_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            rng = np.random.default_rng(905)
            for index in range(3):
                Image.fromarray(rng.integers(0, 256, (270, 270, 3), dtype=np.uint8)).save(root/f"images/{index}.png")
            args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
            with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
                 patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
                 redirect_stdout(io.StringIO()):
                reference = small_config(root)
                trainer.train(reference, args, 0, 1, torch.device("cpu"))
                skipped = small_config(root)
                skipped["train"]["output"] = str(root/"overflow")
                with patch.object(trainer.torch.amp, "GradScaler", OneOverflowScaler):
                    trainer.train(skipped, args, 0, 1, torch.device("cpu"))
            full = read_checkpoint(root/"codec/complete.pt")
            retry = read_checkpoint(root/"overflow/complete.pt")
            self.assertEqual(retry["attempts"], full["attempts"]+1)
            self.assertEqual(retry["micro_seen"], full["micro_seen"])
            for weights in ("raw", "ema"):
                for part, values in full[weights].items():
                    for name, value in values.items():
                        self.assertTrue(torch.equal(value, retry[weights][part][name]), f"{weights}.{part}.{name}")

    def test_two_rank_accumulation_and_auxiliary_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
            mp.spawn(ddp_worker, args=(directory,), nprocs=2, join=True)
            state = read_checkpoint(root/"codec/complete.pt")
            self.assertEqual(state["step"], 2)
            self.assertEqual(len(state["rng_by_rank"]), 2)
            self.assertEqual(state["contract"]["world_size"], 2)

    def test_slow_rank_zero_save_does_not_timeout_training_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
            mp.spawn(ddp_worker, args=(directory, True), nprocs=2, join=True)
            self.assertEqual(read_checkpoint(root/"codec/complete.pt")["step"], 2)

    def test_validation_selects_best_and_resume_retains_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
            config = small_config(root)
            config["train"].update(steps=3, eval_every=1, save_every=1)
            qualities = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
            reports = [{quality: {"loss": loss} for quality in qualities} for loss in (3., 1., 2.)]
            args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=2)
            with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
                 patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
                 patch.object(trainer, "validate", side_effect=reports), redirect_stdout(io.StringIO()):
                trainer.train(config, args, 0, 1, torch.device("cpu"))
                args.resume, args.max_updates = str(root/"codec/paused.pt"), None
                trainer.train(config, args, 0, 1, torch.device("cpu"))
            self.assertTrue(all((root/f"codec/{step:07d}.pt").exists() for step in (1, 2, 3)))
            self.assertEqual(read_checkpoint(root/"codec/best.pt")["step"], 2)
            final = read_checkpoint(root/"codec/complete.pt")
            self.assertEqual(final["step"], 3)
            self.assertEqual(final["validation_best"]["step"], 2)
            self.assertEqual(final["validation_best"]["score"], 1.)

    def test_cross_rate_trainer_runs_both_global_and_mixed_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.random.default_rng(904).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
            config = small_config(root, "self_flow")
            source_config = deepcopy(config)
            source_config["train"]["phase"] = "joint"
            source = tiny_model("D").eval()
            source.vae.requires_grad_(False)
            save_atomic(root/"source.pt", {"method": METHOD, "schema": SCHEMA, "phase": "joint", "backend": "D",
                                          "step": 2, "config": source_config, "raw": source.export_state(),
                                          "assets": {"test": "tiny-frozen-AE"}})
            def factory(cfg, device):
                model = tiny_model("D", representation=True).to(device)
                model.vae.requires_grad_(False)
                return model
            with patch.object(trainer, "make_model", side_effect=factory), \
                 patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
                 patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0)), \
                 redirect_stdout(io.StringIO()):
                trainer.train(config, SimpleNamespace(resume=None, init_checkpoint=str(root/"source.pt"), max_updates=None),
                              0, 1, torch.device("cpu"))
            saved = read_checkpoint(root/"self_flow/complete.pt")
            self.assertTrue(saved["teacher"])
            self.assertIn("projector", saved["raw"])
            records = [json.loads(line) for line in (root/"self_flow/train.jsonl").read_text().splitlines()]
            self.assertEqual([row["mixed"] for row in records], [False, True])
            self.assertTrue(all("sf_weight" in row and "teacher_valid" in row for row in records))

    def test_counter_sampler_resumes_without_worker_rng(self):
        full = list(itertools.islice(CounterSampler(91, 903, rank=1, world_size=2, batch_size=2), 12))
        resumed = list(itertools.islice(CounterSampler(91, 903, rank=1, world_size=2, batch_size=2,
                                                        micro_batches_seen=3), 6))
        self.assertEqual(full[6:], resumed)

    def test_recipes_keep_the_paired_factors_fixed(self):
        root = Path(__file__).resolve().parents[1]/"configs"/"bt_self_flow"
        for phase in ("backend", "joint"):
            c, d = load_config(root/f"c_{phase}.yaml"), load_config(root/f"d_{phase}.yaml")
            d["model"]["backend"] = c["model"]["backend"]
            d["train"]["output"] = c["train"]["output"]
            self.assertEqual(c, d)
        d, e = load_config(root/"d_mix.yaml"), load_config(root/"e_self_flow.yaml")
        e["self_flow"]["enabled"] = False
        e["train"]["output"] = d["train"]["output"]
        self.assertEqual(d, e)

    def test_real_trainer_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root/"images"
            images.mkdir()
            rng = np.random.default_rng(903)
            for i in range(4):
                Image.fromarray(rng.integers(0, 256, (260, 260, 3), dtype=np.uint8)).save(images/f"{i}.png")
            config = {"data": {"train": [str(images)], "valid": [str(images)], "crop": 256,
                               "workers": 0, "valid_limit": 2},
                      "model": {"backend": "C"}, "self_flow": {"enabled": False, "mixed_sampling": False},
                      "optimizer": {"codec_lr": 1e-4, "joint_codec_lr": 5e-5, "backend_lr": 1e-4,
                                    "vae_lr": 1e-5, "aux_lr": 1e-3, "warmup": 1, "max_grad_norm": 4},
                      "train": {"phase": "codec", "steps": 4, "batch_size": 1, "accumulation_steps": 2,
                                "seed": 903, "eval_every": 100, "save_every": 100, "log_every": 100,
                                "data_scale_samples": 2, "calibration_samples": 2,
                                "router_uniform_steps": 0, "router_ramp_steps": 1}}
            arguments = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
            with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
                 patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
                 redirect_stdout(io.StringIO()):
                config["train"]["output"] = str(root/"full")
                trainer.train(deepcopy(config), arguments, 0, 1, torch.device("cpu"))
                config["train"]["output"] = str(root/"split")
                arguments.max_updates = 2
                trainer.train(deepcopy(config), arguments, 0, 1, torch.device("cpu"))
                arguments.resume = str(root/"split"/"paused.pt")
                arguments.max_updates = None
                trainer.train(deepcopy(config), arguments, 0, 1, torch.device("cpu"))
            full, split = read_checkpoint(root/"full"/"complete.pt"), read_checkpoint(root/"split"/"complete.pt")
            for weights in ("raw", "ema"):
                for part in full[weights]:
                    for name, value in full[weights][part].items():
                        self.assertTrue(torch.equal(value, split[weights][part][name]), f"{weights}.{part}.{name}")
            self.assertEqual(full["step"], split["step"])
            self.assertEqual(full["micro_seen"], split["micro_seen"])
            self.assertEqual(full["scheduler"], split["scheduler"])
            for key, values in full["optimizer"]["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, split["optimizer"]["state"][key][name]))

    def test_real_wrapper_shares_initialization_and_builds_no_gan(self):
        from diffusers import AutoencoderDC

        def vae_factory(*args, **kwargs):
            return AutoencoderDC(latent_channels=4, encoder_block_out_channels=(16, 16),
                                 decoder_block_out_channels=(16, 16), encoder_layers_per_block=(1, 1),
                                 decoder_layers_per_block=(1, 1), encoder_qkv_multiscales=((), ()),
                                 decoder_qkv_multiscales=((), ()), attention_head_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory)/"text.pt"
            torch.save({"fixed_text_embed": torch.randn(1, 2, 12), "hidden_size": 12}, text)
            scheduler = SimpleNamespace(sigmas=torch.tensor([.7]), timesteps=torch.tensor([700.]))
            models = []
            with patch("models.DiT_IC.AutoencoderDC.from_pretrained", side_effect=vae_factory), \
                 patch("models.DiT_IC.SanaTransformer2DModel.from_pretrained", side_effect=lambda *a, **k: tiny_sana()), \
                 patch("models.DiT_IC.make_1step_sched", return_value=scheduler), redirect_stdout(io.StringIO()):
                for backend in ("C", "D"):
                    torch.manual_seed(903)
                    models.append(BTCodec("unused", str(text), backend=backend,
                                           codec_params={"ch_emd": 4, "channel": 16, "channel_out": 4},
                                           dit_rank=2, vae_rank=2, student_layer=1, teacher_layer=3))
            c, d = [model.export_state() for model in models]
            for part in c:
                for name, value in c[part].items():
                    self.assertTrue(torch.equal(value, d[part][name]), f"{part}.{name}")
            for model in models:
                self.assertIsNone(model.loss)
                self.assertFalse(any(p.requires_grad for p in model.vae.get_base_model().encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
