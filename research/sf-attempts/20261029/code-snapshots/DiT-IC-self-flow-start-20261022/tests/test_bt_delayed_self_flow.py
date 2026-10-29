"""Exercise the RD-to-SF boundary, exact resumes and two-rank synchronization."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.model import BTCrossRateTeacher, cross_rate_loss
from test_bt_backend import tiny_model
from test_bt_training import small_config


def config_for(root, output="full"):
    config = small_config(root, "joint")
    config["self_flow"].update(enabled=True, mixed_sampling=True, start_step=2,
                               mixed_every=2, high_fraction=.25, ramp=2, ema_decay=.9999)
    config["train"].update(steps=6, output=str(root/output))
    config["objective"] = {"base_weight": .1, "repaired_weight": 0.}
    config["image_loss"] = {"mse_weight": 4., "lpips_weight": 1., "dists_weight": 1.}
    return config


def make_source(root):
    (root/"images").mkdir(exist_ok=True)
    rng = np.random.default_rng(904)
    for index in range(3):
        Image.fromarray(rng.integers(0, 256, (270, 270, 3), dtype=np.uint8)).save(root/f"images/{index}.png")
    model = tiny_model("C").eval()
    model.vae.requires_grad_(False)
    config = config_for(root)
    save_atomic(root/"source.pt", dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared",
                                     step=0, config=config, raw=model.export_state(), assets={"test": "delayed"}))


def factory(config, device):
    model = tiny_model("D", representation=True).to(device)
    model.vae.requires_grad_(False)
    return model


def nonempty_sf(student, teacher, teacher_rms, student_rms):
    # Guarantee a nonempty target when checking gradient/update behavior; the
    # production gate is separately exercised by the real tiny codec tests.
    return cross_rate_loss(student, teacher, torch.zeros_like(teacher_rms), torch.ones_like(student_rms))


def ddp_delayed_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root/'rendezvous'}", timeout=timedelta(seconds=180))
    models, teachers = [], []
    def record_model(config, device):
        model = factory(config, device)
        models.append(model)
        return model
    def record_teacher(*args, **kwargs):
        teacher = BTCrossRateTeacher(*args, **kwargs)
        teachers.append(teacher)
        return teacher
    try:
        config = config_for(root, "ddp")
        config["train"].update(batch_size=2, effective_batch=8)
        arguments = SimpleNamespace(resume=None, init_checkpoint=str(root/"source.pt"), max_updates=4)
        with patch.object(trainer, "make_model", side_effect=record_model), \
             patch.object(trainer, "BTCrossRateTeacher", side_effect=record_teacher), \
             patch.object(trainer, "asset_identity", return_value={"test": "delayed"}), \
             patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0)), \
             patch.object(trainer, "cross_rate_loss", side_effect=nonempty_sf), redirect_stdout(io.StringIO()):
            trainer.train(config, arguments, rank, 2, torch.device("cpu"))
            arguments.resume, arguments.init_checkpoint, arguments.max_updates = str(root/"ddp/paused.pt"), None, None
            trainer.train(config, arguments, rank, 2, torch.device("cpu"))
        assert len(teachers) == 2  # Activation once, then restore the saved teacher.
        for model in models:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, 0)
                    assert torch.equal(parameter, reference)
        for teacher in teachers:
            for value in teacher.get_ema_state_dict().values():
                reference = value.detach().clone()
                dist.broadcast(reference, 0)
                assert torch.equal(value, reference)
    finally:
        dist.destroy_process_group()


class DelayedSelfFlowTest(unittest.TestCase):
    def test_amp_calibration_preserves_small_auxiliary_gradient(self):
        latent = torch.ones(32, requires_grad=True)
        loss = latent.half().float().square().mean()*1e-8
        self.assertEqual(trainer.interface_gradient_norm(loss, latent).item(), 0.)
        scaled = trainer.interface_gradient_norm(loss, latent, 65536.)
        expected = torch.full_like(latent, 2e-8/32).norm()
        self.assertTrue(torch.allclose(scaled, expected, rtol=.002, atol=0.))

    def test_calibration_backward_preserves_fp32_inside_forward_autocast(self):
        generator = torch.Generator().manual_seed(904)
        latent = torch.randn(32, 32, generator=generator, requires_grad=True)
        matrix = torch.randn(32, 32, generator=generator)
        loss = (latent @ matrix).square().mean()
        expected = torch.autograd.grad(loss, latent, retain_graph=True)[0].norm()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            measured = trainer.interface_gradient_norm(loss, latent, 256.)
        self.assertTrue(torch.allclose(measured, expected, rtol=1e-6, atol=0.))

    def test_sampling_changes_at_boundary_without_changing_base_q_stream(self):
        config = config_for(Path("/unused"))
        plain = deepcopy(config)
        plain["self_flow"].update(enabled=False, mixed_sampling=False)
        for step in range(6):
            control = trainer.sample_control(config, step, 0, (2, 8, 8), "cpu")
            ordinary = trainer.sample_control(plain, step, 0, (2, 8, 8), "cpu")
            self.assertEqual(control.value, ordinary.value)
            self.assertEqual(control.mixed, step in (2, 4))
            if control.mixed:
                self.assertEqual(control.high_mask.sum().item(), 2*16)
                other = trainer.sample_control(config, step, 1, (2, 8, 8), "cpu")
                self.assertFalse(torch.equal(control.high_mask, other.high_mask))

    def test_teacher_birth_and_resume_across_both_sides_of_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root)
            initial_projector = deepcopy(tiny_model("D", representation=True).export_state()["projector"])
            births = []
            def record_teacher(*args, **kwargs):
                teacher = BTCrossRateTeacher(*args, **kwargs)
                births.append(deepcopy(teacher.get_ema_state_dict()))
                return teacher
            with patch.object(trainer, "make_model", side_effect=factory), \
                 patch.object(trainer, "BTCrossRateTeacher", side_effect=record_teacher), \
                 patch.object(trainer, "asset_identity", return_value={"test": "delayed"}), \
                 patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0)), \
                 patch.object(trainer, "cross_rate_loss", side_effect=nonempty_sf), redirect_stdout(io.StringIO()):
                args = SimpleNamespace(resume=None, init_checkpoint=str(root/"source.pt"), max_updates=None)
                trainer.train(config_for(root), args, 0, 1, torch.device("cpu"))
                args.max_updates = 2
                trainer.train(config_for(root, "split"), args, 0, 1, torch.device("cpu"))
                prefix = read_checkpoint(root/"split/paused.pt")
                self.assertNotIn("teacher", prefix)
                self.assertIsNone(prefix["teacher_initialized_step"])
                for name, value in initial_projector.items():
                    self.assertTrue(torch.equal(value, prefix["raw"]["projector"][name]), name)
                args.resume, args.init_checkpoint, args.max_updates = str(root/"split/paused.pt"), None, 4
                trainer.train(config_for(root, "split"), args, 0, 1, torch.device("cpu"))
                active = read_checkpoint(root/"split/paused.pt")
                self.assertEqual(active["teacher_initialized_step"], 2)
                self.assertTrue(any(not torch.equal(value, active["raw"]["projector"][name])
                                    for name, value in initial_projector.items()))
                args.max_updates = None
                trainer.train(config_for(root, "split"), args, 0, 1, torch.device("cpu"))
            # The first teacher really starts from the student after its two
            # RD updates, including all conditioning/text weights.
            for name, value in births[0].items():
                if name.startswith("quality_conditioner."):
                    expected = prefix["raw"]["conditioner"][name.removeprefix("quality_conditioner.")]
                    self.assertTrue(torch.equal(value, expected), name)
                elif name.startswith("text_projection."):
                    self.assertTrue(torch.equal(value, prefix["raw"]["text"][name.removeprefix("text_projection.")]))
            full, split = read_checkpoint(root/"full/complete.pt"), read_checkpoint(root/"split/complete.pt")
            self.assertEqual(full["scheduler"], split["scheduler"])
            self.assertEqual(full["micro_seen"], split["micro_seen"])
            self.assertEqual(full["rep_weight"], split["rep_weight"])
            for weights in ("raw", "ema"):
                for part, values in full[weights].items():
                    for name, value in values.items():
                        self.assertTrue(torch.equal(value, split[weights][part][name]), f"{weights}.{part}.{name}")
            for name, value in full["teacher"].items():
                self.assertTrue(torch.equal(value, split["teacher"][name]), name)
            rows = [json.loads(line) for line in (root/"full/train.jsonl").read_text().splitlines()]
            self.assertEqual([row["sf_active"] for row in rows], [False, False, True, True, True, True])
            self.assertEqual([row["mixed"] for row in rows], [False, False, True, False, True, False])
            self.assertGreater(rows[2]["sf_weight"], 0)
            self.assertEqual([row.get("sf_ramp") for row in rows[2:]], [.5, 1., 1., 1.])
            self.assertTrue(all(row["weighted_repaired"] == 0 for row in rows))

    def test_two_rank_transition_and_active_teacher_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root)
            mp.spawn(ddp_delayed_worker, args=(directory,), nprocs=2, join=True)
            checkpoint = read_checkpoint(root/"ddp/complete.pt")
            self.assertEqual(checkpoint["step"], 6)
            self.assertEqual(checkpoint["teacher_initialized_step"], 2)
            self.assertEqual(len(checkpoint["rng_by_rank"]), 2)
            self.assertEqual(checkpoint["contract"]["world_size"], 2)
            self.assertGreater(checkpoint["rep_weight"], 0)
            rows = [json.loads(line) for line in (root/"ddp/train.jsonl").read_text().splitlines()]
            self.assertGreater(rows[2]["sf_weight"], 0)
            self.assertGreater(rows[2]["sf_gradient_norm"], 0)

    def test_production_recipe_matches_agreed_objective_and_batch(self):
        path = Path(__file__).resolve().parents[1]/"configs/bt_self_flow/joint_sf_after20k.yaml"
        config = load_config(path)
        self.assertEqual(config["image_loss"], dict(mse_weight=4., lpips_weight=1., dists_weight=1.))
        self.assertEqual(config["objective"], dict(base_weight=.1, repaired_weight=0.))
        self.assertEqual(config["self_flow"]["start_step"], 20000)
        self.assertEqual(config["train"]["steps"], 200000)
        self.assertEqual(2*config["train"]["batch_size"]*config["train"]["accumulation_steps"], 8)
        self.assertFalse(config["model"]["codec"].get("precision_conditioning", False))


if __name__ == "__main__":
    unittest.main()
