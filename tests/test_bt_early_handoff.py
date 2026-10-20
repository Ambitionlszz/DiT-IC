from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic, sha256_file
from models.bt_self_flow.losses import ImageLoss
from scripts.launch_bt_lowrate_c import SingleCRun
from test_bt_backend import tiny_model
from test_bt_training import small_config


def early_source(root, step=3, phase="codec"):
    (root/"images").mkdir()
    Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
    config = small_config(root)
    config["train"]["steps"] = 5
    model = tiny_model("C")
    model.distortion_scale.calibrated.fill_(False)
    raw = clone_state({k: v for k, v in model.export_state().items() if k in ("codec", "distortion_scale")})
    path = root/"source.pt"
    save_atomic(path, dict(method=METHOD, schema=SCHEMA, phase=phase, backend="C", step=step,
                          raw=raw, ema=raw, calibration=None, config=config, assets={"test": "early"}))
    target = small_config(root, "joint")
    target["model"]["backend"] = "C"
    target["train"]["require_quality_span"] = True
    target["image_loss"] = dict(base_weight=0., latent_weight=0.)
    return path, target


def calibration(model, dataset, output, samples, seed):
    assert not model.distortion_scale.calibrated.item()
    model.distortion_scale.calibrate(torch.tensor(1.), [torch.tensor([.1, .5, 1.])])
    report = {"rate_span_pass": True, "quality_span_pass": True, "samples": samples, "seed": seed}
    output.write_text(json.dumps(report))
    return report


def make_c_model(config, device):
    model = tiny_model("C").to(device)
    model.vae.requires_grad_(False)
    return model


@pytest.mark.parametrize("step,phase,message", [(3, "codec", "has not completed"),
                                                (0, "codec", "outside"),
                                                (6, "codec", "outside"),
                                                (3, "joint", "requires a common codec")])
def test_early_source_requires_explicit_opt_in_and_valid_phase(tmp_path, step, phase, message):
    source, config = early_source(tmp_path, step, phase)
    with pytest.raises(ValueError, match=message), patch.object(trainer, "make_model") as factory:
        trainer.prepare_backend(config, source, tmp_path/"init.pt", torch.device("cpu"))
    factory.assert_not_called()


def test_early_handoff_calibrates_preserves_source_and_trains_c(tmp_path):
    source, config = early_source(tmp_path)
    original_hash = sha256_file(source)
    output = tmp_path/"backend_init.pt"
    with patch.object(trainer, "asset_identity", return_value={"test": "early"}), \
         patch.object(trainer, "make_model", side_effect=make_c_model), \
         patch.object(trainer, "calibrate_interface", side_effect=calibration) as calibrate, \
         redirect_stdout(io.StringIO()):
        trainer.prepare_backend(config, source, output, torch.device("cpu"), allow_incomplete_codec=True)
    calibrate.assert_called_once()
    initialized, original = read_checkpoint(output), read_checkpoint(source)
    assert sha256_file(source) == original_hash == initialized["source_sha256"]
    assert initialized["source_step"] == 3 and initialized["source_planned_steps"] == 5
    assert initialized["early_codec_handoff"] and initialized["source_weights"] == "raw"
    assert initialized["phase"] == "backend_init" and initialized["step"] == 0
    assert initialized["raw"]["distortion_scale"]["calibrated"].item()
    for key, value in original["raw"]["codec"].items():
        assert torch.equal(value, initialized["raw"]["codec"][key])
    args = SimpleNamespace(resume=None, init_checkpoint=str(output), max_updates=None)
    with patch.object(trainer, "asset_identity", return_value={"test": "early"}), \
         patch.object(trainer, "make_model", side_effect=make_c_model), \
         patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0, **kw)), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device("cpu"))
    completed = read_checkpoint(tmp_path/"joint/complete.pt")
    assert completed["step"] == config["train"]["steps"]
    assert completed["source_sha256"] == sha256_file(output)
    assert completed["config"]["image_loss"] == {"base_weight": 0., "latent_weight": 0.}


@pytest.mark.parametrize("failed_gate", ["rate_span_pass", "quality_span_pass"])
def test_early_handoff_rejects_failed_calibration(tmp_path, failed_gate):
    source, config = early_source(tmp_path)

    def failed(*args):
        result = calibration(*args)
        result[failed_gate] = False
        return result

    with patch.object(trainer, "asset_identity", return_value={"test": "early"}), \
         patch.object(trainer, "make_model", side_effect=make_c_model), \
         patch.object(trainer, "calibrate_interface", side_effect=failed), \
         pytest.raises(ValueError, match="common codec"):
        trainer.prepare_backend(config, source, tmp_path/"init.pt", torch.device("cpu"), allow_incomplete_codec=True)
    assert not (tmp_path/"init.pt").exists()


def test_completed_codec_keeps_existing_calibration_path(tmp_path):
    source, config = early_source(tmp_path, step=5)
    checkpoint = read_checkpoint(source)
    checkpoint["raw"]["distortion_scale"]["calibrated"].fill_(True)
    checkpoint["calibration"] = {"rate_span_pass": True, "quality_span_pass": True}
    save_atomic(source, checkpoint)
    with patch.object(trainer, "asset_identity", return_value={"test": "early"}), \
         patch.object(trainer, "make_model", side_effect=make_c_model), \
         patch.object(trainer, "calibrate_interface") as calibrate:
        trainer.prepare_backend(config, source, tmp_path/"init.pt", torch.device("cpu"))
    calibrate.assert_not_called()
    assert not read_checkpoint(tmp_path/"init.pt")["early_codec_handoff"]


def test_supervisor_skips_warmup_and_resumes_c_with_selected_source(tmp_path):
    source = tmp_path/"selected_30000.pt"
    source.touch()
    run = SingleCRun(tmp_path, ["GPU-a", "GPU-b"], "python", warmup_checkpoint=source)
    commands = []

    def start(name, command, gpus):
        commands.append((name, command, gpus))
        path = tmp_path/"backend_init.pt" if name == "prepare" else tmp_path/"C/joint/complete.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with patch.object(run, "start", side_effect=start), patch.object(run, "wait"):
        run.execute()
    assert [name for name, _, _ in commands] == ["prepare", "c_joint"]
    assert str(source) in commands[0][1] and "--allow-incomplete-codec" in commands[0][1]
    assert commands[1][2] == ["GPU-a", "GPU-b"]
    assert "--nproc_per_node=2" in commands[1][1]
    (tmp_path/"C/joint/complete.pt").unlink()
    (tmp_path/"C/joint/paused.pt").touch()
    commands.clear()
    with patch.object(run, "start", side_effect=start), patch.object(run, "wait"):
        run.execute()
    assert len(commands) == 1
    assert commands[0][1][-2:] == ["--resume", str(tmp_path/"C/joint/paused.pt")]


def test_missing_selected_source_never_starts_new_warmup(tmp_path):
    run = SingleCRun(tmp_path, ["GPU-a", "GPU-b"], "python", warmup_checkpoint=tmp_path/"missing.pt")
    with patch.object(run, "start") as start, pytest.raises(FileNotFoundError):
        run.execute()
    start.assert_not_called()
