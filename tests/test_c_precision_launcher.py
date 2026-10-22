"""Stage transitions and signal handling for the two-rank C deployment."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

from scripts.launch_c_precision import PrecisionCRun, check_handoff, resolved_configs
from test_bt_supervisor import WORKER, wait_file


def test_two_rank_recipe_preserves_update_budget_and_effective_batch(tmp_path):
    configs = resolved_configs(tmp_path)
    common, joint = configs["common"], configs["c_joint"]
    assert common["train"]["steps"] == 50000
    assert joint["train"]["steps"] == 200000
    assert (common["train"]["batch_size"], common["train"]["accumulation_steps"]) == (8, 1)
    assert (joint["train"]["batch_size"], joint["train"]["accumulation_steps"]) == (2, 2)
    assert joint["train"]["effective_batch"] == 8
    assert joint["objective"] == dict(base_weight=.1, repaired_weight=0.)
    assert common["quality"] == joint["quality"]
    assert common["model"]["codec"] == joint["model"]["codec"]
    assert joint["train"]["require_quality_span"]


def handoff_states(tmp_path):
    configs = resolved_configs(tmp_path)
    codec = {"weight": torch.tensor([1.])}
    source = dict(phase="codec", step=50000, config=configs["common"], assets={"test": "same"},
                  ema=dict(codec=codec), calibration=dict(rate_span_pass=True, quality_span_pass=True,
                                                         weights="deployment_ema"))
    shared = dict(phase="backend_init", backend="shared", codec_weights="ema", assets={"test": "same"},
                  raw=dict(codec=deepcopy(codec), distortion_scale=dict(calibrated=torch.tensor(True))))
    return configs, source, shared


def test_handoff_requires_exact_ema_and_successful_calibration(tmp_path):
    configs, source, shared = handoff_states(tmp_path)
    with patch("scripts.launch_c_precision.read_checkpoint", side_effect=[source, shared]):
        report = check_handoff("common", "shared", configs["c_joint"])
    assert report["quality_span_pass"] and report["codec_weights"] == "ema"
    shared["raw"]["codec"]["weight"].add_(1)
    with patch("scripts.launch_c_precision.read_checkpoint", side_effect=[source, shared]), \
         pytest.raises(ValueError, match="differs from warmup EMA"):
        check_handoff("common", "shared", configs["c_joint"])


def recording_run(tmp_path, failed_handoff=False):
    (tmp_path/"logs").mkdir()

    class RecordingRun(PrecisionCRun):
        def __init__(self):
            super().__init__(tmp_path, ["0", "6"], sys.executable, resolved_configs(tmp_path))
            self.events = []

        def start_reservations(self):
            self.events.append("reserve")

        def stop_reservations(self):
            self.events.append("release")

        def start(self, name, command, gpus):
            self.events.append(name)
            assert gpus == ["0"]
            assert "--prepare-backend" in command
            (tmp_path/"backend_init.pt").write_bytes(b"fixture")

        def wait(self, name):
            pass

        def train_stage(self, name, directory, initialization=None):
            self.events.append(name)
            command = self.command(name, directory, initialization, distributed=True)
            assert "--nproc_per_node=2" in command
            directory.mkdir(parents=True)
            (directory/"complete.pt").write_bytes(b"fixture")

    return RecordingRun()


def test_release_only_after_handoff_and_then_start_two_rank_c(tmp_path):
    run = recording_run(tmp_path)
    def checked(*args):
        run.events.append("checked")
        return dict(quality_span_pass=True)
    with patch("scripts.launch_c_precision.check_handoff", side_effect=checked):
        run.execute()
    assert run.events == ["reserve", "common", "prepare", "checked", "release", "c_joint"]
    assert run.state["status"] == "completed"


def test_failed_handoff_cannot_start_c(tmp_path):
    run = recording_run(tmp_path)
    with patch("scripts.launch_c_precision.check_handoff", side_effect=ValueError("quality span")), \
         pytest.raises(ValueError, match="quality span"):
        run.execute()
    assert "c_joint" not in run.events and "release" not in run.events


@pytest.mark.parametrize("name", ["common", "c_joint"])
def test_stop_signals_two_trainers_without_killing_data_workers(tmp_path, name):
    (tmp_path/"logs").mkdir()
    script = tmp_path/"train_bt_self_flow.py"
    script.write_text(WORKER)
    directories = [tmp_path/str(i) for i in range(2)]
    for directory in directories:
        directory.mkdir()
    launcher = tmp_path/"launcher.py"
    launcher.write_text("""
import subprocess, sys
children=[subprocess.Popen([sys.executable,sys.argv[1],p]) for p in sys.argv[2:]]
sys.exit(max(child.wait() for child in children))
""")
    run = PrecisionCRun(tmp_path, ["0", "6"], sys.executable, {})
    run.start(name, [sys.executable, str(launcher), str(script), *map(str, directories)], ["0", "6"])
    process = run.children[name]
    try:
        for directory in directories:
            wait_file(directory/"ready")
        run.stop(None, None)
        assert process.wait(timeout=10) == 0
        run.reap_stopped()
        for directory in directories:
            assert json.loads((directory/"paused.json").read_text())["data_worker_alive"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
