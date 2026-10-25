"""Bounded recovery requires a fresh, complete, paired FP32 replay checkpoint."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from models.bt_self_flow.checkpoint import METHOD, SCHEMA, save_atomic
from scripts.launch_quantization_residual import ResidualRun
from models.bt_self_flow.experiment import load_config, training_contract


def request_restart(root, step=12, retry=True):
    path = root/"joint/paused.pt"
    save_atomic(path, dict(method=METHOD, schema=SCHEMA, step=step, attempts=step+1,
                           full_precision_retry=retry, rng_by_rank=[{}, {}]))
    request = dict(reason="amp_nonfinite_forward", step=step, attempts=step+1,
                   world_size=2, checkpoint="paused.pt", checkpoint_size=path.stat().st_size,
                   checkpoint_mtime_ns=path.stat().st_mtime_ns)
    (root/"joint/fp32_restart.json").write_text(json.dumps(request))
    return path


def test_stable_recipe_preserves_optimization_and_saves_every_500_updates():
    base = load_config("configs/bt_self_flow/quantization_residual/c_sf_b4a2.json")
    stable = load_config("configs/bt_self_flow/quantization_residual/c_sf_b4a2_stable.json")
    dataset = SimpleNamespace(fingerprint="same-training-images")
    assert training_contract(base, dataset, 2) == training_contract(stable, dataset, 2)
    assert stable["train"]["save_every"] == 500

def test_restart_requires_fresh_complete_fp32_checkpoint(tmp_path):
    (tmp_path/"joint").mkdir()
    run = ResidualRun(tmp_path, ["0", "1"], sys.executable)
    assert run.fp32_restart_checkpoint() is None
    checkpoint = request_restart(tmp_path)
    assert run.fp32_restart_checkpoint()[0] == checkpoint
    request_restart(tmp_path, retry=False)
    assert run.fp32_restart_checkpoint() is None
    request_restart(tmp_path)
    checkpoint.write_bytes(b"incomplete")
    assert run.fp32_restart_checkpoint() is None


def test_joint_restarts_once_and_then_completes(tmp_path):
    (tmp_path/"joint").mkdir()
    run = ResidualRun(tmp_path, ["0", "1"], sys.executable)
    commands = []
    def start(name, command, gpus):
        commands.append(command)
    def wait(name):
        if len(commands) == 1:
            request_restart(tmp_path)
            raise RuntimeError("worker exited after durable recovery save")
    with patch.object(run, "start", side_effect=start), patch.object(run, "wait", side_effect=wait):
        run.train_joint(tmp_path/"config.json", "init.pt")
    assert len(commands) == 2 and "--resume" in commands[1]
    assert commands[1][-1] == str(tmp_path/"joint/paused.pt")
    assert len(json.loads((tmp_path/"numerical_restarts.json").read_text())) == 1
    assert not (tmp_path/"joint/fp32_restart.json").exists()


def test_restart_budget_and_unrelated_failure_do_not_loop(tmp_path):
    (tmp_path/"joint").mkdir()
    run = ResidualRun(tmp_path, ["0", "1"], sys.executable)
    calls = []
    def wait(name):
        calls.append(name)
        request_restart(tmp_path, step=len(calls))
        raise RuntimeError("worker failed")
    with patch.object(run, "start"), patch.object(run, "wait", side_effect=wait):
        with pytest.raises(RuntimeError, match="worker failed"):
            run.train_joint(tmp_path/"config.json", "init.pt", max_numerical_restarts=1)
    assert len(calls) == 2
    # A leftover marker from the previous child cannot restart an unrelated
    # failure (e.g. CUDA OOM, missing data or a failed model initialization).
    with patch.object(run, "start"), patch.object(run, "wait", side_effect=RuntimeError("CUDA OOM")):
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            run.train_joint(tmp_path/"config.json", "init.pt")
