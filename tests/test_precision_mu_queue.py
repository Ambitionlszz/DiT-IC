"""The queued run must be the audited precision B and must pass its gates."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.queue_precision_mu_pit import Supervisor, METHOD, GPU_UUID, write_json


def plan(tmp_path):
    return dict(source=str(tmp_path), python="python", config="b.yaml", initialization="shared.pt",
                method=METHOD, gpu_index=2, gpu_uuid=GPU_UUID, poll_seconds=0, max_idle_memory_mib=256,
                required_idle_polls=2, commit="pinned", contract={"decoder": METHOD}, assets={"weights": "same"},
                initialization_sha256="same_init")


def test_waits_for_two_idle_polls_without_torch_or_cuda(tmp_path):
    watcher = Supervisor(tmp_path, plan(tmp_path))
    observations = [dict(used_mib=20948, utilization=0), dict(used_mib=16, utilization=0),
                    dict(used_mib=300, utilization=0), dict(used_mib=16, utilization=0),
                    dict(used_mib=16, utilization=0)]
    with patch.object(watcher, "gpu", side_effect=observations) as gpu:
        assert watcher.wait_idle()
    assert gpu.call_count == 5
    assert json.loads((tmp_path / "status.json").read_text())["consecutive_idle_polls"] == 2
    assert watcher.environment()["CUDA_VISIBLE_DEVICES"] == GPU_UUID


@pytest.mark.parametrize("failure", ["preflight", "initialization", "contract", "method", "source", "readiness"])
def test_never_releases_training_gate_after_failed_checks(tmp_path, failure):
    values = plan(tmp_path)
    watcher = Supervisor(tmp_path, values)
    launches = []
    class Child:
        pid = 123
        def poll(self): return None
        def terminate(self): pass
    def launch(command, log, stage):
        launches.append(stage)
        watcher.child = Child()
        if stage == "preflight":
            write_json(tmp_path / "preflight/full_batch8.json",
                       dict(status="failed" if failure == "preflight" else "passed", decoder_method=METHOD, batch_size=8,
                            initialization_sha256="wrong_init" if failure == "initialization" else "same_init"))
        else:
            (tmp_path / "joint").mkdir()
            resolved = dict(contract=deepcopy(values["contract"]), assets=values["assets"],
                            git_commit=values["commit"], config=dict(decoder=dict(method=METHOD)))
            if failure == "contract": resolved["contract"] = {"wrong_batch": 1}
            if failure == "method": resolved["config"]["decoder"]["method"] = "old_B"
            if failure == "source": resolved["git_commit"] = "wrong_commit"
            write_json(tmp_path / "joint/resolved.json", resolved)
            write_json(tmp_path / "startup_ready.json", dict(pid=999 if failure == "readiness" else 123,
                                                            batch_size=8, accumulation_steps=1, ready_for_updates=True))
    with patch.object(watcher, "verify_plan"), patch.object(watcher, "wait_idle", return_value=True), \
         patch.object(watcher, "launch", side_effect=launch), patch.object(watcher, "await_child"):
        with pytest.raises(ValueError): watcher.run()
    assert not (tmp_path / "startup_gate.json").exists()
    assert launches == (["preflight"] if failure in ("preflight", "initialization") else ["preflight", "training_initializing"])


def test_success_releases_only_precision_b_then_tracks_training(tmp_path):
    values = plan(tmp_path)
    watcher = Supervisor(tmp_path, values)
    commands = []
    class Child:
        pid = 123
        def poll(self): return None
    def launch(command, log, stage):
        commands.append(command)
        watcher.child = Child()
        if stage == "preflight":
            write_json(tmp_path / "preflight/full_batch8.json", dict(status="passed", decoder_method=METHOD, batch_size=8,
                                                                    initialization_sha256="same_init"))
        else:
            (tmp_path / "joint").mkdir()
            write_json(tmp_path / "joint/resolved.json", dict(contract=values["contract"], assets=values["assets"],
                       git_commit=values["commit"], config=dict(decoder=dict(method=METHOD))))
            write_json(tmp_path / "startup_ready.json", dict(pid=123, batch_size=8, accumulation_steps=1, ready_for_updates=True))
    with patch.object(watcher, "verify_plan"), patch.object(watcher, "wait_idle", return_value=True), \
         patch.object(watcher, "launch", side_effect=launch), patch.object(watcher, "await_child"):
        watcher.run()
    assert watcher.state["stage"] == "completed"
    assert json.loads((tmp_path / "startup_gate.json").read_text())["decoder_method"] == METHOD
    assert commands[0][commands[0].index("--batch-size") + 1] == "8"
    assert commands[0][commands[0].index("--updates") + 1] == "3"
    assert "--resume" not in commands[1]


def test_stop_file_cancels_waiting_without_launch(tmp_path):
    watcher = Supervisor(tmp_path, plan(tmp_path))
    (tmp_path / "STOP").touch()
    with patch.object(watcher, "gpu") as gpu:
        assert not watcher.wait_idle()
    gpu.assert_not_called()
