"""Safety checks for the GPU5 holder-to-training handoff; no GPU required."""

import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from scripts import launch_c_v3_single as launch


def identities():
    return ({"pid": 111, "state": "S", "token": "old-controller", "command":
             ["python", "watch_two_idle_gpus.py", "--slots", "1"]},
            {"pid": 222, "state": "S", "token": "old-holder", "command":
             ["python", "train_depart.py", "--gb", "27"]})


def test_refuses_other_gpu_process_and_multi_holder_controller(tmp_path, monkeypatch):
    watcher, holder = identities()
    monkeypatch.setattr(launch, "process_state", lambda pid: {111: watcher, 222: holder}.get(pid))
    process = tmp_path/"222"
    process.mkdir()
    (process/"stat").write_text("222 (python) S 111 0 0\n")
    (process/"environ").write_bytes(b"CUDA_VISIBLE_DEVICES=GPU-five\0")
    assert launch.inspect_holder(111, 222, "GPU-five", tmp_path) == (watcher, holder)
    with pytest.raises(ValueError, match="exact physical GPU"):
        launch.inspect_holder(111, 222, "GPU-other", tmp_path)
    holder["command"][1] = "train_bt_self_flow.py"
    with pytest.raises(ValueError, match="training process"):
        launch.inspect_holder(111, 222, "GPU-five", tmp_path)
    holder["command"][1] = "train_depart.py"
    watcher["command"][-1] = "2"
    with pytest.raises(ValueError, match="single holder"):
        launch.inspect_holder(111, 222, "GPU-five", tmp_path)


def test_holder_release_requires_preloaded_cuda_replacements(tmp_path):
    run = launch.Run.__new__(launch.Run)
    run.root = tmp_path
    with pytest.raises(ValueError, match="replacement CUDA processes must be ready"):
        run.take_over()


def test_guard_is_allocated_before_controller_restart(tmp_path, monkeypatch):
    watcher, holder = identities()
    events, live = [], {222: True}
    run = launch.Run.__new__(launch.Run)
    run.root = tmp_path
    run.args = SimpleNamespace(replace_watcher_pid=111, replace_holder_pid=222)
    run.manifest = {"gpu_uuid": "GPU-five"}
    run.worker = SimpleNamespace(pid=444)
    run.guard = SimpleNamespace(pid=333, poll=lambda: None)
    run.stopping = False
    for name in ("guard.standby.json", "warmup.allocation.ready.json", "preflight.allocation.ready.json"):
        (tmp_path/name).write_text("{}")
    monkeypatch.setattr(launch, "inspect_holder", lambda *_: (watcher, holder))
    monkeypatch.setattr(launch, "alive", lambda value: live.get(value["pid"], True))
    def send(value, signum):
        events.append((value["pid"], signum))
        if value["pid"] == 222 and signum == signal.SIGTERM:
            live[222] = False
    monkeypatch.setattr(launch, "send", send)
    original = launch.write_json
    def write(path, value):
        events.append(path.name)
        original(path, value)
        if path.name == "guard.allocate.json":
            (tmp_path/"guard.ready.json").write_text("{}")
    monkeypatch.setattr(launch, "write_json", write)
    run.update = lambda *_, **__: None
    run.take_over()
    assert events.index((111, signal.SIGSTOP)) < events.index((222, signal.SIGTERM))
    assert events.index((222, signal.SIGTERM)) < events.index("guard.allocate.json")
    assert events.index("guard.allocate.json") < events.index((111, signal.SIGCONT))
    report = json.loads((tmp_path/"handover.json").read_text())
    assert report["worker_cuda_allocation_retained"]
    assert report["replacement_guard_pid"] == 333


def test_single_gpu_configs_preserve_eight_by_one_and_stage_identity(tmp_path):
    values = launch.configs(tmp_path)
    assert values["warmup"]["execution"]["frozen_encoder_batch_size"] == 2
    assert values["joint"]["execution"]["frozen_encoder_batch_size"] == 0
    assert values["warmup"]["train"]["steps"] == 50000
    assert values["joint"]["train"]["steps"] == 200000
    assert values["warmup"]["model"] == values["joint"]["model"]


def test_failed_fullcheck_restores_protection_before_releasing_warmup():
    events = []
    run = launch.Run.__new__(launch.Run)
    run.stopping, run.holder_release_requested = False, True
    run.preflight = SimpleNamespace(poll=lambda: 1)
    run.worker = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append("release-warmup"))
    run.restore_protection = lambda: events.append("20gib-ready")
    assert run.recover_failure() is None
    assert events == ["20gib-ready", "release-warmup"]


def test_failed_protection_recovery_retains_the_live_cuda_worker():
    events = []
    run = launch.Run.__new__(launch.Run)
    run.stopping, run.holder_release_requested = False, True
    run.preflight = SimpleNamespace(poll=lambda: 1)
    run.worker = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append("release-warmup"))
    def failed():
        raise RuntimeError("reservation unavailable")
    run.restore_protection = failed
    assert "reservation unavailable" in run.recover_failure()
    assert events == []


def test_failed_guard_keeps_supervisor_alive_for_the_retained_cuda_worker():
    run = launch.Run.__new__(launch.Run)
    run.guard = SimpleNamespace(poll=lambda: 1)
    run.preflight = SimpleNamespace(poll=lambda: 1)
    run.worker = SimpleNamespace(poll=lambda: None)
    assert run.protection_alive("guard recovery failed")
    run.worker = SimpleNamespace(poll=lambda: 1)
    assert not run.protection_alive("guard recovery failed")
