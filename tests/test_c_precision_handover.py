"""Reservation ownership must change before the old trainers release CUDA."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.bt_self_flow.losses import rate_beta
from models.bt_self_flow.packet import RateControl
from scripts import c_precision_handover as handover
from scripts.launch_c_precision import resolved_configs
from test_bt_supervisor import wait_file


def test_selected_rate_weights_reach_loss_and_both_stages(tmp_path):
    configs = resolved_configs(tmp_path)
    expected = [16., 8., 4., 2., 1., .5]
    for config in configs.values():
        quality = config["quality"]
        assert quality["rate_coefficients"] == expected
        assert [rate_beta(RateControl(q), quality) for q in quality["values"]] == pytest.approx(expected)
        assert config["train"]["require_quality_span"]


def replacement_fixture(tmp_path, fail_standby=False):
    old = dict(supervisor=dict(pid=1, token="parent"), launcher=dict(pid=2, token="launcher"),
               ranks=[dict(pid=3, token="rank0"), dict(pid=4, token="rank1")],
               holders={"gpu0": dict(pid=5, token="holder0"), "gpu1": dict(pid=6, token="holder1")})
    living, events = set(range(1, 7)), []
    new_ready = set()

    class Run:
        root = tmp_path
        gpus = ["gpu0", "gpu1"]
        reserve_gib = 18.
        reservations = {}

        def spawn_reservations(self, gated=False):
            assert gated
            events.append("new_contexts_ready")
            for index, gpu in enumerate(self.gpus):
                (self.root/f"reserve_gpu{index}.standby.json").write_text("{}")
                self.reservations[gpu] = (SimpleNamespace(pid=20+index, poll=lambda: None),
                                          self.root/f"reserve_gpu{index}.json")

        def wait_reservations(self, predicate, timeout=60):
            if fail_standby:
                raise TimeoutError("standby failed")
            for index, gpu in enumerate(self.gpus):
                if (self.root/f"reserve_gpu{index}.allocate.json").exists() and gpu not in new_ready:
                    assert {3, 4} <= living
                    assert old["holders"][gpu]["pid"] not in living
                    self.reservations[gpu][1].write_text("{}")
                    events.append(f"new_{gpu}_ready")
                    new_ready.add(gpu)
            assert predicate()

        def update(self, status):
            events.append(status)

    def send(identity, signum):
        pid = identity["pid"]
        if pid in (5, 6):
            assert signum == signal.SIGTERM and "new_contexts_ready" in events
            assert {3, 4} <= living
            living.remove(pid)
            events.append(f"old_holder_{pid}_released")
        elif signum == signal.SIGTERM:
            assert new_ready == {"gpu0", "gpu1"}
            events.append("old_trainers_stop_requested")
        elif signum == signal.SIGCONT:
            events.append("old_supervisor_resumed")
            if not fail_standby:
                living.difference_update({1, 2, 3, 4})
        else:
            assert signum == signal.SIGSTOP
            events.append("old_supervisor_parked")

    return Run(), old, living, events, send


def test_handover_allocates_both_replacements_before_stopping_trainers(tmp_path):
    run, old, living, events, send = replacement_fixture(tmp_path)
    with patch.object(handover, "inspect_previous", return_value=old), \
         patch.object(handover, "send", side_effect=send), \
         patch.object(handover, "alive", side_effect=lambda identity: identity["pid"] in living):
        handover.take_over(run, tmp_path/"previous")
    assert events == ["old_supervisor_parked", "new_contexts_ready", "old_holder_5_released",
                      "new_gpu0_ready", "old_holder_6_released", "new_gpu1_ready",
                      "waiting_previous_checkpoint", "old_trainers_stop_requested", "old_supervisor_resumed"]
    assert json.loads((tmp_path/"handover.json").read_text())["status"] == "completed"


def test_failed_standby_resumes_original_without_releasing_its_memory(tmp_path):
    run, old, living, events, send = replacement_fixture(tmp_path, fail_standby=True)
    with patch.object(handover, "inspect_previous", return_value=old), \
         patch.object(handover, "send", side_effect=send), \
         pytest.raises(TimeoutError, match="standby failed"):
        handover.take_over(run, tmp_path/"previous")
    assert living == set(range(1, 7))
    assert events == ["old_supervisor_parked", "new_contexts_ready", "old_supervisor_resumed"]


def test_reused_pid_is_not_signalled():
    with patch.object(handover, "process_state", return_value=dict(pid=123, token="new", state="S")), \
         patch.object(handover.os, "kill") as kill:
        handover.send(dict(pid=123, token="old"), signal.SIGTERM)
    kill.assert_not_called()


def test_foreign_supervisor_is_rejected_before_signalling(tmp_path):
    (tmp_path/"pipeline.json").write_text(json.dumps(dict(status="latent_warmup_ddp", supervisor_pid=123)))
    with patch.object(handover, "process_state", return_value=dict(pid=123, token="new", state="S",
                                                                  command=["unrelated.py"])), \
         patch.object(handover.os, "kill") as kill, \
         pytest.raises(ValueError, match="supervisor identity"):
        handover.inspect_previous(tmp_path, ["gpu0", "gpu1"], 18.)
    kill.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is inaccessible")
def test_cuda_holder_preinitializes_then_allocates_only_after_gate(tmp_path):
    gate, standby, status = [tmp_path/name for name in ("gate.json", "standby.json", "status.json")]
    script = Path(__file__).resolve().parents[1]/"scripts/reserve_gpu_memory.py"
    process = subprocess.Popen([sys.executable, str(script), "--gib", ".01",
                                "--minimum-free-gib", ".5", "--status", str(status),
                                "--parent-pid", str(os.getpid()), "--allocation-gate", str(gate),
                                "--standby-status", str(standby)])
    try:
        wait_file(standby)
        assert process.poll() is None and not status.exists()
        gate.write_text("{}")
        wait_file(status)
        record = json.loads(status.read_text())
        assert record["ready"] and record["allocated_bytes"] >= int(.01*1024**3)
        assert not standby.exists()
    finally:
        if process.poll() is None:
            process.terminate()
        assert process.wait(timeout=30) == 0
    assert not status.exists()

