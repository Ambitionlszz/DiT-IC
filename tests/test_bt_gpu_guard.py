import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from scripts.guard_bt_training import TrainingGuard, signal_group


def until(guard, predicate, timeout=15):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        guard.tick()
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("guardian condition was not reached")


def make_guard(root, command, defer=False):
    rows = {3: dict(uuid="GPU-three", used_mib=0, free_mib=32000),
            5: dict(uuid="GPU-five", used_mib=0, free_mib=32000)}
    holder = lambda gpu: [sys.executable, "-u", "-c",
                          'import os,time; print("[Stress] running", os.environ["CUDA_VISIBLE_DEVICES"], flush=True); time.sleep(300)']
    guard = TrainingGuard(root, {gpu: row["uuid"] for gpu, row in rows.items()}, command, holder,
                          snapshot=lambda: rows, defer=defer)
    return guard, rows


def cleanup(guard):
    if guard.child:
        signal_group(guard.child.pid, signal.SIGKILL)
        guard.child.wait(timeout=5)
    if guard.pending_cleanup:
        signal_group(guard.pending_cleanup[0], signal.SIGKILL)
    guard.stop_holders()


def test_child_failure_reclaims_exact_pair_and_restarts_dead_holder(tmp_path):
    guard, rows = make_guard(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    try:
        rows[3]["used_mib"] = 2000  # An unrelated allocation must never be evicted.
        until(guard, lambda: 5 in guard.holders)
        assert 3 not in guard.holders
        assert guard.status == "holding_after_failure" and guard.last_exit == 7
        rows[3]["used_mib"] = 0
        until(guard, lambda: len(guard.holders) == 2 and all(
            "[Stress] running" in path.read_text() for _, path in guard.holders.values()))
        for gpu, (_, path) in guard.holders.items():
            assert guard.devices[gpu] in path.read_text()
        old_pid = guard.holders[3][0].pid
        peer_pid = guard.holders[5][0].pid
        guard.holders[3][0].kill()
        guard.holders[3][0].wait(timeout=5)
        until(guard, lambda: guard.holders[3][0].pid != old_pid)
        assert guard.holders[5][0].pid == peer_pid
        guard.release_requested = True
        assert not guard.tick()
        assert json.loads((tmp_path/"guard.json").read_text())["status"] == "released"
    finally:
        cleanup(guard)


def test_killed_supervisor_cleans_orphan_and_reclaims_both_gpus(tmp_path):
    orphan_pid = tmp_path/"orphan.pid"
    command = [sys.executable, "-u", "-c", "\n".join([
        "import subprocess,sys,time",
        "from pathlib import Path",
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])",
        f"Path({str(orphan_pid)!r}).write_text(str(p.pid))",
        "time.sleep(300)",
    ])]
    guard, _ = make_guard(tmp_path, command, defer=True)
    try:
        guard.tick()
        assert not guard.child and not guard.holders
        guard.start_requested = True
        until(guard, orphan_pid.exists)
        worker = int(orphan_pid.read_text())
        guard.child.kill()
        until(guard, lambda: len(guard.holders) == 2)
        status = f"/proc/{worker}/stat"
        if os.path.exists(status):
            assert open(status).read().split()[2] == "Z", "orphan survived pipeline cleanup"
        assert guard.last_exit == -signal.SIGKILL
    finally:
        cleanup(guard)


def test_resume_releases_holders_and_launches_pipeline_again(tmp_path):
    guard, _ = make_guard(tmp_path, [sys.executable, "-c", "raise SystemExit(0)"])
    try:
        until(guard, lambda: len(guard.holders) == 2)
        assert guard.status == "holding_after_completion"
        holders = [process for process, _ in guard.holders.values()]
        guard.start_requested = True
        guard.tick()
        assert guard.child is not None and not guard.holders
        assert all(process.poll() is not None for process in holders)
        until(guard, lambda: len(guard.holders) == 2)
    finally:
        cleanup(guard)


def test_full_log_disk_does_not_prevent_cuda_holder_launch(tmp_path):
    guard, _ = make_guard(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    original_open = Path.open

    def full_holder_logs(path, *args, **kwargs):
        if path.name.startswith("holder_gpu"):
            raise OSError(28, "No space left on device")
        return original_open(path, *args, **kwargs)

    try:
        with patch.object(Path, "open", full_holder_logs):
            until(guard, lambda: len(guard.holders) == 2)
            time.sleep(.1)
            assert all(process.poll() is None and path is None for process, path in guard.holders.values())
    finally:
        cleanup(guard)


def test_pipeline_start_failure_also_reclaims_both_gpus(tmp_path):
    guard, _ = make_guard(tmp_path, [str(tmp_path/"missing-python")])
    try:
        with pytest.raises(FileNotFoundError):
            guard.tick()
        until(guard, lambda: len(guard.holders) == 2)
        assert guard.status == "holding_after_start_failure"
        assert guard.last_exit == -1
    finally:
        cleanup(guard)


def test_subreaper_cleans_rank_in_separate_session_after_supervisor_is_killed(tmp_path):
    code = '''
import os,sys,time
from pathlib import Path
from scripts.guard_bt_training import TrainingGuard, enable_subreaper
root=Path(sys.argv[1])
enable_subreaper()
worker="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],start_new_session=True); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(300)"
holder=lambda gpu: [sys.executable,'-u','-c','import time; print("[Stress] running",flush=True); time.sleep(300)']
guard=TrainingGuard(root,{3:'GPU-three',5:'GPU-five'},[sys.executable,'-c',worker,str(root/'rank.pid')],holder,
                    snapshot=lambda:{3:dict(uuid='GPU-three',used_mib=0),5:dict(uuid='GPU-five',used_mib=0)},subreaper=True)
deadline=time.monotonic()+15
try:
    while not (root/'rank.pid').exists():
        guard.tick(); time.sleep(.05)
        assert time.monotonic()<deadline
    rank=int((root/'rank.pid').read_text())
    guard.child.kill()
    while len(guard.holders)!=2:
        guard.tick(); time.sleep(.05)
        assert time.monotonic()<deadline
    guard.tick()
    stat=Path(f'/proc/{rank}/stat')
    assert not stat.exists() or stat.read_text().split()[2]=='Z'
finally:
    guard.stop_holders()
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=20)
