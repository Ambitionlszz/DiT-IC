"""Stop must let trainers save while their data workers remain alive."""

import json
from pathlib import Path
import sys
import time

import pytest

from scripts.launch_bt_pair import PairedRun


WORKER = '''
import json, multiprocessing as mp, os, signal, sys, time
from pathlib import Path
def data_worker():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    time.sleep(30)
if __name__ == "__main__":
    root = Path(sys.argv[1])
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *a: stopping.__setitem__(0, True))
    child = mp.Process(target=data_worker)
    child.start()
    (root/"ready").write_text(str(os.getpid()))
    while not stopping[0]:
        time.sleep(.01)
    # Represents the checkpoint written at a completed update boundary.
    (root/"paused.json").write_text(json.dumps({"data_worker_alive": child.is_alive()}))
    child.terminate()
    child.join()
'''


def wait_file(path):
    deadline = time.monotonic()+10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"worker did not become ready: {path}")
        time.sleep(.02)


@pytest.mark.parametrize("distributed,name", [(False, "c_joint"), (True, "common"), (True, "d_joint")])
def test_stop_preserves_workers_until_checkpoint(tmp_path, distributed, name):
    (tmp_path/"logs").mkdir()
    script = tmp_path/"train_bt_self_flow.py"
    script.write_text(WORKER)
    directories = [tmp_path/str(i) for i in range(2 if distributed else 1)]
    for directory in directories:
        directory.mkdir()
    if distributed:
        launcher = tmp_path/"launcher.py"
        launcher.write_text('''
import subprocess, sys
children = [subprocess.Popen([sys.executable, sys.argv[1], path]) for path in sys.argv[2:]]
codes = [child.wait() for child in children]
sys.exit(max(codes))
''')
        command = [sys.executable, str(launcher), str(script), *map(str, directories)]
    else:
        command = [sys.executable, str(script), str(directories[0])]
    run = PairedRun(tmp_path, ["0", "1"], sys.executable)
    run.start(name, command, ["0"])
    if distributed and name != "common":
        # Emulate the production torchrun command identity for this job name.
        run.state["jobs"][name]["command"] = [*command, "torch.distributed.run"]
    process = run.children[name]
    try:
        for directory in directories:
            wait_file(directory/"ready")
        run.stop(None, None)
        assert process.wait(timeout=10) == 0
        run.reap_stopped()
        assert run.state["jobs"][name]["status"] == "paused"
        assert run.state["jobs"][name]["returncode"] == 0
        assert not run.children
        for directory in directories:
            assert json.loads((directory/"paused.json").read_text())["data_worker_alive"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_supervisor_runs_both_joint_arms_directly_from_shared_init(tmp_path):
    (tmp_path/"common").mkdir()
    (tmp_path/"common/complete.pt").write_bytes(b"completed common fixture")
    (tmp_path/"logs").mkdir()

    class RecordingRun(PairedRun):
        def __init__(self):
            super().__init__(tmp_path, ["0", "1"], sys.executable)
            self.started = []

        def start(self, name, command, gpus):
            self.started.append((name, command, gpus))
            if name == "prepare":
                (tmp_path/"backend_init.pt").write_bytes(b"shared init fixture")
            else:
                directory = tmp_path/name[0].upper()/"joint"
                directory.mkdir(parents=True)
                (directory/"complete.pt").write_bytes(b"joint complete fixture")

        def wait(self, name):
            pass

        def finish(self, name):
            return True

    run = RecordingRun()
    run.execute()
    assert [name for name, _, _ in run.started] == ["prepare", "c_joint", "d_joint"]
    for name, command, gpus in run.started[1:]:
        assert command[command.index("--init-checkpoint")+1] == str(tmp_path/"backend_init.pt")
        assert gpus == (["0"] if name == "c_joint" else ["1"])
    assert run.state["status"] == "completed"


@pytest.mark.parametrize("backend", ["C", "D"])
def test_single_backend_uses_both_cards_and_shared_init(tmp_path, backend):
    (tmp_path/"common").mkdir()
    (tmp_path/"common/complete.pt").write_bytes(b"completed common fixture")
    (tmp_path/"backend_init.pt").write_bytes(b"shared init fixture")
    (tmp_path/"logs").mkdir()
    class RecordingRun(PairedRun):
        def start(self, name, command, gpus):
            self.started = (name, command, gpus)
            directory = tmp_path/backend/"joint"
            directory.mkdir(parents=True)
            (directory/"complete.pt").write_bytes(b"complete fixture")
        def finish(self, name):
            return True
    run = RecordingRun(tmp_path, ["0", "1"], sys.executable, backend=backend)
    run.execute()
    name, command, gpus = run.started
    assert name == f"{backend.lower()}_joint" and gpus == ["0", "1"]
    assert "torch.distributed.run" in command and "--nproc_per_node=2" in command
    assert command[command.index("--init-checkpoint")+1] == str(tmp_path/"backend_init.pt")
    assert run.state["status"] == "completed"


@pytest.mark.parametrize("completed", [False, True])
def test_finish_distinguishes_paused_from_completed_training(tmp_path, completed):
    (tmp_path/"logs").mkdir()
    directory = tmp_path/"D/joint"
    directory.mkdir(parents=True)
    (directory/("complete.pt" if completed else "paused.pt")).write_bytes(b"checkpoint fixture")
    run = PairedRun(tmp_path, ["0", "1"], sys.executable)
    run.start("d_joint", [sys.executable, "-c", "pass"], ["1"])
    assert run.children["d_joint"].wait(timeout=10) == 0
    assert run.finish("d_joint")
    assert run.state["jobs"]["d_joint"]["status"] == ("completed" if completed else "paused")
    assert not run.children
