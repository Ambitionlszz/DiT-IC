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


@pytest.mark.parametrize("distributed", [False, True])
def test_stop_preserves_workers_until_checkpoint(tmp_path, distributed):
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
    name = "common" if distributed else "c_joint"
    run.start(name, command, ["0"])
    process = run.children[name]
    try:
        for directory in directories:
            wait_file(directory/"ready")
        run.stop(None, None)
        assert process.wait(timeout=10) == 0
        for directory in directories:
            assert json.loads((directory/"paused.json").read_text())["data_worker_alive"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
