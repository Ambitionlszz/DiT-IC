"""Freeze and supervise the reviewed two-GPU C residual-SF experiment."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.launch_bt_pair import PairedRun, write_json
from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.experiment import load_config


def resume_checkpoint(directory):
    candidates = [p for p in directory.glob("*.pt") if p.stem.isdecimal()
                  or p.name in ("paused.pt", "pre_sf.pt", "sf_start.pt")]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None


class ResidualRun(PairedRun):
    def exit_status(self, name, code):
        return "failed" if code else "completed" if (self.root/"joint/complete.pt").exists() else "paused"

    def stop(self, signum, frame):
        self.stop_requested = True
        for process in list(self.children.values()):
            if process.poll() is not None:
                continue
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            targets = []
            for child in children.read_text().split() if children.exists() else []:
                try:
                    command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    continue
                if any(part.rsplit(b"/", 1)[-1] == b"train_bt_self_flow.py" for part in command):
                    targets.append(int(child))
            for pid in targets or [process.pid]:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(x.isdecimal() for x in gpus):
        raise ValueError("two distinct GPU indices are required")
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        git = ["git", "-c", f"safe.directory={ROOT}"]
        if subprocess.check_output(git+["status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise RuntimeError("commit reviewed code before launching a research run")
        commit = subprocess.check_output(git+["rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        config = load_config(args.config)
        config["train"]["output"] = str(root/"joint")
        if 2*config["train"]["batch_size"]*config["train"]["accumulation_steps"] != 16:
            raise ValueError("research run must preserve effective batch 16")
        manifest = dict(repository=str(ROOT), commit=commit, gpus=gpus, python=sys.executable,
                        initialization=str(Path(args.init_checkpoint).resolve()),
                        initialization_sha256=sha256_file(args.init_checkpoint), config=config)
        if (root/"manifest.json").exists() and json.loads((root/"manifest.json").read_text()) != manifest:
            raise ValueError("frozen run code/config/init differs")
        (root/"configs").mkdir(exist_ok=True)
        (root/"logs").mkdir(exist_ok=True)
        write_json(root/"manifest.json", manifest)
        write_json(root/"configs/joint.json", config)
        run = ResidualRun(root, gpus, sys.executable)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, run.stop)
        try:
            if (root/"joint/complete.pt").exists():
                run.update("completed")
                return
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "train_bt_self_flow.py", "--config", str(root/"configs/joint.json")]
            previous = resume_checkpoint(root/"joint")
            command += ["--resume", str(previous)] if previous else ["--init-checkpoint", args.init_checkpoint]
            run.update("training")
            run.start("joint", command, gpus)
            run.wait("joint")
            run.update("completed" if (root/"joint/complete.pt").exists() else "paused")
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested else "failed")
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
