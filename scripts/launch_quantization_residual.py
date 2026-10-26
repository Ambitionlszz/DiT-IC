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


def check_effective_batch(config, world_size):
    settings = config["train"]
    actual = world_size*settings["batch_size"]*settings["accumulation_steps"]
    if actual != settings["effective_batch"]:
        raise ValueError("GPU count/batch/accumulation differs from the declared effective batch")
    return actual


class ResidualRun(PairedRun):
    def exit_status(self, name, code):
        return "failed" if code else "completed" if (self.root/"joint/complete.pt").exists() else "paused"

    def fp32_restart_checkpoint(self):
        request_path = self.root/"joint/fp32_restart.json"
        if not request_path.exists():
            return None
        request = json.loads(request_path.read_text())
        path = self.root/"joint/paused.pt"
        if (request.get("reason") != "amp_nonfinite_forward" or request.get("checkpoint") != path.name
                or request.get("world_size") != len(self.gpus) or not path.exists()
                or path.stat().st_size != request.get("checkpoint_size")
                or path.stat().st_mtime_ns != request.get("checkpoint_mtime_ns")):
            return None
        checkpoint = read_checkpoint(path)
        if (not checkpoint.get("full_precision_retry") or checkpoint["step"] != request.get("step")
                or checkpoint["attempts"] != request.get("attempts")
                or len(checkpoint.get("rng_by_rank", [])) != len(self.gpus)):
            return None
        return path, request

    def joint_command(self, config_path, initialization, previous):
        command = [self.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "train_bt_self_flow.py", "--config", str(config_path)]
        return command+(["--resume", str(previous)] if previous else ["--init-checkpoint", initialization])

    def train_joint(self, config_path, initialization, max_numerical_restarts=3):
        history_path = self.root/"numerical_restarts.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        self.state["numerical_restarts"] = history
        while True:
            previous = resume_checkpoint(self.root/"joint")
            # A failed launch/OOM must never reuse an earlier restart request.
            (self.root/"joint/fp32_restart.json").unlink(missing_ok=True)
            self.update("training")
            self.start("joint", self.joint_command(config_path, initialization, previous), self.gpus)
            try:
                self.wait("joint")
                return
            except RuntimeError:
                retry = None if self.stop_requested else self.fp32_restart_checkpoint()
                if retry is None or len(history) >= max_numerical_restarts:
                    raise
                checkpoint, request = retry
                history.append(dict(**request, checkpoint_path=str(checkpoint), restarted_at=self.now()))
                write_json(history_path, history)
                self.update("recovering_fp32")

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
        check_effective_batch(config, len(gpus))
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
            run.train_joint(root/"configs/joint.json", args.init_checkpoint)
            run.update("completed" if (root/"joint/complete.pt").exists() else "paused")
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested else "failed")
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
