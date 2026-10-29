"""Share two GPUs between matched C/SF continuations at validation boundaries."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from scripts.launch_bt_pair import write_json
from scripts.launch_quantization_residual import ResidualRun, resume_checkpoint
from scripts.prepare_c_sf_ablation import paired_configs


def next_segment(progress, total, interval):
    candidates = [name for name in ("c_sf", "pure_c") if progress[name] < total]
    if not candidates:
        return None
    name = min(candidates, key=lambda name: progress[name])
    return name, min(total, (progress[name]//interval+1)*interval)


class AblationRun(ResidualRun):
    def stop(self, signum, frame):
        self.stop_requested = True
        # Our direct children are supervisors. Let each supervisor signal its
        # trainer ranks and wait for the durable checkpoint; bypassing it to
        # terminate torchrun would start the launcher's finite kill timeout.
        for process in list(self.children.values()):
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def exit_status(self, name, code):
        return "failed" if code else "completed" if (self.root/name/"joint/complete.pt").exists() else "paused"

    def execute(self, configs, source, interval):
        while not self.stop_requested:
            progress = {}
            for name in configs:
                checkpoint = resume_checkpoint(self.root/name/"joint")
                if (self.root/name/"joint/complete.pt").exists():
                    checkpoint = self.root/name/"joint/complete.pt"
                if checkpoint is None:
                    raise ValueError(f"migrate the source checkpoint for {name} before launching")
                progress[name] = read_checkpoint(checkpoint)["step"]
            self.state["progress"] = progress
            selected = next_segment(progress, configs["pure_c"]["train"]["steps"], interval)
            if selected is None:
                self.update("completed")
                return
            name, target = selected
            self.state["active_branch"], self.state["segment_target"] = name, target
            self.update("training")
            command = [self.python, "scripts/launch_quantization_residual.py", "--root", str(self.root/name),
                       "--config", str(self.root/"configs"/f"{name}.json"), "--init-checkpoint", str(source),
                       "--gpus", ",".join(self.gpus), "--max-updates", str(target)]
            self.start(name, command, self.gpus)
            self.wait(name)
            checkpoint = resume_checkpoint(self.root/name/"joint")
            if (self.root/name/"joint/complete.pt").exists():
                checkpoint = self.root/name/"joint/complete.pt"
            if read_checkpoint(checkpoint)["step"] < target:
                raise RuntimeError("segment exited before its target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--interval", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(item.isdecimal() for item in gpus):
        raise ValueError("two distinct GPU indices are required")
    if args.interval <= 0 or args.interval % 2000:
        raise ValueError("segments must end on a 2K validation boundary")
    with (root/"ablation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        git = ["git", "-c", f"safe.directory={ROOT}"]
        if subprocess.check_output(git+["status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise RuntimeError("commit reviewed code before launching")
        configs = paired_configs()
        for name, config in configs.items():
            config["train"]["output"] = str(root/name/"joint")
        manifest = dict(repository=str(ROOT),
                        commit=subprocess.check_output(git+["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        source=str(args.checkpoint.resolve()), source_sha256=sha256_file(args.checkpoint),
                        gpus=gpus, interval=args.interval, configs=configs)
        frozen = root/"ablation_manifest.json"
        if frozen.exists() and json.loads(frozen.read_text()) != manifest:
            raise ValueError("ablation code/config/source differs from frozen manifest")
        (root/"configs").mkdir(exist_ok=True)
        (root/"logs").mkdir(exist_ok=True)
        write_json(frozen, manifest)
        for name, config in configs.items():
            write_json(root/"configs"/f"{name}.json", config)
        run = AblationRun(root, gpus, sys.executable)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, run.stop)
        try:
            run.execute(configs, args.checkpoint.resolve(), args.interval)
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested else "failed")
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
