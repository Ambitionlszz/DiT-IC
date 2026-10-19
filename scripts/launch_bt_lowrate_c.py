"""One two-GPU low-rate codec warmup followed by one two-GPU no-aux C run."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.experiment import load_config
from scripts.launch_bt_pair import PairedRun, write_json


def resolved_configs(root):
    common = load_config(REPO/"configs/bt_self_flow/lowrate_common_ddp.yaml")
    joint = load_config(REPO/"configs/bt_self_flow/lowrate_c_no_aux_ddp.yaml")
    common["train"]["output"] = str(root/"common")
    joint["train"]["output"] = str(root/"C/joint")
    if tuple(common["train"][key] for key in ("batch_size", "accumulation_steps", "effective_batch")) != (32, 1, 64):
        raise ValueError("warmup requires per-rank batch 32, accumulation 1, effective batch 64")
    if tuple(joint["train"][key] for key in ("batch_size", "accumulation_steps", "effective_batch")) != (2, 2, 8):
        raise ValueError("C requires per-rank batch 2, accumulation 2, effective batch 8")
    if common["model"]["codec"] != joint["model"]["codec"]:
        raise ValueError("warmup and C quantization settings differ")
    if any(joint["image_loss"][key] != 0 for key in ("base_weight", "latent_weight")):
        raise ValueError("both C auxiliary losses must be disabled")
    return {"common": common, "c_joint": joint}


class SingleCRun(PairedRun):
    def directory(self, name):
        return self.root/"common" if name == "common" else self.root/"C/joint"

    def start(self, name, command, gpus):
        if self.stop_requested:
            raise InterruptedError("supervisor stop requested")
        log_path = self.root/"logs"/f"{name}.log"
        env = dict(self.env, CUDA_VISIBLE_DEVICES=",".join(gpus))
        # Stay in the guardian's pipeline process group so orphaned torchrun,
        # ranks and DataLoader workers are also cleaned up after a crash.
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=False)
        self.children[name] = process
        self.state["jobs"][name] = dict(pid=process.pid, gpus=gpus, command=command,
                                        log=str(log_path), started_at=self.now(), status="running")
        self.update()

    def exit_status(self, name, code):
        if code:
            return "failed"
        return "completed" if name == "prepare" or (self.directory(name)/"complete.pt").exists() else "paused"

    def stop(self, signum, frame):
        self.stop_requested = True
        for process in list(self.children.values()):
            if process.poll() is not None:
                continue
            targets = []
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            if children.exists():
                for child in children.read_text().split():
                    try:
                        command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                    except FileNotFoundError:
                        continue
                    if any(arg.rsplit(b"/", 1)[-1] == b"train_bt_self_flow.py" for arg in command):
                        targets.append(int(child))
            for pid in targets or [process.pid]:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def execute(self):
        common = self.root/"common/complete.pt"
        if not common.exists():
            self.update("common_frontend")
            self.start("common", self.command("common", common.parent, distributed=True), self.gpus)
            self.wait("common")
            if not common.exists():
                raise InterruptedError("warmup paused; restart to resume")
        shared = self.root/"backend_init.pt"
        if not shared.exists():
            self.update("shared_initialization")
            self.start("prepare", [self.python, "train_bt_self_flow.py", "--config",
                                   str(self.root/"configs/c_joint.yaml"), "--prepare-backend", str(common),
                                   "--output", str(shared)], [self.gpus[0]])
            self.wait("prepare")
        directory = self.root/"C/joint"
        if not (directory/"complete.pt").exists():
            self.update("c_joint_ddp")
            self.start("c_joint", self.command("c_joint", directory, shared, distributed=True), self.gpus)
            self.wait("c_joint")
            if not (directory/"complete.pt").exists():
                raise InterruptedError("C joint paused; restart to resume")
        self.update("completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpus", default="0,1", help="two physical indices or GPU UUIDs")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("exactly two distinct GPUs are required")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        git = ["git", "-c", f"safe.directory={REPO}"]
        commit = subprocess.check_output(git+["rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if subprocess.check_output(git+["status", "--porcelain"], cwd=REPO, text=True).strip():
            raise RuntimeError("commit reviewed code before launching or resuming")
        configs = resolved_configs(root)
        manifest = dict(repository=str(REPO), commit=commit, python=args.python, gpus=gpus,
                        experiment="lowrate_c_no_aux_two_gpu", configs=configs)
        path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("run manifest differs; use a new run root")
        write_json(path, manifest)
        (root/"configs").mkdir(exist_ok=True)
        (root/"logs").mkdir(exist_ok=True)
        for name, config in configs.items():
            path = root/"configs"/f"{name}.yaml"
            text = json.dumps(config, indent=2)+"\n"
            if path.exists() and path.read_text() != text:
                raise RuntimeError(f"frozen config changed: {path}")
            path.write_text(text)
        run = SingleCRun(root, gpus, args.python)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, run.stop)
        try:
            run.execute()
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested or isinstance(error, InterruptedError) else "failed")
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
