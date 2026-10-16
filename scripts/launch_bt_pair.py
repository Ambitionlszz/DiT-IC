"""Durable, restartable common -> shared init -> parallel C/D experiment.

Run this in tmux. The first invocation freezes resolved configs and the commit;
subsequent invocations resume the latest saved update in the same run root.
An active lock prevents duplicate supervisors. This script never stops other jobs.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.experiment import load_config


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def resume_checkpoint(directory):
    candidates = [p for p in directory.glob("*.pt") if p.stem.isdecimal() or p.name == "paused.pt"]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None


class PairedRun:
    def __init__(self, root, gpus, python):
        self.root, self.gpus, self.python = root, gpus, python
        self.children, self.stop_requested = {}, False
        self.state = {"supervisor_pid": os.getpid(), "started_at": self.now(), "status": "starting", "jobs": {}}
        self.env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        PYTHONUNBUFFERED="1", PYTHONPATH=str(REPO)+os.pathsep+os.environ.get("PYTHONPATH", ""))
        self.env.setdefault("TORCH_HOME", "/data2/ls/checkpoints/torch-hub")
        self.env.setdefault("HF_HOME", "/data2/ls/checkpoints/hf-cache")

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def update(self, status=None):
        if status:
            self.state["status"] = status
        self.state["updated_at"] = self.now()
        write_json(self.root/"pipeline.json", self.state)

    def start(self, name, command, gpus):
        if self.stop_requested:
            raise InterruptedError("supervisor stop requested")
        log_path = self.root/"logs"/f"{name}.log"
        env = dict(self.env, CUDA_VISIBLE_DEVICES=",".join(gpus))
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.children[name] = process
        self.state["jobs"][name] = {"pid": process.pid, "gpus": gpus, "command": command,
                                    "log": str(log_path), "started_at": self.now(), "status": "running"}
        self.update()

    def finish(self, name):
        process = self.children[name]
        code = process.poll()
        if code is None:
            return False
        job = self.state["jobs"][name]
        job.update(status="completed" if code == 0 else "failed", returncode=code, ended_at=self.now())
        del self.children[name]
        self.update()
        if code:
            raise RuntimeError(f"{name} exited {code}; see {job['log']}")
        if self.stop_requested:
            raise InterruptedError("supervisor stop requested")
        return True

    def wait(self, name):
        while not self.finish(name):
            time.sleep(2)

    def command(self, name, directory, initialization=None, distributed=False):
        prefix = [self.python]
        if distributed:
            prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command = prefix+["train_bt_self_flow.py", "--config", str(self.root/"configs"/f"{name}.yaml")]
        latest = resume_checkpoint(directory)
        if latest:
            command += ["--resume", str(latest)]
        elif initialization:
            command += ["--init-checkpoint", str(initialization)]
        return command

    def execute(self):
        common = self.root/"common/complete.pt"
        if not common.exists():
            self.update("common_frontend")
            self.start("common", self.command("common", common.parent, distributed=True), self.gpus)
            self.wait("common")
            if not common.exists():
                raise InterruptedError("common phase paused; restart supervisor to resume")
        shared = self.root/"backend_init.pt"
        if not shared.exists():
            self.update("shared_initialization")
            self.start("prepare", [self.python, "train_bt_self_flow.py", "--config",
                                    str(self.root/"configs/c_joint.yaml"), "--prepare-backend", str(common),
                                    "--output", str(shared)], [self.gpus[0]])
            self.wait("prepare")
        self.update("paired_training")
        phases = {}

        def start_joint(backend):
            directory = self.root/backend/"joint"
            if (directory/"complete.pt").exists():
                return
            name = f"{backend.lower()}_joint"
            self.start(name, self.command(name, directory, shared),
                       [self.gpus[0 if backend == "C" else 1]])
            phases[backend] = (name, directory)

        for backend in ("C", "D"):
            start_joint(backend)
        while phases:
            for backend, (name, directory) in list(phases.items()):
                if self.finish(name):
                    if not (directory/"complete.pt").exists():
                        raise InterruptedError(f"{name} paused; restart supervisor to resume")
                    phases.pop(backend)
            time.sleep(2)
        self.update("completed")

    def stop(self, signum, frame):
        self.stop_requested = True
        for name, process in list(self.children.items()):
            if process.poll() is None:
                targets = [process.pid]
                if name == "common":
                    # Signal the actual DDP trainers, keeping torchrun alive to
                    # wait for their multi-GB paused checkpoint. Signalling the
                    # launcher starts its finite shutdown/kill timeout instead.
                    children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
                    workers = []
                    if children.exists():
                        for child in children.read_text().split():
                            try:
                                command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                            except FileNotFoundError:
                                continue
                            if any(arg.rsplit(b"/", 1)[-1] == b"train_bt_self_flow.py" for arg in command):
                                workers.append(int(child))
                    if workers:
                        targets = workers
                # Do not signal the process group: DataLoader children need to
                # stay alive until the trainer finishes its accumulated update.
                for pid in targets:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def reap_stopped(self):
        """Publish each peer's real exit status after an interrupted pair."""
        for name, process in list(self.children.items()):
            code = process.wait()
            directory = self.root/"common" if name == "common" else self.root/name[0].upper()/"joint"
            status = "failed"
            if code == 0:
                status = "completed" if (directory/"complete.pt").exists() else "paused"
            self.state["jobs"][name].update(status=status, returncode=code, ended_at=self.now())
            del self.children[name]
        self.update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdecimal() for g in gpus):
        parser.error("exactly two distinct physical GPU indices are required")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        changes = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "status", "--porcelain"], cwd=REPO, text=True)
        if changes.strip():
            raise RuntimeError("commit the reviewed code before launching or resuming a research run")
        manifest = {"repository": str(REPO), "commit": commit, "python": args.python, "gpus": gpus}
        manifest_path = root/"manifest.json"
        configs = {}
        for name in ("common", "c_joint", "d_joint"):
            config = load_config(REPO/"configs/bt_self_flow"/f"{name}.yaml")
            phase, backend = config["train"]["phase"], config["model"]["backend"]
            config["train"]["output"] = str(root/("common" if phase == "codec" else f"{backend}/{phase}"))
            configs[name] = config
        manifest["configs"] = configs
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("run manifest differs; do not silently change a paired experiment")
        else:
            write_json(manifest_path, manifest)
        (root/"configs").mkdir(exist_ok=True)
        (root/"logs").mkdir(exist_ok=True)
        for name, config in configs.items():
            config_path = root/"configs"/f"{name}.yaml"
            text = yaml.safe_dump(config, sort_keys=False)
            if config_path.exists() and config_path.read_text() != text:
                raise RuntimeError(f"frozen config changed: {config_path}")
            config_path.write_text(text)
        run = PairedRun(root, gpus, args.python)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, run.stop)
        try:
            run.execute()
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested or isinstance(error, InterruptedError) else "failed")
            # A failed arm must not silently leave its peer running unsupervised.
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
