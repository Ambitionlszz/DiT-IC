"""Two-GPU scratch warmup, then two C runs differing only in auxiliary losses.

Run in tmux. Configs and code are frozen in the manifest. Restart the same
command to resume each stage. A failed C arm lets the other arm finish.
"""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.experiment import load_config
from scripts.launch_bt_pair import PairedRun, write_json


ARMS = {"c_joint": "C_aux", "c_no_latent": "C_no_latent"}


def resolved_configs(root):
    configs = {}
    for name, recipe in (("common", "lowrate_common"), ("c_joint", "lowrate_c_joint"),
                         ("c_no_latent", "lowrate_c_no_latent")):
        config = load_config(REPO/"configs/bt_self_flow"/f"{recipe}.yaml")
        directory = root/"common" if name == "common" else root/ARMS[name]/"joint"
        config["train"]["output"] = str(directory)
        configs[name] = config
    keep, remove = configs["c_joint"], configs["c_no_latent"]
    comparison = deepcopy(remove)
    comparison["train"]["output"] = keep["train"]["output"]
    comparison["image_loss"].update(base_weight=.1, latent_weight=.1)
    if comparison != keep or any(remove["image_loss"][key] != 0 for key in ("base_weight", "latent_weight")):
        raise ValueError("C recipes must differ only in output and the two auxiliary weights")
    common = configs["common"]
    for config in (keep, remove):
        if config["model"]["backend"] != "C" or config["self_flow"].get("enabled"):
            raise ValueError("this ablation requires C without Self-Flow")
        if config["model"]["codec"] != common["model"]["codec"]:
            raise ValueError("warmup and joint stages must use identical quantization settings")
    return configs


class CAblationRun(PairedRun):
    def directory(self, name):
        return self.root/"common" if name == "common" else self.root/ARMS[name]/"joint"

    def exit_status(self, name, code):
        if code:
            return "failed"
        if name == "prepare":
            return "completed"
        return "completed" if (self.directory(name)/"complete.pt").exists() else "paused"

    def train_arms(self, shared):
        pending, failures = {}, {}
        self.update("paired_c_training")
        for (name, _), gpu in zip(ARMS.items(), self.gpus):
            directory = self.directory(name)
            if (directory/"complete.pt").exists():
                self.state["jobs"][name] = {"status": "completed", "gpus": [gpu]}
                continue
            self.start(name, self.command(name, directory, shared), [gpu])
            pending[name] = directory
        while pending:
            for name, directory in list(pending.items()):
                try:
                    done = self.finish(name)
                except RuntimeError as error:
                    failures[name] = str(error)
                    pending.pop(name)
                    self.state["failures"] = failures
                    self.update("degraded_peer_continues")
                    continue
                if done:
                    pending.pop(name)
                    if not (directory/"complete.pt").exists():
                        failures[name] = "trainer paused before its configured endpoint"
                        self.state["failures"] = failures
                        self.update("degraded_peer_continues")
            if pending:
                time.sleep(2)
        if failures:
            raise RuntimeError(f"C experiment failures: {failures}")
        self.update("completed")

    def execute(self):
        common = self.root/"common/complete.pt"
        if not common.exists():
            self.update("common_frontend")
            self.start("common", self.command("common", common.parent, distributed=True), self.gpus)
            self.wait("common")
            if not common.exists():
                raise InterruptedError("common warmup paused; restart supervisor to resume")
        shared = self.root/"backend_init.pt"
        if not shared.exists():
            self.update("shared_initialization")
            self.start("prepare", [self.python, "train_bt_self_flow.py", "--config",
                                    str(self.root/"configs/c_joint.yaml"), "--prepare-backend", str(common),
                                    "--output", str(shared)], [self.gpus[0]])
            self.wait("prepare")
        self.train_arms(shared)


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
        git = ["git", "-c", f"safe.directory={REPO}"]
        commit = subprocess.check_output(git+["rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if subprocess.check_output(git+["status", "--porcelain"], cwd=REPO, text=True).strip():
            raise RuntimeError("commit reviewed code before launching or resuming")
        configs = resolved_configs(root)
        manifest = {"repository": str(REPO), "commit": commit, "python": args.python, "gpus": gpus,
                    "experiment": "lowrate_c_auxiliary_ablation", "configs": configs}
        manifest_path = root/"manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("run manifest differs; use a new run root")
        write_json(manifest_path, manifest)
        (root/"configs").mkdir(exist_ok=True)
        (root/"logs").mkdir(exist_ok=True)
        for name, config in configs.items():
            path = root/"configs"/f"{name}.yaml"
            text = json.dumps(config, indent=2)+"\n"  # JSON is a YAML subset.
            if path.exists() and path.read_text() != text:
                raise RuntimeError(f"frozen config changed: {path}")
            path.write_text(text)
        run = CAblationRun(root, gpus, args.python)
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
