"""Restartable two-GPU precision C: latent warmup, checked handoff, 200K C."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.experiment import check_stage_model, load_config
from scripts.launch_bt_pair import PairedRun, write_json


def resolved_configs(root, smoke=False):
    recipe = REPO/"configs/bt_self_flow/precision_v2"
    configs = {"common": load_config(recipe/"common.yaml"),
               "c_joint": load_config(recipe/"c_joint_ddp.yaml")}
    for name, config in configs.items():
        config["train"]["output"] = str(root/("common" if name == "common" else "C/joint"))
        if smoke:
            config["train"].update(steps=14, eval_every=14, save_every=14, log_every=1,
                                   data_scale_samples=2, calibration_samples=4,
                                   require_quality_span=False, packet_samples=24)
            config["data"]["workers"] = 0
    common, joint = configs["common"], configs["c_joint"]
    check_stage_model(joint, common)
    for config in configs.values():
        settings = config["train"]
        if 2*settings["batch_size"]*settings["accumulation_steps"] != settings["effective_batch"]:
            raise ValueError("two-rank effective batch contract differs")
    return configs


def check_handoff(common, shared, config):
    source, initialization = read_checkpoint(common), read_checkpoint(shared)
    if source["phase"] != "codec" or source["step"] != source["config"]["train"]["steps"]:
        raise ValueError("latent warmup is incomplete")
    calibration = source.get("calibration", {})
    if not calibration.get("rate_span_pass"):
        raise ValueError("latent warmup failed actual bitrate span")
    if config["train"].get("require_quality_span", False) and not calibration.get("quality_span_pass"):
        raise ValueError("latent warmup failed quality span")
    if initialization["phase"] != "backend_init" or initialization["backend"] != "shared":
        raise ValueError("invalid C initialization")
    if initialization.get("codec_weights") != "ema" or calibration.get("weights") != "deployment_ema":
        raise ValueError("C requires the calibrated deployment EMA frontend")
    if initialization["assets"] != source["assets"]:
        raise ValueError("pretrained assets differ at handoff")
    check_stage_model(config, source["config"])
    for name, tensor in source["ema"]["codec"].items():
        if not tensor.equal(initialization["raw"]["codec"][name]):
            raise ValueError(f"shared initialization differs from warmup EMA: {name}")
    if not initialization["raw"]["distortion_scale"]["calibrated"].item():
        raise ValueError("shared distortion calibration is missing")
    return dict(step=source["step"], rate_span_pass=True,
                quality_span_pass=calibration.get("quality_span_pass"), codec_weights="ema")


class PrecisionCRun(PairedRun):
    def __init__(self, root, gpus, python, configs, reserve_gib=18., smoke=False):
        super().__init__(root, gpus, python)
        self.configs, self.reserve_gib, self.smoke = configs, reserve_gib, smoke
        self.reservations = {}

    def start_reservations(self):
        if self.reservations or self.reserve_gib == 0:
            return
        for index, gpu in enumerate(self.gpus):
            status = self.root/f"reserve_gpu{index}.json"
            status.unlink(missing_ok=True)
            command = [self.python, str(REPO/"scripts/reserve_gpu_memory.py"),
                       "--gib", str(self.reserve_gib), "--status", str(status),
                       "--parent-pid", str(os.getpid())]
            with (self.root/"logs"/f"reserve_gpu{index}.log").open("a") as log:
                process = subprocess.Popen(command, cwd=REPO,
                                           env=dict(self.env, CUDA_VISIBLE_DEVICES=gpu),
                                           stdout=log, stderr=subprocess.STDOUT)
            self.reservations[gpu] = (process, status)
        deadline = time.monotonic()+60
        while True:
            if any(process.poll() is not None for process, _ in self.reservations.values()):
                raise RuntimeError("memory reservation failed; inspect reservation logs")
            if all(status.exists() for _, status in self.reservations.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("memory reservation readiness timed out")
            time.sleep(.5)
        self.state["reservations"] = {gpu: json.loads(status.read_text())
                                      for gpu, (_, status) in self.reservations.items()}
        self.update()

    def stop_reservations(self):
        for process, _ in self.reservations.values():
            if process.poll() is None:
                process.terminate()
        for process, status in self.reservations.values():
            process.wait(timeout=30)
            status.unlink(missing_ok=True)
        self.reservations.clear()
        self.state["reservations"] = {}
        self.update()

    def wait(self, name):
        while not self.finish(name):
            if any(process.poll() is not None for process, _ in self.reservations.values()):
                raise RuntimeError("warmup reservation exited unexpectedly")
            time.sleep(2)

    def stop(self, signum, frame):
        self.stop_requested = True
        for process in list(self.children.values()):
            if process.poll() is not None:
                continue
            workers = []
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            if children.exists():
                for pid in children.read_text().split():
                    try:
                        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    except FileNotFoundError:
                        continue
                    if any(arg.rsplit(b"/", 1)[-1] == b"train_bt_self_flow.py" for arg in command):
                        workers.append(int(pid))
            for pid in workers or [process.pid]:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def train_stage(self, name, directory, initialization=None):
        if (directory/"complete.pt").exists():
            return
        command = self.command(name, directory, initialization, distributed=True)
        if self.smoke and not (directory/"paused.pt").exists() and "--resume" not in command:
            self.start(name, command+["--max-updates", "2"], self.gpus)
            self.wait(name)
            if not (directory/"paused.pt").exists():
                raise RuntimeError("smoke did not publish a pause checkpoint")
            command = self.command(name, directory, initialization, distributed=True)
        self.start(name, command, self.gpus)
        self.wait(name)
        if not (directory/"complete.pt").exists():
            raise InterruptedError(f"{name} paused; restart the same command to resume")

    def execute(self):
        common, shared = self.root/"common/complete.pt", self.root/"backend_init.pt"
        if not common.exists():
            self.update("latent_warmup_ddp")
            self.start_reservations()
            self.train_stage("common", common.parent)
        self.update("checking_handoff")
        if not shared.exists():
            self.start("prepare", [self.python, "train_bt_self_flow.py", "--config",
                                  str(self.root/"configs/c_joint.yaml"), "--prepare-backend", str(common),
                                  "--output", str(shared)], [self.gpus[0]])
            self.wait("prepare")
        self.state["handoff"] = check_handoff(common, shared, self.configs["c_joint"])
        self.update("handoff_passed")
        self.stop_reservations()
        if self.stop_requested:
            raise InterruptedError("stop requested before C joint")
        self.update("c_joint_ddp")
        self.train_stage("c_joint", self.root/"C/joint", shared)
        self.update("completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="two physical indices or UUIDs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reserve-gib", type=float, default=18.)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or args.reserve_gib < 0:
        parser.error("two distinct GPUs and nonnegative reservation are required")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise RuntimeError("commit reviewed source before running the pipeline")
        configs = resolved_configs(root, args.smoke)
        manifest = dict(repository=str(REPO), commit=commit, python=args.python, gpus=gpus,
                        reserve_gib=args.reserve_gib, smoke=args.smoke, configs=configs)
        path = root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("run manifest differs; choose a new root")
        write_json(path, manifest)
        for directory in ("configs", "logs"):
            (root/directory).mkdir(exist_ok=True)
        for name, config in configs.items():
            write_json(root/"configs"/f"{name}.yaml", config)
        run = PrecisionCRun(root, gpus, args.python, configs, args.reserve_gib, args.smoke)
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
        finally:
            run.stop_reservations()


if __name__ == "__main__":
    main()
