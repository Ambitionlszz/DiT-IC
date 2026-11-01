"""Prepare an audited B+precision run, then use the first idle authorized GPU.

Preparation uses CPU only. The detached watcher does not import torch or
allocate CUDA memory while waiting. A full batch-8 preflight and an actual
trainer startup gate must pass before the first formal update.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
METHOD = "precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4"
GPU_UUID = "GPU-820974f0-0aed-b2b3-236a-04bacb2df8e8"
GPU_IDENTITIES = {1: "GPU-ccb69b0d-acbb-96eb-9f08-b91776b96d65", 2: GPU_UUID}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            value.update(chunk)
    return value.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True, timeout=30).strip()


def prepare(root, initialization, gpus=(1, 2)):
    sys.path.insert(0, str(ROOT))
    import torch
    import yaml
    from models.bt_self_flow.checkpoint import read_checkpoint
    from models.bt_self_flow.data import ImageDataset
    from models.bt_self_flow.experiment import (load_config, make_model, training_contract,
                                               asset_identity, optimizer_groups)
    from train_bt_self_flow import initialize_from

    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise ValueError("plan already exists; use --watch for the existing immutable plan")
    if command_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("prepare requires a clean committed source checkout")
    commit = command_output(["git", "rev-parse", "HEAD"])
    config = load_config(ROOT / "configs/bt_self_flow/precision_v3/mu_pit_b_precision_joint.yaml")
    baseline = load_config(ROOT / "configs/bt_self_flow/precision_v3/c_joint.yaml")
    config["train"]["output"] = str(root / "joint")
    if config["decoder"]["method"] != METHOD:
        raise ValueError("only B with additive precision AdaLN is authorized")
    for key in ("model", "data", "optimizer", "objective", "quality", "image_loss", "self_flow"):
        if config[key] != baseline[key]:
            raise ValueError(f"B differs from native new C: {key}")
    data = ImageDataset(config["data"]["train"], config["data"]["crop"])
    validation = ImageDataset(config["data"]["valid"], config["data"]["valid_crop"], False,
                              config["data"]["valid_limit"])
    contract = training_contract(config, data, 1)
    native_contract = training_contract(baseline, data, 1)
    comparison = deepcopy(contract)
    comparison.pop("decoder")
    if comparison != native_contract:
        raise ValueError("training contract differs beyond D")
    if (config["train"]["batch_size"], config["train"]["accumulation_steps"],
            config["train"]["effective_batch"], config["train"]["steps"]) != (8, 1, 8, 200000):
        raise ValueError("unexpected batch or update contract")
    checkpoint = read_checkpoint(initialization)
    if checkpoint["phase"] != "backend_init" or checkpoint["backend"] != "shared":
        raise ValueError("requires the shared new-C warmup initialization")
    assets = asset_identity(config)
    if checkpoint["assets"] != assets:
        raise ValueError("pretrained assets differ from initialization")
    native = make_model(baseline, torch.device("cpu"))
    native_rng = torch.get_rng_state().clone()
    initialize_from(native, checkpoint, "joint")
    candidate = make_model(config, torch.device("cpu"))
    if not torch.equal(native_rng, torch.get_rng_state()):
        raise ValueError("D initialization changed the shared model RNG stream")
    initialize_from(candidate, checkpoint, "joint")
    matched = {}
    shared = candidate.export_state()
    for part, values in native.export_state().items():
        for name, value in values.items():
            if not torch.equal(value, shared[part][name]):
                raise ValueError(f"shared initial tensor differs: {part}/{name}")
        matched[part] = len(values)
    if candidate.pixel_adapter.qk_source_code.item() != 1:
        raise ValueError("Q/K must come from mu")
    for block in candidate.pixel_adapter.blocks:
        head = block.precision_modulation
        if head.in_features != 128 or head.out_features != 64 or head.weight.count_nonzero() or head.bias.count_nonzero():
            raise ValueError("precision head shape or initialization differs")
    groups, auxiliary = optimizer_groups(candidate, config)
    audit = dict(status="passed", created=now(), commit=commit, matched_initial_tensors=matched,
                 shared_constructor_rng_identical=True, training_contract_matches_c_except_D=True,
                 training_images=len(data), training_dataset_fingerprint=data.fingerprint,
                 validation_images=len(validation), validation_dataset_fingerprint=validation.fingerprint,
                 source_warmup_sha256=checkpoint.get("source_sha256"), initialization_sha256=digest(initialization),
                 assets=assets, precision_head="128->64 per block; zero affine adjustment; mu-only gates and Q/K",
                 optimizer_groups=[dict(name=g["name"], lr=g["lr"], parameters=sum(p.numel() for p in g["params"])) for g in groups],
                 auxiliary_parameters=sum(p.numel() for p in auxiliary), calibration=checkpoint.get("calibration"))
    (root / "b_joint.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / "c_baseline.resolved.json", baseline)
    write_json(root / "strict_ablation_audit.json", audit)
    plan = dict(created=now(), source=str(ROOT), commit=commit, python=sys.executable,
                initialization=str(initialization), initialization_sha256=audit["initialization_sha256"],
                config=str(root / "b_joint.resolved.yaml"), config_sha256=digest(root / "b_joint.resolved.yaml"),
                audit_sha256=digest(root / "strict_ablation_audit.json"), contract=contract, assets=assets,
                gpu_candidates=[dict(index=index, uuid=GPU_IDENTITIES[index]) for index in gpus],
                poll_seconds=5, max_idle_memory_mib=256,
                required_idle_polls=2, target_updates=200000, method=METHOD)
    write_json(root / "plan.json", plan)
    print(json.dumps(audit, indent=2), flush=True)


class Supervisor:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.child = None
        self.stopping = False
        # Old GPU2-only plans remain usable. One supervisor chooses once, then
        # keeps the same physical device for both preflight and formal training.
        self.devices = plan.get("gpu_candidates", [dict(index=plan.get("gpu_index"), uuid=plan.get("gpu_uuid"))])
        self.selected_gpu = None
        self.state = dict(supervisor_pid=os.getpid(), started=now(), gpu_candidates=self.devices,
                          decoder_method=METHOD)

    def publish(self, stage, **values):
        changed = self.state.get("stage") != stage
        self.state.update(stage=stage, updated=now(), **values)
        write_json(self.root / "status.json", self.state)
        if changed:
            print(json.dumps(self.state), flush=True)

    def stop(self, signum=None, frame=None):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            # Signal only the child created by this supervisor. The trainer
            # handles SIGTERM by saving a complete paused checkpoint.
            self.child.terminate()

    def check_stop(self):
        if (self.root / "STOP").exists() and not self.stopping:
            self.stop()
        return self.stopping

    def gpu(self, device):
        values = command_output(["nvidia-smi", "-i", device["uuid"],
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]).split(",")
        index, uuid, used, total, utilization = [v.strip() for v in values]
        if int(index) != device["index"] or uuid != GPU_IDENTITIES.get(int(index)):
            raise ValueError("authorized physical GPU identity changed")
        return dict(index=int(index), uuid=uuid, used_mib=int(used), total_mib=int(total), utilization=int(utilization))

    def wait_idle(self):
        devices = [self.selected_gpu] if self.selected_gpu else self.devices
        idle = {device["index"]: 0 for device in devices}
        while not self.check_stop():
            observations = []
            for device in devices:
                gpu = self.gpu(device)
                index = device["index"]
                idle[index] = idle[index] + 1 if gpu["used_mib"] <= self.plan["max_idle_memory_mib"] and gpu["utilization"] <= 5 else 0
                observations.append(gpu)
                if idle[index] >= self.plan["required_idle_polls"]:
                    self.selected_gpu = device
                    write_json(self.root / "selected_gpu.json", dict(**device, selected=now()))
                    self.publish("gpu_selected", gpu_index=index, gpu_uuid=device["uuid"],
                                 gpu=gpu, consecutive_idle_polls=idle)
                    return True
            self.publish("waiting_for_gpu", gpu_observations=observations, consecutive_idle_polls=idle)
            time.sleep(self.plan["poll_seconds"])
        return False

    def environment(self):
        if self.selected_gpu is None:
            raise ValueError("select one idle authorized GPU before launching a child")
        env = os.environ.copy()
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=self.selected_gpu["uuid"], OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
                   C_V3_SUPERVISOR_PID=str(os.getpid()))
        return env

    def launch(self, command, log, stage):
        with (self.root / log).open("ab", buffering=0) as stream:
            self.child = subprocess.Popen(command, cwd=self.plan["source"], env=self.environment(),
                                          stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        self.publish(stage, child_pid=self.child.pid, command=command, child_log=str(self.root / log))

    def await_child(self, stage):
        while self.child.poll() is None:
            self.check_stop()
            self.publish(stage)
            time.sleep(self.plan["poll_seconds"])
        code = self.child.returncode
        if code:
            raise RuntimeError(f"{stage} exited with {code}; inspect {self.state['child_log']}")
        return code

    def verify_plan(self):
        if (self.plan["method"] != METHOD or not self.devices
                or len({device["index"] for device in self.devices}) != len(self.devices)
                or any(device["uuid"] != GPU_IDENTITIES.get(device["index"]) for device in self.devices)):
            raise ValueError("unexpected experiment or device")
        if command_output(["git", "rev-parse", "HEAD"]) != self.plan["commit"]:
            raise ValueError("source commit changed")
        if command_output(["git", "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError("tracked source changed")
        for path, expected in ((self.plan["config"], self.plan["config_sha256"]),
                               (self.plan["initialization"], self.plan["initialization_sha256"]),
                               (self.root / "strict_ablation_audit.json", self.plan["audit_sha256"])):
            if digest(path) != expected:
                raise ValueError(f"audited file changed: {path}")

    def run(self):
        self.verify_plan()
        if not self.wait_idle():
            self.publish("cancelled")
            return
        self.verify_plan()
        report = self.root / "preflight/full_batch8.json"
        report.parent.mkdir(exist_ok=True)
        command = [self.plan["python"], "-u", "scripts/check_pixel_adapter.py", "--scope", "full",
                   "--config", self.plan["config"], "--device", "cuda", "--batch-size", "8", "--updates", "3",
                   "--init-checkpoint", self.plan["initialization"], "--output", str(report)]
        self.launch(command, "preflight/full_batch8.log", "preflight")
        self.await_child("preflight")
        if self.check_stop():
            self.publish("cancelled")
            return
        result = json.loads(report.read_text())
        if result.get("status") != "passed" or result.get("decoder_method") != METHOD or result.get("batch_size") != 8:
            raise ValueError("full batch-8 precision B preflight failed")
        if result.get("initialization_sha256") != self.plan["initialization_sha256"]:
            raise ValueError("preflight used a different initialization")
        if not self.wait_idle():
            self.publish("cancelled")
            return
        self.verify_plan()
        ready, gate = self.root / "startup_ready.json", self.root / "startup_gate.json"
        if ready.exists() or gate.exists() or (self.root / "joint/resolved.json").exists():
            raise ValueError("training was already initialized; explicit resume is required")
        command = [self.plan["python"], "-u", "train_bt_self_flow.py", "--config", self.plan["config"],
                   "--device", "cuda", "--init-checkpoint", self.plan["initialization"], "--checkpoint-steps", "10",
                   "--startup-ready", str(ready), "--startup-gate", str(gate)]
        self.launch(command, "train.stdout.log", "training_initializing")
        started = time.monotonic()
        while not ready.exists():
            if self.child.poll() is not None:
                raise RuntimeError("trainer exited before startup gate; inspect train.stdout.log")
            if self.check_stop():
                self.await_child("stopping")
                self.publish("cancelled")
                return
            if time.monotonic() - started > 1800:
                self.stop()
                raise TimeoutError("trainer startup exceeded 30 minutes")
            time.sleep(1)
        resolved = json.loads((self.root / "joint/resolved.json").read_text())
        readiness = json.loads(ready.read_text())
        if (readiness.get("pid"), readiness.get("batch_size"), readiness.get("accumulation_steps"),
                readiness.get("ready_for_updates")) != (self.child.pid, 8, 1, True):
            self.stop()
            raise ValueError("trainer readiness identity differs")
        if resolved["contract"] != self.plan["contract"] or resolved["assets"] != self.plan["assets"]:
            self.stop()
            raise ValueError("actual trainer contract/dataset/assets differ from audited plan")
        if resolved["git_commit"] != self.plan["commit"] or resolved["config"]["decoder"]["method"] != METHOD:
            self.stop()
            raise ValueError("actual trainer source/method differs")
        write_json(gate, dict(train=True, verified=now(), decoder_method=METHOD))
        self.publish("training", startup_verified=True, preflight=str(report))
        self.await_child("training")
        self.publish("paused" if self.stopping else "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--gpus", type=int, nargs="+", choices=(1, 2), default=[1, 2],
                        help="physical candidates for a new plan; select exactly one when it becomes idle")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    os.chdir(ROOT)
    if args.prepare_only:
        if not args.init_checkpoint:
            parser.error("--prepare-only requires --init-checkpoint")
        if len(args.gpus) != len(set(args.gpus)):
            parser.error("GPU candidates must be unique")
        prepare(root, Path(args.init_checkpoint).resolve(), args.gpus)
        return
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((root / "plan.json").read_text())
        if str(ROOT) != plan["source"]:
            raise ValueError("watch from the same pinned checkout used for preparation")
        watcher = Supervisor(root, plan)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, watcher.stop)
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        try:
            watcher.run()
        except Exception as error:
            watcher.stop()
            watcher.publish("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
