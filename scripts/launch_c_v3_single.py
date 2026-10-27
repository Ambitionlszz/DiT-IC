"""Single-GPU C v3: protected warmup followed by a gated 200K joint run."""

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

from models.bt_self_flow.experiment import asset_identity, check_stage_model, load_config
from scripts.c_precision_handover import alive, process_state, send
from scripts.launch_c_precision import check_handoff


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def configs(root):
    recipe = REPO/"configs/bt_self_flow/precision_v3"
    values = {"warmup": load_config(recipe/"common.yaml"), "joint": load_config(recipe/"c_joint.yaml")}
    for stage, cfg in values.items():
        cfg["train"]["output"] = str(root/("common" if stage == "warmup" else "C/joint"))
        if (cfg["train"]["batch_size"], cfg["train"]["accumulation_steps"],
                cfg["train"]["effective_batch"]) != (8, 1, 8):
            raise ValueError("both C v3 stages must use single-GPU batch8, accumulation1")
    check_stage_model(values["joint"], values["warmup"])
    return values


def inspect_holder(watcher_pid, holder_pid, gpu_uuid, proc_root=Path("/proc")):
    watcher, holder = process_state(watcher_pid), process_state(holder_pid)
    if not watcher or not holder or watcher["state"] == "Z" or holder["state"] == "Z":
        raise ValueError("the authorized GPU holder/controller is no longer running")
    if not any(Path(arg).name == "watch_two_idle_gpus.py" for arg in watcher["command"]):
        raise ValueError("refusing to stop an unrecognized controller")
    command = watcher["command"]
    if "--slots" not in command or command[command.index("--slots")+1] != "1":
        raise ValueError("controller must manage only the authorized single holder")
    if not any(Path(arg).name == "train_depart.py" for arg in holder["command"]):
        raise ValueError("refusing to stop a training process")
    stat = (proc_root/str(holder_pid)/"stat").read_text().rpartition(")")[2].split()
    environment = (proc_root/str(holder_pid)/"environ").read_bytes().split(b"\0")
    if int(stat[1]) != watcher_pid or f"CUDA_VISIBLE_DEVICES={gpu_uuid}".encode() not in environment:
        raise ValueError("holder parent or exact physical GPU differs")
    return watcher, holder


class Run:
    def __init__(self, args, manifest):
        self.args, self.root, self.manifest = args, args.root, manifest
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu_uuid"],
                        TORCH_HOME=os.environ.get("TORCH_HOME", str(Path.home()/".cache/torch")),
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            self.env.pop(key, None)
        self.guard = self.worker = self.preflight = None
        self.stopping = False
        self.holder_release_requested = False
        self.state = dict(supervisor_pid=os.getpid(), gpu=args.gpu, gpu_uuid=manifest["gpu_uuid"],
                          reserve_gib=args.reserve_gib, commit=manifest["commit"])

    def update(self, status, **values):
        self.state.update(status=status, updated_at_unix=time.time(), **values)
        write_json(self.root/"status.json", self.state)
        print(json.dumps(self.state), flush=True)

    def spawn(self, command, name):
        with (self.root/f"{name}.log").open("a") as log:
            return subprocess.Popen(command, cwd=REPO, env=self.env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)

    def wait_for(self, predicate, processes=(), timeout=1800):
        deadline = time.monotonic()+timeout
        while not predicate():
            if self.stopping:
                raise InterruptedError("supervisor stop requested")
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a gated process exited; inspect its log")
            if self.guard is not None and self.guard.poll() is not None:
                raise RuntimeError("warmup protection exited unexpectedly")
            if time.monotonic() > deadline:
                raise TimeoutError("gated process readiness timed out")
            time.sleep(.01)

    def start_guard(self):
        for name in ("guard.ready.json", "guard.standby.json", "guard.allocate.json"):
            (self.root/name).unlink(missing_ok=True)
        write_json(self.root/"guard.control.json", dict(gib=self.args.reserve_gib))
        self.guard = self.spawn([sys.executable, "-u", str(REPO/"scripts/reserve_gpu_memory.py"),
                                "--gib", str(self.args.reserve_gib), "--minimum-free-gib", "1",
                                "--status", str(self.root/"guard.ready.json"),
                                "--parent-pid", str(os.getpid()),
                                "--allocation-gate", str(self.root/"guard.allocate.json"),
                                "--standby-status", str(self.root/"guard.standby.json"),
                                "--control", str(self.root/"guard.control.json")], "guard")
        self.wait_for(lambda: (self.root/"guard.standby.json").exists(), (self.guard,))
        self.update("protection_standby", guard_pid=self.guard.pid)

    def start_worker(self, stage, initialization=None):
        for suffix in ("allocation.ready.json", "allocation.gate.json", "updates.ready.json", "updates.gate.json"):
            (self.root/f"{stage}.{suffix}").unlink(missing_ok=True)
        command = [sys.executable, "-u", str(REPO/"scripts/c_v3_gated_worker.py"),
                   "--config", str(self.root/f"{stage}.json"),
                   "--allocation-ready", str(self.root/f"{stage}.allocation.ready.json"),
                   "--allocation-gate", str(self.root/f"{stage}.allocation.gate.json"),
                   "--parent-pid", str(os.getpid()),
                   "--startup-ready", str(self.root/f"{stage}.updates.ready.json"),
                   "--startup-gate", str(self.root/f"{stage}.updates.gate.json")]
        if stage == "warmup":
            command += ["--memory-limit-gib", str(self.args.warmup_memory_gib),
                        "--initial-state", str(self.root/"initial_codec.pt")]
        if initialization:
            command += ["--init-checkpoint", str(initialization)]
        self.worker = self.spawn(command, stage)
        self.update("preloading", stage=stage, worker_pid=self.worker.pid, command=command)
        self.wait_for(lambda: (self.root/f"{stage}.allocation.ready.json").exists(), (self.worker,))

    def start_preflight(self):
        command = [sys.executable, "-u", str(REPO/"scripts/check_c_precision_full.py"),
                   "--config", str(self.root/"joint.json"), "--device", "cuda", "--batch-size", "8",
                   "--output", str(self.root/"preflight/full_cuda_b8.json"),
                   "--allocation-ready", str(self.root/"preflight.allocation.ready.json"),
                   "--allocation-gate", str(self.root/"preflight.allocation.gate.json"),
                   "--compute-ready", str(self.root/"preflight.compute.ready.json"),
                   "--compute-gate", str(self.root/"preflight.compute.gate.json"),
                   "--hold-ready", str(self.root/"preflight.hold.ready.json"),
                   "--hold-release", str(self.root/"preflight.hold.release.json"),
                   "--parent-pid", str(os.getpid())]
        self.preflight = self.spawn(command, "preflight_gpu")
        self.update("preloading_gpu_check", preflight_pid=self.preflight.pid)
        self.wait_for(lambda: (self.root/"preflight.allocation.ready.json").exists(), (self.preflight,))

    def stop_guard(self):
        guard, self.guard = self.guard, None
        if guard is not None:
            guard.terminate()
            guard.wait(timeout=10)

    def resize_guard(self, gib):
        write_json(self.root/"guard.control.json", dict(gib=gib))
        def ready():
            path = self.root/"guard.ready.json"
            if not path.exists():
                return False
            value = json.loads(path.read_text())
            return value.get("ready") and value["allocated_bytes"] == int(gib*1024**3)
        self.wait_for(ready, (self.guard,), timeout=30)

    def restore_protection(self):
        if self.guard is None or self.guard.poll() is not None:
            self.start_guard()
            write_json(self.root/"guard.allocate.json", dict(allocate=True))
        self.resize_guard(self.args.reserve_gib)

    def take_over(self):
        required = ("guard.standby.json", "warmup.allocation.ready.json", "preflight.allocation.ready.json")
        if not all((self.root/name).is_file() for name in required):
            raise ValueError("all replacement CUDA processes must be ready before holder release")
        watcher, holder = inspect_holder(self.args.replace_watcher_pid, self.args.replace_holder_pid,
                                        self.manifest["gpu_uuid"])
        send(watcher, signal.SIGSTOP)
        try:
            self.holder_release_requested = True
            send(holder, signal.SIGTERM)
            self.wait_for(lambda: not alive(holder), timeout=30)
            released = time.monotonic()
            write_json(self.root/"guard.allocate.json", dict(allocate=True))
            self.wait_for(lambda: (self.root/"guard.ready.json").exists(), (self.guard,), timeout=30)
            protected = time.monotonic()
            write_json(self.root/"handover.json", dict(previous_watcher=watcher, previous_holder=holder,
                       replacement_guard_pid=self.guard.pid, preloaded_worker_pid=self.worker.pid,
                       seconds_from_holder_exit_to_guard_ready=protected-released,
                       worker_cuda_allocation_retained=True))
        finally:
            send(watcher, signal.SIGTERM)
            send(watcher, signal.SIGCONT)
        self.update("protected", guard_pid=self.guard.pid)

    def open_allocation(self, stage):
        write_json(self.root/f"{stage}.allocation.gate.json", dict(allocate=True))
        self.wait_for(lambda: (self.root/f"{stage}.updates.ready.json").exists(), (self.worker,))

    def wait_worker(self, stage):
        self.wait_for(lambda: self.worker.poll() is not None, timeout=365*24*3600)
        if self.worker.returncode:
            raise RuntimeError(f"{stage} exited {self.worker.returncode}; inspect {stage}.log")
        directory = self.root/("common" if stage == "warmup" else "C/joint")
        if not (directory/"complete.pt").is_file():
            raise RuntimeError(f"{stage} did not complete; a paused checkpoint may be available")

    def execute(self):
        self.start_guard()
        self.start_worker("warmup")
        self.start_preflight()
        self.take_over()
        write_json(self.root/"preflight.allocation.gate.json", dict(allocate=True))
        self.wait_for(lambda: (self.root/"preflight.compute.ready.json").exists(), (self.preflight,))
        self.resize_guard(0)
        write_json(self.root/"preflight.compute.gate.json", dict(check=True))
        self.update("checking_cuda_batch8", stage="preflight")
        self.wait_for(lambda: (self.root/"preflight.hold.ready.json").exists(), (self.preflight,))
        self.resize_guard(self.args.reserve_gib)
        write_json(self.root/"preflight.hold.release.json", dict(release=True))
        self.wait_for(lambda: self.preflight.poll() is not None)
        if self.preflight.returncode:
            raise RuntimeError("CUDA batch8 preflight did not exit successfully")
        self.update("cuda_batch8_passed", gpu_report=str(self.root/"preflight/full_cuda_b8.json"))
        self.open_allocation("warmup")
        write_json(self.root/"warmup.updates.gate.json", dict(train=True))
        self.update("running", stage="warmup", worker_pid=self.worker.pid)
        self.wait_worker("warmup")
        self.update("preparing_joint", stage="prepare")
        shared = self.root/"backend_init.pt"
        self.worker = self.spawn([sys.executable, "-u", "train_bt_self_flow.py", "--config",
                                 str(self.root/"joint.json"), "--device", "cpu", "--prepare-backend",
                                 str(self.root/"common/complete.pt"), "--output", str(shared)], "prepare")
        self.wait_for(lambda: self.worker.poll() is not None)
        if self.worker.returncode:
            raise RuntimeError("EMA handoff preparation failed; inspect prepare.log")
        handoff = check_handoff(self.root/"common/complete.pt", shared, self.manifest["configs"]["joint"])
        write_json(self.root/"handoff_check.json", handoff)
        self.start_worker("joint", shared)
        self.open_allocation("joint")
        # Model, loss, optimizers and data workers are resident before releasing protection.
        self.stop_guard()
        write_json(self.root/"joint.updates.gate.json", dict(train=True))
        self.update("running", stage="joint", worker_pid=self.worker.pid, guard_pid=None)
        self.wait_worker("joint")
        self.update("completed", stage="joint")

    def stop(self, *_):
        self.stopping = True
        for process in (self.worker, self.preflight):
            if process is not None and process.poll() is None:
                process.terminate()

    def recover_failure(self):
        protection_error = None
        if not self.stopping and self.holder_release_requested:
            try:
                if self.preflight is not None and self.preflight.poll() is None:
                    self.preflight.terminate()
                    self.preflight.wait(timeout=30)
                self.restore_protection()
            except BaseException as recovery:
                protection_error = f"{type(recovery).__name__}: {recovery}"
        for process in (self.worker, self.preflight):
            if process is not None and process.poll() is None and protection_error is None:
                process.terminate()
        return protection_error

    def protection_alive(self, protection_error):
        if self.guard is not None and self.guard.poll() is None:
            return True
        return bool(protection_error and any(process is not None and process.poll() is None
                                            for process in (self.worker, self.preflight)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--reserve-gib", type=float, default=20.)
    parser.add_argument("--warmup-memory-gib", type=float, default=9.5)
    parser.add_argument("--replace-watcher-pid", type=int, required=True)
    parser.add_argument("--replace-holder-pid", type=int, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.gpu != 5 or args.reserve_gib <= 0 or args.warmup_memory_gib <= 0:
        parser.error("this launch is authorized for GPU5 with positive memory budgets")
    if args.reserve_gib+args.warmup_memory_gib > 30:
        parser.error("leave GPU context/driver memory outside the two allocation budgets")
    lock = (args.root/"supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.run:
        fcntl.flock(lock, fcntl.LOCK_UN)
        with (args.root/"supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                      *sys.argv[1:], "--run"], cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid, root=str(args.root))), flush=True)
        return
    report = json.loads(args.preflight_report.read_text())
    if report.get("status") != "passed" or report.get("batch_size") != 8 or "C" not in report.get("branches", {}):
        raise ValueError("complete pretrained batch8 preflight must pass before releasing the holder")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("commit reviewed source before launch")
    gpu_uuid = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
    inspect_holder(args.replace_watcher_pid, args.replace_holder_pid, gpu_uuid)
    values = configs(args.root)
    assets = asset_identity(values["joint"])
    if assets != report["assets"]:
        raise ValueError("preflight pretrained assets differ from the formal run")
    manifest = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    repository=str(REPO), gpu=args.gpu, gpu_uuid=gpu_uuid, configs=values,
                    assets=assets, reserve_gib=args.reserve_gib, warmup_memory_gib=args.warmup_memory_gib,
                    preflight_report=str(args.preflight_report.resolve()), historical_weights_used=False)
    path = args.root/"manifest.json"
    if path.exists():
        raise ValueError("pipeline was already launched; inspect its status before restarting")
    write_json(path, manifest)
    for name, cfg in values.items():
        write_json(args.root/f"{name}.json", cfg)
    run = Run(args, manifest)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, run.stop)
    try:
        run.execute()
    except BaseException as error:
        protection_error = run.recover_failure()
        run.update("paused" if isinstance(error, InterruptedError) else "failed",
                   error=f"{type(error).__name__}: {error}", protection_error=protection_error)
        # Keep protection alive on a failed run so its reserved card stays occupied.
        while run.holder_release_requested and not run.stopping and run.protection_alive(protection_error):
            time.sleep(.5)
        raise
    finally:
        if run.guard is not None and run.guard.poll() is None:
            run.guard.terminate()
            run.guard.wait(timeout=10)


if __name__ == "__main__":
    main()
