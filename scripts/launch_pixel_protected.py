"""Preload, verify B8, and run C/PiT without an unprotected process handoff."""

import argparse
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GIB = 1024 ** 3
MIB = 1024 ** 2


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity(pid, proc_root=Path("/proc")):
    directory = proc_root / str(pid)
    stat = (directory / "stat").read_text().rpartition(")")[2].split()
    return dict(pid=pid, state=stat[0], parent=int(stat[1]), start=stat[19],
                command=(directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0"))


def same_process(current, expected):
    return all(current[key] == expected[key] for key in ("pid", "parent", "start", "command"))


def verify_holder(pid, gpu_uuid, proc_root=Path("/proc")):
    identity = process_identity(pid, proc_root)
    if identity["state"] == "Z" or not any(Path(arg).name == "train_depart.py"
                                            for arg in identity["command"]):
        raise ValueError("the authorized PID is not a live memory holder")
    environment = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    if f"CUDA_VISIBLE_DEVICES={gpu_uuid}".encode() not in environment:
        raise ValueError("holder is on a different physical GPU")
    controller = process_identity(identity["parent"], proc_root)
    if not any("watch_two_idle_gpus" in arg for arg in controller["command"]):
        raise ValueError("unrecognized reservation controller")
    return identity


def verify_failed_resident(pid, gpu_uuid, previous_root, proc_root=Path("/proc")):
    record = json.loads((previous_root / "status.json").read_text())
    identity = process_identity(pid, proc_root)
    command = identity["command"]
    environment = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    if (record.get("pid") != pid or record.get("gpu_uuid") != gpu_uuid
            or record.get("phase") != "failed_reserved" or identity["state"] == "Z"
            or not any(Path(arg).name == "launch_pixel_protected.py" for arg in command)
            or "--root" not in command or Path(command[command.index("--root") + 1]) != previous_root
            or f"CUDA_VISIBLE_DEVICES={gpu_uuid}".encode() not in environment):
        raise ValueError("refusing to replace a resident outside the failed protected run")
    guard_pid = record.get("guard_pid")
    if guard_pid and (proc_root / str(guard_pid)).exists():
        guard = process_identity(guard_pid, proc_root)
        if guard["state"] != "Z":
            raise ValueError("previous fallback guard must be safely handed off first")
    return identity


def inspect_reports(paths):
    values = [json.loads(path.read_text()) for path in paths]
    if not values or any(value.get("status") != "passed" or value.get("batch_size") != 8
                         or len(value.get("steps", [])) < 2
                         or value.get("scope") != "pretrained_sana_dc_ae_codec_perceptual_rd" for value in values):
        raise ValueError("complete batch8 checks must pass before releasing the holder")
    return values


def reservation_guard(args):
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    blocks = [torch.empty(16 * MIB, dtype=torch.uint8, device="cuda:0")]
    torch.cuda.synchronize()
    write_json(args.root / "guard.ready.json", dict(pid=os.getpid(), cuda_initialized=True))
    active = False
    target = 28 * GIB
    last = 0.
    while True:
        if (args.root / "guard.reserve").exists() or not Path(f"/proc/{args.parent_pid}").exists():
            active = True
        if active:
            free, _ = torch.cuda.mem_get_info(0)
            needed = min(target - sum(block.numel() for block in blocks), free - 384 * MIB)
            if needed >= 16 * MIB:
                try:
                    blocks.append(torch.empty(min(needed, 256 * MIB), dtype=torch.uint8, device="cuda:0"))
                except torch.cuda.OutOfMemoryError:
                    pass
            if time.monotonic() - last > .25:
                write_json(args.root / "guard.status.json", dict(pid=os.getpid(), active=True,
                           allocated_bytes=torch.cuda.memory_allocated(), updated_at=time.time()))
                last = time.monotonic()
        time.sleep(.002 if active else .005)


def source_identity():
    files = sorted(path for base in (REPO / "models/bt_self_flow", REPO / "configs/bt_self_flow/pixel_adapter",
                                    REPO / "scripts") for path in base.glob("*.py" if base.name != "pixel_adapter" else "*.yaml"))
    files += [REPO / "train_bt_self_flow.py"]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def run(args):
    import torch
    import train_bt_self_flow as entry
    from models.bt_self_flow.activation_storage import preserve_stride_cpu_offload
    from models.bt_self_flow.checkpoint import clone_state, read_checkpoint, save_atomic, sha256_file
    from models.bt_self_flow.experiment import asset_identity, check_stage_model, load_config, make_model
    from scripts import check_pixel_adapter as check

    torch.set_num_threads(4)
    def authorized_holder():
        if args.previous_run_root:
            return verify_failed_resident(args.replace_holder_pid, args.gpu_uuid, args.previous_run_root)
        return verify_holder(args.replace_holder_pid, args.gpu_uuid)

    holder = authorized_holder()
    cpu_reports = inspect_reports([args.cpu_report])
    configs = {stage: load_config(REPO / "configs/bt_self_flow/pixel_adapter" / name)
               for stage, name in (("warmup", "common.yaml"), ("joint", "joint.yaml"))}
    for stage, cfg in configs.items():
        settings = cfg["train"]
        if (settings["batch_size"], settings["accumulation_steps"], settings["effective_batch"]) != (8, 1, 8):
            raise ValueError("single GPU batch8 accumulation1 is required")
        settings["output"] = str(args.root / ("common" if stage == "warmup" else "content_quantization"))
        write_json(args.root / f"{stage}.json", cfg)
    if configs["warmup"]["train"]["steps"] != 50000 or configs["joint"]["train"]["steps"] != 200000:
        raise ValueError("the matched C 50K + 200K schedule changed")
    check_stage_model(configs["joint"], configs["warmup"])
    manifest = dict(pid=os.getpid(), gpu_uuid=args.gpu_uuid, gpu_index=5, batch_size=8, accumulation_steps=1,
                    phase="preloading", self_flow_enabled=False, pixel_depth=2, config=configs,
                    previous_holder=holder, cpu_reports=cpu_reports, source_hashes=source_identity(),
                    autocast_weight_cache=False,
                    activation_storage="cpu_saved_tensors" if args.offload_activations else "native",
                    skip_warmup=bool(args.init_checkpoint),
                    started_at=time.time())

    def status(phase, **values):
        manifest.update(phase=phase, updated_at=time.time(), **values)
        write_json(args.root / "status.json", manifest)
        print(json.dumps(dict(phase=phase, pid=os.getpid(), **values)), flush=True)

    status("preloading")
    # A shared C initialization already contains the calibrated 50K frontend.
    # Keep its exact state, including original LoRA and text weights.
    warmup = None if args.init_checkpoint else make_model(configs["warmup"], torch.device("cpu"))
    joint = make_model(configs["joint"], torch.device("cpu"))
    if args.init_checkpoint:
        source = read_checkpoint(args.init_checkpoint)
        calibration = source.get("calibration") or {}
        if (source["phase"] != "backend_init" or source["backend"] != "shared"
                or source.get("codec_weights") != "ema" or not source.get("source_sha256")
                or not calibration.get("rate_span_pass") or not calibration.get("quality_span_pass")):
            raise ValueError("direct joint training requires calibrated shared C EMA initialization")
        check_stage_model(configs["joint"], source["config"])
        if source["assets"] != asset_identity(configs["joint"]):
            raise ValueError("shared initialization pretrained assets differ")
        entry.initialize_from(joint, source, "joint")
        manifest["initialization"] = dict(path=str(args.init_checkpoint), sha256=sha256_file(args.init_checkpoint),
                                         warmup_sha256=source["source_sha256"], calibration=calibration)
        del source
    initial_joint = clone_state(joint.export_state(), cpu=True)
    if warmup is not None:
        save_atomic(args.root / "initial_codec.pt", dict(raw=clone_state(warmup.export_state(), cpu=True),
                    config=configs["warmup"], seed=903, trained_updates=0))
    else:
        save_atomic(args.root / "initial_joint.pt", dict(raw=initial_joint, config=configs["joint"],
                    seed=903, trained_updates=0, initialization=manifest["initialization"]))
    torch.cuda.set_device(0)
    probe = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    with (args.root / "guard.log").open("a") as output:
        guard = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--guard",
                                  "--root", str(args.root), "--parent-pid", str(os.getpid())],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 env=os.environ.copy(), start_new_session=True)
    status("cuda_standby", guard_pid=guard.pid)
    while not (args.root / "guard.ready.json").exists():
        if guard.poll() is not None:
            raise RuntimeError("fallback reservation failed to initialize")
        time.sleep(.01)

    def reserve_local(wait_seconds=0.):
        blocks = []
        deadline = time.monotonic() + wait_seconds
        while torch.cuda.memory_allocated() < 28 * GIB:
            free, _ = torch.cuda.mem_get_info()
            capacity = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() - 768 * MIB
            chunk = min(256 * MIB, 28 * GIB - torch.cuda.memory_allocated(), capacity)
            if chunk < 16 * MIB:
                if time.monotonic() < deadline:
                    time.sleep(.002)
                    continue
                break
            try:
                blocks.append(torch.empty(int(chunk), dtype=torch.uint8, device="cuda:0"))
            except torch.cuda.OutOfMemoryError:
                break
        return blocks

    transferred = False
    blocks = []
    try:
        if not same_process(authorized_holder(), holder):
            raise ValueError("holder process identity changed during preload")
        # The two-slot controller stays alive; only its authorized GPU5 child exits.
        transferred = True
        os.kill(holder["pid"], signal.SIGTERM)
        while True:
            try:
                if process_identity(holder["pid"])["state"] == "Z":
                    break
            except FileNotFoundError:
                break
            time.sleep(.002)
        released = time.monotonic()
        blocks = reserve_local(wait_seconds=10.)
        if torch.cuda.memory_allocated() < 26 * GIB:
            raise RuntimeError("could not establish replacement memory protection")
        status("protected", holder_to_reserve_seconds=time.monotonic() - released,
               allocated_bytes=torch.cuda.memory_allocated())
        # No empty_cache: the same allocator reuses the reserved blocks for training.
        blocks.clear()
        del probe
        torch.cuda.reset_peak_memory_stats()
        check.make_model = lambda cfg, device: joint.to(device)
        verification = argparse.Namespace(config=str(args.root / "joint.json"), batch_size=8, updates=2,
                                          retain_calibrated_scale=bool(args.init_checkpoint))
        storage = preserve_stride_cpu_offload() if args.offload_activations else nullcontext()
        status("checking_cuda_batch8")
        debug_context = (torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
                         if args.debug_checkpoint else nullcontext())
        with storage, debug_context:
            report = check.check_full(verification, torch.device("cuda:0"))
        report.update(status="passed", batch_size=8, device="cuda:0", research_training=False,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), activation_storage=manifest["activation_storage"])
        write_json(args.root / "full_cuda_b8.json", report)
        # Engineering updates are discarded and never enter the research trajectory.
        joint.to("cpu")
        joint.load_bt_state(initial_joint)
        del initial_joint
        gc.collect()
        status("cuda_batch8_passed", peak_allocated_bytes=report["peak_allocated_bytes"])
        original_make_model = entry.make_model

        def train_stage(stage, model, initialization=None):
            configuration = configs[stage]

            def preloaded(cfg, device):
                if cfg != configuration:
                    raise ValueError("preloaded model configuration differs")
                return model.to(device)

            entry.make_model = preloaded
            arguments = argparse.Namespace(resume=None, init_checkpoint=initialization, max_updates=None,
                                           checkpoint_steps=(10,))
            status("training", stage=stage)
            # This only moves saved autograd tensors; it leaves forwards, losses,
            # batch statistics, optimizer updates and the RNG contract unchanged.
            context = (preserve_stride_cpu_offload()
                       if args.offload_activations and stage == "joint" else nullcontext())
            with context:
                entry.train(configuration, arguments, 0, 1, torch.device("cuda:0"))
            entry.make_model = original_make_model

        if args.init_checkpoint:
            shared = args.init_checkpoint
        else:
            train_stage("warmup", warmup)
            complete = args.root / "common/complete.pt"
            state = read_checkpoint(complete)
            if state["step"] != 50000 or state["phase"] != "codec":
                raise RuntimeError("warmup stopped early; keep GPU reserved for recovery")
            del state
            warmup.to("cpu")
            status("preparing_joint")
            shared = args.root / "backend_init.pt"
            entry.make_model = lambda cfg, device: joint.to(device)
            entry.prepare_backend(configs["joint"], complete, shared, torch.device("cpu"))
            entry.make_model = original_make_model
        train_stage("joint", joint, str(shared))
        endpoint = read_checkpoint(args.root / "content_quantization/complete.pt")
        if endpoint["phase"] != "joint" or endpoint["step"] != 200000:
            raise RuntimeError("joint training stopped early; keep GPU reserved for recovery")
        del endpoint
        status("completed", stage="joint")
        (args.root / "guard.reserve").touch()
        blocks = reserve_local()
        while True:
            time.sleep(.25)
    except BaseException as error:
        # Wake the already initialized guard before unwinding CUDA graphs.
        if transferred:
            (args.root / "guard.reserve").touch()
        text = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        traceback.clear_frames(error.__traceback__)
        if not transferred:
            guard.terminate()
            raise
        gc.collect()
        blocks = reserve_local()
        status("failed_reserved", error=text, guard_pid=guard.pid,
               allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        while True:
            time.sleep(.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--replace-holder-pid", type=int)
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="shared calibrated C initialization; skip repeated warmup")
    parser.add_argument("--previous-run-root", type=Path, help="explicit failed resident recovery only")
    parser.add_argument("--offload-activations", action="store_true")
    parser.add_argument("--debug-checkpoint", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.guard:
        reservation_guard(args)
        return
    if not args.gpu_uuid or not args.replace_holder_pid or not args.cpu_report:
        parser.error("exact GPU UUID, authorized holder PID and passed CPU report are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES must select the exact physical GPU UUID")
    if any(int(os.environ.get(key, "1" if key == "WORLD_SIZE" else "0")) != expected
           for key, expected in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))):
        parser.error("clear distributed environment for single-GPU batch8")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "status.json").exists():
        parser.error("run root exists; choose a new root")
    run(args)


if __name__ == "__main__":
    main()
