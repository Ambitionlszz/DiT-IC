"""Resume the matched C/PiT experiment with a CUDA-ready fallback holder."""

import argparse
from copy import deepcopy
import fcntl
import gc
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

from scripts.launch_pixel_protected import GIB, MIB, process_identity, same_process, source_identity, write_json


def verify_reservation(status_path, gpu_uuid, proc_root=Path("/proc")):
    record = json.loads(status_path.read_text())
    identity = process_identity(record["pid"], proc_root)
    command = identity["command"]
    environment = (proc_root / str(identity["pid"]) / "environ").read_bytes().split(b"\0")
    if (not record.get("ready") or record.get("gpu") != gpu_uuid
            or identity["state"] == "Z" or record.get("parent_pid") != 1
            or str(REPO / "scripts/reserve_gpu_memory.py") not in command
            or "--status" not in command or command[command.index("--status") + 1] != str(status_path)
            or f"CUDA_VISIBLE_DEVICES={gpu_uuid}".encode() not in environment):
        raise ValueError("refusing to replace an unrelated GPU reservation")
    return identity


def verify_failed_resume(root, gpu_uuid, proc_root=Path("/proc")):
    record = json.loads((root / "status.json").read_text())
    identity = process_identity(record["pid"], proc_root)
    command = identity["command"]
    environment = (proc_root / str(identity["pid"]) / "environ").read_bytes().split(b"\0")
    if (record.get("phase") != "failed_reserved" or record.get("gpu_uuid") != gpu_uuid
            or identity["state"] == "Z" or str(REPO / "scripts/resume_pixel_protected.py") not in command
            or "--root" not in command or command[command.index("--root") + 1] != str(root)
            or f"CUDA_VISIBLE_DEVICES={gpu_uuid}".encode() not in environment):
        raise ValueError("refusing to replace an unrelated or running training process")
    guard_pid = record.get("guard_pid")
    if guard_pid and (proc_root / str(guard_pid)).exists() and process_identity(guard_pid, proc_root)["state"] != "Z":
        raise ValueError("previous fallback guard must be handed off while its worker retains the GPU")
    return identity


def resume_configuration(checkpoint, output):
    config = deepcopy(checkpoint["config"])
    settings = config["train"]
    decoder = config.get("decoder", {})
    if (checkpoint["phase"] != "joint" or checkpoint["backend"] != "C"
            or decoder.get("method") != "sana_conditioned_pixel_adapter_v1"
            or decoder.get("pixel_adapter", {}).get("depth") != 2
            or config["self_flow"].get("enabled")
            or "pixel_adapter" not in checkpoint["raw"] or "pixel_adapter" not in checkpoint["ema"]
            or (settings["batch_size"], settings["accumulation_steps"], settings["effective_batch"]) not in ((8, 1, 8), (4, 2, 8))
            or checkpoint["contract"]["world_size"] != 1
            or not 0 < checkpoint["step"] < settings["steps"]):
        raise ValueError("expected a resumable single-GPU effective-B8 C/two-PiT joint checkpoint")
    settings["output"] = str(output)
    return config


def repartition_microbatches(checkpoint, batch_size):
    original = checkpoint["config"]["train"]
    if batch_size not in (4, 8) or original["batch_size"] * original["accumulation_steps"] != 8:
        raise ValueError("only authorized effective-batch-eight layouts are supported")
    consumed_images = checkpoint["micro_seen"] * original["batch_size"]
    if consumed_images % batch_size:
        raise ValueError("new microbatch would skip or repeat consumed training images")
    migrated = dict(checkpoint)
    for part in ("config", "contract"):
        migrated[part] = deepcopy(checkpoint[part])
        migrated[part]["train"].update(batch_size=batch_size, accumulation_steps=8 // batch_size)
    migrated["micro_seen"] = consumed_images // batch_size
    return migrated


def stage_resume_checkpoints(checkpoint, source, output):
    from models.bt_self_flow.checkpoint import publish_checkpoint_alias

    output.mkdir(parents=True, exist_ok=True)
    paths = {source.name: source}
    best = checkpoint.get("validation_best")
    if best:
        name = best["checkpoint"]
        if Path(name).name != name or not Path(name).stem.isdecimal():
            raise ValueError("invalid restored best-checkpoint filename")
        paths[name] = source.parent / name
    for name, origin in paths.items():
        target = output / name
        if target.exists():
            if not target.samefile(origin):
                raise ValueError("resume output already contains different recovery weights")
        else:
            publish_checkpoint_alias(origin, target)


def parent_is_alive(pid, start, proc_root=Path("/proc")):
    try:
        identity = process_identity(pid, proc_root)
    except FileNotFoundError:
        return False
    return identity["state"] != "Z" and identity["start"] == start


def cpu_reservation_guard(args):
    write_json(args.root / "guard.ready.json", dict(pid=os.getpid(), cuda_initialized=False, mode="cpu"))
    while not (args.root / "guard.reserve").exists() and parent_is_alive(args.parent_pid, args.parent_start):
        time.sleep(.005)
    command = [sys.executable, "-u", str(REPO / "scripts/launch_pixel_protected.py"), "--guard",
               "--root", str(args.root), "--parent-pid", str(args.parent_pid)]
    os.execv(sys.executable, command)


def run(args):
    import torch
    import train_bt_self_flow as entry
    from models.bt_self_flow.activation_storage import preserve_stride_cpu_offload
    from models.bt_self_flow.checkpoint import read_checkpoint
    from models.bt_self_flow.data import ImageDataset
    from models.bt_self_flow.experiment import asset_identity, make_model, optimization_contract, training_contract

    torch.set_num_threads(4)
    reservation = args.root / "reservation.json"
    def authorized_holder():
        if args.previous_run_root:
            return verify_failed_resume(args.previous_run_root, args.gpu_uuid)
        return verify_reservation(reservation, args.gpu_uuid)

    holder = authorized_holder()
    checkpoint = read_checkpoint(args.resume)
    config = resume_configuration(checkpoint, args.root / "content_quantization")
    step = checkpoint["step"]
    manifest = dict(pid=os.getpid(), gpu_uuid=args.gpu_uuid, gpu_index=args.gpu_index,
                    source_checkpoint=str(args.resume), source_step=step,
                    source_checkpoint_bytes=args.resume.stat().st_size,
                    batch_size=args.batch_size, accumulation_steps=8 // args.batch_size, effective_batch=8, pixel_depth=2,
                    self_flow_enabled=False, activation_storage="cpu_saved_tensors", guard_mode="cpu",
                    allocator=os.environ.get("PYTORCH_ALLOC_CONF"),
                    source_hashes=source_identity(), config=config, previous_holder=holder,
                    started_at=time.time())

    def status(phase, **values):
        manifest.update(phase=phase, updated_at=time.time(), **values)
        write_json(args.root / "status.json", manifest)
        print(json.dumps(dict(phase=phase, pid=os.getpid(), **values)), flush=True)

    status("preloading")
    write_json(args.root / "joint.json", config)
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    if training_contract(config, dataset, 1) != optimization_contract(checkpoint["contract"]):
        raise ValueError("checkpoint optimization contract or training dataset differs")
    if asset_identity(config) != checkpoint["assets"]:
        raise ValueError("checkpoint pretrained assets differ")
    original_layout = dict(batch_size=config["train"]["batch_size"],
                           accumulation_steps=config["train"]["accumulation_steps"], micro_seen=checkpoint["micro_seen"])
    checkpoint = repartition_microbatches(checkpoint, args.batch_size)
    config = resume_configuration(checkpoint, args.root / "content_quantization")
    manifest.update(config=config, microbatch_migration=dict(source=original_layout,
                    target=dict(batch_size=args.batch_size, accumulation_steps=8 // args.batch_size,
                                micro_seen=checkpoint["micro_seen"]), optimizer_state_retained=True))
    write_json(args.root / "joint.json", config)
    status("preloading", optimization_contract_verified=True)
    stage_resume_checkpoints(checkpoint, args.resume, Path(config["train"]["output"]))
    model = make_model(config, torch.device("cpu"))
    model.load_bt_state(checkpoint["raw"])
    del dataset
    gc.collect()
    torch.cuda.set_device(0)
    probe = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda:0")
    torch.cuda.synchronize()
    with (args.root / "guard.log").open("a") as log:
        guard = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                  "--cpu-guard", "--root", str(args.root), "--parent-pid", str(os.getpid()),
                                  "--parent-start", process_identity(os.getpid())["start"]],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=os.environ.copy(), start_new_session=True)
    status("cuda_standby", guard_pid=guard.pid)
    deadline = time.monotonic() + 60
    while not (args.root / "guard.ready.json").exists():
        if guard.poll() is not None or time.monotonic() > deadline:
            guard.terminate()
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
            raise ValueError("reservation identity changed during preload")
        transferred = True
        os.kill(holder["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while Path(f"/proc/{holder['pid']}").exists() and process_identity(holder["pid"])["state"] != "Z":
            if time.monotonic() > deadline:
                raise RuntimeError("temporary reservation did not exit")
            time.sleep(.002)
        released = time.monotonic()
        blocks = reserve_local(wait_seconds=2.)
        if torch.cuda.memory_allocated() < 26 * GIB:
            raise RuntimeError("insufficient available memory for the validated B8 training footprint")
        status("protected", holder_to_reserve_seconds=time.monotonic() - released,
               allocated_bytes=torch.cuda.memory_allocated())
        blocks.clear()
        del probe
        torch.cuda.reset_peak_memory_stats()

        def preloaded(configuration, device):
            if configuration != config:
                raise ValueError("preloaded resume model configuration differs")
            return model.to(device)

        entry.make_model = preloaded
        original_read_checkpoint = entry.read_checkpoint

        def exact_resume(path):
            if Path(path).resolve() == args.resume.resolve():
                return checkpoint
            return original_read_checkpoint(path)

        entry.read_checkpoint = exact_resume
        arguments = argparse.Namespace(resume=str(args.resume), init_checkpoint=None, max_updates=None,
                                       checkpoint_steps=(step + 10,))
        status("training", stage="joint", resume_step=step)
        with preserve_stride_cpu_offload():
            entry.train(config, arguments, 0, 1, torch.device("cuda:0"))
        endpoint = read_checkpoint(Path(config["train"]["output"]) / "complete.pt")
        if endpoint["step"] != config["train"]["steps"] or endpoint["phase"] != "joint":
            raise RuntimeError("joint training stopped before its endpoint")
        status("completed", stage="joint")
        (args.root / "guard.reserve").touch()
        blocks = reserve_local()
        while True:
            time.sleep(.25)
    except BaseException as error:
        if transferred:
            (args.root / "guard.reserve").touch()
        traceback.print_exc()
        traceback.clear_frames(error.__traceback__)
        if not transferred:
            guard.terminate()
            raise
        gc.collect()
        blocks = reserve_local()
        status("failed_reserved", error=f"{type(error).__name__}: {error}", guard_pid=guard.pid,
               allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        while True:
            time.sleep(.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--previous-run-root", type=Path)
    parser.add_argument("--cpu-guard", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--parent-start")
    args = parser.parse_args()
    if args.cpu_guard:
        if not args.parent_pid or not args.parent_start:
            parser.error("CPU guard requires the exact parent PID and start identity")
        cpu_reservation_guard(args)
        return
    if not args.resume or not args.gpu_uuid or args.gpu_index is None:
        parser.error("checkpoint, exact GPU UUID and physical GPU index are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES must select the exact GPU UUID")
    if any(int(os.environ.get(key, default)) != expected for key, default, expected in
           (("WORLD_SIZE", "1", 1), ("RANK", "0", 0), ("LOCAL_RANK", "0", 0))):
        parser.error("single-GPU continuation requires world size 1 and rank 0")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / "status.json").exists():
        parser.error("run root already exists")
    run(args)


if __name__ == "__main__":
    main()
