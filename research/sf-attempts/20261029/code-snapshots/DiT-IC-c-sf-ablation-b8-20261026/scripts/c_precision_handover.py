"""Replace warmup reservations while its two trainers keep CUDA allocated."""

import json
import os
from pathlib import Path
import signal

from scripts.launch_bt_pair import write_json


def process_state(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    return dict(pid=int(pid), token=fields[19], state=fields[0],
                command=[arg.decode() for arg in command if arg])


def alive(identity):
    current = process_state(identity["pid"])
    return bool(current and current["token"] == identity["token"] and current["state"] != "Z")


def send(identity, signum):
    if alive(identity):
        try:
            os.kill(identity["pid"], signum)
        except ProcessLookupError:
            pass


def inspect_previous(root, gpus, reserve_gib):
    state = json.loads((root/"pipeline.json").read_text())
    if state["status"] != "latent_warmup_ddp":
        raise ValueError("replacement requires an active latent warmup")
    supervisor = process_state(state["supervisor_pid"])
    if not supervisor or supervisor["state"] == "Z" or str(root) not in supervisor["command"] or not any(
            Path(arg).name == "launch_c_precision.py" for arg in supervisor["command"]):
        raise ValueError("previous supervisor identity differs")
    job = state["jobs"]["common"]
    if job["gpus"] != gpus:
        raise ValueError("previous warmup uses different GPUs")
    launcher = process_state(job["pid"])
    if not launcher or str(root/"configs/common.yaml") not in launcher["command"]:
        raise ValueError("previous trainer launcher identity differs")
    ranks = []
    for pid in Path(f"/proc/{launcher['pid']}/task/{launcher['pid']}/children").read_text().split():
        rank = process_state(int(pid))
        if rank and str(root/"configs/common.yaml") in rank["command"]:
            ranks.append(rank)
    if len(ranks) != 2:
        raise ValueError("previous warmup must have two live trainers")
    holders = {}
    for gpu in gpus:
        record = state["reservations"][gpu]
        process = process_state(record["pid"])
        if not process or record["parent_pid"] != supervisor["pid"] or not record["ready"] or not any(
                Path(arg).name == "reserve_gpu_memory.py" for arg in process["command"]):
            raise ValueError("previous memory reservation identity differs")
        if record["allocated_bytes"] != int(reserve_gib*1024**3):
            raise ValueError("previous memory reservation size differs")
        holders[gpu] = process
    return dict(supervisor=supervisor, launcher=launcher, ranks=ranks, holders=holders)


def take_over(run, previous):
    snapshot = inspect_previous(previous, run.gpus, run.reserve_gib)
    supervisor = snapshot["supervisor"]
    swapped = False
    record = dict(previous_root=str(previous), previous=snapshot, status="preparing")
    path = run.root/"handover.json"
    write_json(path, record)
    # The old supervisor checks holder liveness. Park it only during the swap;
    # its trainers continue keeping their CUDA memory until both replacements fit.
    send(supervisor, signal.SIGSTOP)
    try:
        run.spawn_reservations(gated=True)
        run.wait_reservations(lambda: all((run.root/f"reserve_gpu{i}.standby.json").exists()
                                         for i in range(2)))
        for index, gpu in enumerate(run.gpus):
            swapped = True
            send(snapshot["holders"][gpu], signal.SIGTERM)
            run.wait_reservations(lambda: not alive(snapshot["holders"][gpu]))
            write_json(run.root/f"reserve_gpu{index}.allocate.json", dict(allocate=True))
            run.wait_reservations(lambda: run.reservations[gpu][1].exists())
        record["status"] = "waiting_previous_checkpoint"
        write_json(path, record)
        run.update("waiting_previous_checkpoint")
    finally:
        if swapped:
            send(supervisor, signal.SIGTERM)
        send(supervisor, signal.SIGCONT)
    identities = [supervisor, snapshot["launcher"], *snapshot["ranks"]]
    run.wait_reservations(lambda: not any(alive(identity) for identity in identities), timeout=1200)
    record["status"] = "completed"
    write_json(path, record)

