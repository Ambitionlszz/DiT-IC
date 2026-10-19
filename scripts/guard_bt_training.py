"""Supervise a training pipeline; reclaim its exact GPUs after any child exit.

The guardian does not import torch. CUDA holders are independent children.
SIGTERM pauses training and retains both GPUs. SIGUSR1 explicitly releases
the reservation and exits. SIGUSR2 releases holders and resumes the pipeline.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def gpu_snapshot():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    return {int(row[0]): dict(uuid=row[1], used_mib=int(row[2]), free_mib=int(row[3]))
            for line in output.splitlines() if (row := [part.strip() for part in line.split(",")])}


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def enable_subreaper():
    # torchrun may give each rank a separate session. Adopt its orphaned
    # descendants so cleanup also works after the launcher itself is killed.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")


def descendants(pid, excluded=()):
    found, pending = {}, [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except FileNotFoundError:
            continue
        for value in children:
            child = int(value)
            if child in excluded or child in found:
                continue
            try:
                token = Path(f"/proc/{child}/stat").read_text().rpartition(")")[2].split()[19]
            except FileNotFoundError:
                continue
            found[child] = token
            pending.append(child)
    return found


def signal_processes(processes, signum):
    for pid, token in processes.items():
        try:
            current = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
            if current == token:
                os.kill(pid, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass


class TrainingGuard:
    def __init__(self, root, devices, command, holder_command, snapshot=gpu_snapshot, defer=False, subreaper=False):
        self.root, self.devices = Path(root), devices
        self.command, self.holder_command, self.snapshot = command, holder_command, snapshot
        self.child, self.holders, self.last_exit = None, {}, None
        self.pending_cleanup = None
        self.subreaper, self.reap_pending = subreaper, set()
        self.start_requested = not defer
        self.pause_requested = self.release_requested = False
        self.status = "waiting_for_handover" if defer else "starting"
        self.root.mkdir(parents=True, exist_ok=True)

    def start(self):
        self.stop_holders()
        with (self.root/"pipeline.log").open("a") as log:
            self.child = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
        self.status = "training"
        self.last_exit = None
        self.start_requested = False
        self.pause_requested = False

    def stop_holders(self):
        for process, _ in self.holders.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in self.holders.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.holders.clear()

    def publish(self):
        holders = {}
        for gpu, (process, path) in self.holders.items():
            ready = process.poll() is None and path is not None and "[Stress] running" in path.read_text(errors="replace")
            holders[str(gpu)] = dict(pid=process.pid, uuid=self.devices[gpu], ready=ready,
                                      returncode=process.poll(), log=str(path) if path is not None else None)
        atomic_json(self.root/"guard.json", dict(
            guard_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), status=self.status,
            devices=self.devices, pipeline_pid=self.child.pid if self.child else None,
            pipeline_returncode=self.last_exit, holders=holders,
            pause_signal="SIGTERM", release_signal="SIGUSR1", resume_signal="SIGUSR2"))

    def tick(self):
        for pid in list(self.reap_pending):
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
                if reaped:
                    self.reap_pending.discard(pid)
            except ChildProcessError:
                self.reap_pending.discard(pid)
        if self.release_requested:
            if self.child and self.child.poll() is None:
                self.pause_requested = True
            elif self.child is None and self.pending_cleanup is None:
                self.stop_holders()
                self.status = "released"
                self.publish()
                return False
        if self.child is not None:
            if self.pause_requested and self.child.poll() is None:
                self.child.send_signal(signal.SIGTERM)
                self.pause_requested = False
                self.status = "pausing"
            code = self.child.poll()
            if code is not None:
                self.last_exit = code
                # The supervisor and all its training descendants share this
                # group, including workers orphaned by a killed supervisor.
                signal_group(self.child.pid, signal.SIGTERM)
                if self.subreaper:
                    processes = descendants(os.getpid(), excluded=[p.pid for p, _ in self.holders.values()])
                    signal_processes(processes, signal.SIGTERM)
                    self.reap_pending.update(processes)
                self.pending_cleanup = (self.child.pid, time.monotonic()+2)
                self.child = None
                self.status = "holding_after_failure" if code else "holding_after_completion"
        if self.pending_cleanup:
            pgid, deadline = self.pending_cleanup
            if time.monotonic() < deadline:
                self.publish()
                return True
            signal_group(pgid, signal.SIGKILL)
            if self.subreaper:
                processes = descendants(os.getpid(), excluded=[p.pid for p, _ in self.holders.values()])
                signal_processes(processes, signal.SIGKILL)
                self.reap_pending.update(processes)
            self.pending_cleanup = None
        if self.child is None and self.start_requested and not self.release_requested:
            try:
                self.start()
            except Exception:
                self.start_requested = False
                self.last_exit = -1
                self.status = "holding_after_start_failure"
                raise
        if self.child is None and self.last_exit is not None and not self.release_requested:
            rows = self.snapshot()
            for gpu, uuid in self.devices.items():
                current = self.holders.get(gpu)
                if current and current[0].poll() is None:
                    continue
                if current:
                    self.holders.pop(gpu)
                row = rows.get(gpu)
                # Reclaim only the selected physical GPUs after our processes
                # have exited. Never evict an unrelated CUDA allocation.
                if not row or row["uuid"] != uuid or row["used_mib"] > 256:
                    continue
                path = self.root/f"holder_gpu{gpu}.log"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                try:
                    log = path.open("w")
                except OSError:
                    # A full experiment disk may be the reason training died.
                    # CUDA reservation must not depend on writable log storage.
                    log, path = None, None
                try:
                    process = subprocess.Popen(self.holder_command(gpu), env=env,
                                               stdout=log if log is not None else subprocess.DEVNULL,
                                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               start_new_session=True)
                finally:
                    if log is not None:
                        log.close()
                self.holders[gpu] = (process, path)
        self.publish()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--holder-script", required=True)
    parser.add_argument("--holder-python", default=sys.executable)
    parser.add_argument("--defer", action="store_true", help="wait for SIGUSR2 before handover")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    indices = [int(value) for value in args.gpus.split(",")]
    if len(indices) != 2 or len(set(indices)) != 2 or not command:
        parser.error("two distinct GPU indices and a pipeline command are required")
    if not Path(args.holder_script).is_file():
        parser.error("holder script does not exist")
    rows = gpu_snapshot()
    devices = {gpu: rows[gpu]["uuid"] for gpu in indices}
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/"guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        enable_subreaper()
        guard = TrainingGuard(root, devices, command, lambda gpu: [
            args.holder_python, "-u", args.holder_script, "--gb", "27", "--util", ".35", "--matrix", "8192"],
            defer=args.defer, subreaper=True)

        def request_pause(_signal, _frame):
            guard.pause_requested = True

        def request_release(_signal, _frame):
            guard.release_requested = True

        def request_resume(_signal, _frame):
            guard.start_requested = True

        signal.signal(signal.SIGTERM, request_pause)
        signal.signal(signal.SIGINT, request_pause)
        signal.signal(signal.SIGUSR1, request_release)
        signal.signal(signal.SIGUSR2, request_resume)
        while True:
            try:
                if not guard.tick():
                    break
            except Exception as error:
                try:
                    print(f"guardian retry: {type(error).__name__}: {error}", flush=True)
                except OSError:
                    pass
            time.sleep(1)


if __name__ == "__main__":
    main()
