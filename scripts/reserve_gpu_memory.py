"""Reserve memory beside a warmup worker without competing for compute."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=18.)
    parser.add_argument("--minimum-free-gib", type=float, default=3.)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--allocation-gate", type=Path)
    parser.add_argument("--standby-status", type=Path)
    parser.add_argument("--control", type=Path, help="optional atomic JSON file containing the desired gib")
    args = parser.parse_args()
    if args.gib <= 0 or args.minimum_free_gib < 0:
        parser.error("invalid memory reservation")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *a: stopping.__setitem__(0, True))
    if args.allocation_gate:
        if args.standby_status is None:
            parser.error("allocation gate requires a standby status file")
        probe = torch.empty(1, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        del probe
        torch.cuda.empty_cache()
        args.standby_status.write_text(json.dumps(dict(pid=os.getpid(), ready=True))+"\n")
        while not args.allocation_gate.exists():
            if stopping[0] or not Path(f"/proc/{args.parent_pid}").exists():
                args.standby_status.unlink(missing_ok=True)
                return
            time.sleep(.01)
        args.standby_status.unlink(missing_ok=True)
    free, total = torch.cuda.mem_get_info(device)
    size = int(args.gib * 1024**3)
    if free < size + args.minimum_free_gib * 1024**3:
        raise RuntimeError("insufficient free memory for reservation and training margin")
    blocks = []
    remaining = size
    while remaining:
        chunk = min(remaining, 256 * 1024**2)
        blocks.append(torch.empty(chunk, dtype=torch.uint8, device=device).fill_(1))
        remaining -= chunk
    torch.cuda.synchronize(device)
    record = dict(pid=os.getpid(), parent_pid=args.parent_pid, ready=True,
                  allocated_bytes=torch.cuda.memory_allocated(device), total_bytes=total,
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    args.status.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.status.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(args.status)
    print(json.dumps(record), flush=True)
    while not stopping[0]:
        try:
            os.kill(args.parent_pid, 0)
        except ProcessLookupError:
            break
        if args.control and args.control.exists():
            requested = float(json.loads(args.control.read_text())["gib"])
            if not math.isfinite(requested) or requested < 0:
                raise ValueError("invalid controlled reservation size")
            target = int(requested*1024**3)
            if target != size:
                if target < size:
                    blocks.clear()
                    torch.cuda.empty_cache()
                    size = 0
                free, _ = torch.cuda.mem_get_info(device)
                if free >= target-size+args.minimum_free_gib*1024**3:
                    remaining = target-size
                    while remaining:
                        chunk = min(remaining, 256*1024**2)
                        blocks.append(torch.empty(chunk, dtype=torch.uint8, device=device).fill_(1))
                        remaining -= chunk
                    size = target
                    torch.cuda.synchronize(device)
                record.update(allocated_bytes=torch.cuda.memory_allocated(device), requested_bytes=target,
                              ready=size == target)
                temporary.write_text(json.dumps(record, indent=2)+"\n")
                temporary.replace(args.status)
        time.sleep(.01 if args.control else .5)
    args.status.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
