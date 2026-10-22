"""Reserve memory beside a warmup worker without competing for compute."""

import argparse
import json
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
    args = parser.parse_args()
    if args.gib <= 0 or args.minimum_free_gib < 0:
        parser.error("invalid memory reservation")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    size = int(args.gib * 1024**3)
    if free < size + args.minimum_free_gib * 1024**3:
        raise RuntimeError("insufficient free memory for reservation and training margin")
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *a: stopping.__setitem__(0, True))
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
        time.sleep(.5)
    args.status.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
