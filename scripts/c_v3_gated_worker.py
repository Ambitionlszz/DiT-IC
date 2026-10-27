"""Preload a C v3 model before handing over its reserved GPU memory."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

import train_bt_self_flow as entry
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.checkpoint import clone_state, save_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allocation-ready", type=Path, required=True)
    parser.add_argument("--allocation-gate", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--initial-state", type=Path)
    args, remaining = parser.parse_known_args()
    config = load_config(args.config)
    model = make_model(config, torch.device("cpu"))
    if args.initial_state:
        if args.initial_state.exists():
            raise ValueError("initial state already exists")
        save_atomic(args.initial_state, dict(raw=clone_state(model.export_state(), cpu=True),
                                            config=config, seed=config["train"]["seed"], trained_updates=0))
    torch.cuda.set_device(0)
    if args.memory_limit_gib is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*1024**3/total, 0)
    # Keep a live allocation even while the previous holder is being replaced.
    probe = torch.empty(64*1024**2, dtype=torch.uint8, device="cuda")
    probe.fill_(1)
    torch.cuda.synchronize()
    args.allocation_ready.write_text(json.dumps(dict(pid=os.getpid(), model_preloaded=True,
                                                    cuda_initialized=True))+"\n")
    while not args.allocation_gate.exists():
        if not Path(f"/proc/{args.parent_pid}").exists():
            return
        time.sleep(.01)
    del probe
    torch.cuda.empty_cache()

    def preloaded(configuration, device):
        if configuration != config:
            raise ValueError("preloaded worker configuration changed")
        return model.to(device)

    entry.make_model = preloaded
    os.environ["C_V3_SUPERVISOR_PID"] = str(args.parent_pid)
    sys.argv = [str(REPO/"train_bt_self_flow.py"), "--config", args.config, *remaining]
    entry.main()


if __name__ == "__main__":
    main()
