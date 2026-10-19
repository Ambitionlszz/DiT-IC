"""Verify the exact production batch sizes, then start an explicitly armed run.

Engineering checkpoints use a separate directory and shortened schedules.
The first optional fault drill deliberately fails CUDA ranks so the external
guardian can demonstrate that it really reclaims both physical GPUs.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def execute(root, name, command, env):
    started = time.monotonic()
    with (root/f"{name}.log").open("a") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = dict(name=name, command=command, returncode=result.returncode,
               seconds=time.monotonic()-started, log=str(root/f"{name}.log"))
    print(json.dumps(row), flush=True)
    if result.returncode:
        raise RuntimeError(f"verification {name} failed: {root/f'{name}.log'}")
    return row


def verify(args, root, commit):
    from models.bt_self_flow.checkpoint import read_checkpoint
    from scripts.launch_bt_lowrate_c import resolved_configs

    root.mkdir(parents=True, exist_ok=True)
    configs = resolved_configs(root)
    for name, config in configs.items():
        config["data"].update(valid_limit=2, workers=4)
        config["train"].update(steps=14, log_every=1, eval_every=2 if name == "common" else 7,
                               save_every=2, data_scale_samples=2, calibration_samples=4,
                               router_uniform_steps=2, router_ramp_steps=3,
                               packet_samples=2, require_quality_span=False, keep_last_checkpoints=2)
        config["optimizer"]["warmup"] = 2
        (root/f"{name}.yaml").write_text(json.dumps(config, indent=2)+"\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    python = sys.executable
    ddp = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train_bt_self_flow.py"]
    commands = []
    common = root/"common/complete.pt"
    paused = root/"common/paused.pt"
    if not common.exists():
        if not paused.exists():
            commands.append(execute(root, "common_pause", ddp+["--config", str(root/"common.yaml"), "--max-updates", "2"], env))
        commands.append(execute(root, "common_resume", ddp+["--config", str(root/"common.yaml"), "--resume", str(paused)], env))
    source = root/"backend_init.pt"
    if not source.exists():
        first_env = dict(env, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0])
        commands.append(execute(root, "prepare", [python, "train_bt_self_flow.py", "--config", str(root/"c_joint.yaml"),
                                                  "--prepare-backend", str(common), "--output", str(source)], first_env))
    complete, paused = root/"C/joint/complete.pt", root/"C/joint/paused.pt"
    if not complete.exists():
        if not paused.exists():
            commands.append(execute(root, "c_joint_pause", ddp+["--config", str(root/"c_joint.yaml"),
                                                              "--init-checkpoint", str(source), "--max-updates", "2"], env))
        commands.append(execute(root, "c_joint_resume", ddp+["--config", str(root/"c_joint.yaml"), "--resume", str(paused)], env))
    throughput = {}
    for name, directory in (("common", root/"common"), ("c_joint", root/"C/joint")):
        state = read_checkpoint(directory/"complete.pt")
        if state["step"] != 14 or state["contract"]["world_size"] != 2 or len(state["rng_by_rank"]) != 2:
            raise RuntimeError(f"{name} did not complete two-rank verification")
        rows = [json.loads(line) for line in (directory/"train.jsonl").read_text().splitlines()]
        steady = [row for row in rows if row["step"] > 10]
        seconds = sum(row["seconds_per_update"] for row in steady)/len(steady)
        throughput[name] = dict(seconds_per_update=seconds, effective_batch=state["config"]["train"]["effective_batch"],
                                 data_wait_seconds=sum(row["data_wait_seconds"] for row in steady)/len(steady),
                                 peak_memory_bytes=max(row["max_memory_bytes"] for row in rows),
                                 nominal_training_days=seconds*(50000 if name == "common" else 200000)/86400)
        del state
    report = dict(status="passed", commit=commit, gpus=args.gpus, output=str(root), commands=commands,
                   throughput=throughput, scope="full 512px public backbones; actual production batches; shortened schedules; pause/resume; real packets")
    (root/"passed.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="engineering verification directory")
    parser.add_argument("--production-root", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--fault-drill", action="store_true")
    parser.add_argument("--fault-worker", action="store_true")
    args = parser.parse_args()
    if args.fault_worker:
        import torch
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        allocation = torch.ones(256*1024**2, dtype=torch.uint8, device=f"cuda:{rank}")
        torch.cuda.synchronize()
        print(json.dumps(dict(fault_drill_rank=rank, allocated_bytes=allocation.numel())), flush=True)
        time.sleep(3 if rank == 0 else 300)
        raise SystemExit(23)
    root, production = Path(args.root).resolve(), Path(args.production_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    production.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    drill = root/f"guard_drill_{commit[:10]}.json"
    if args.fault_drill and not drill.exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(Path(__file__).resolve()), "--root", str(root), "--production-root", str(production),
                   "--gpus", args.gpus, "--fault-worker"]
        result = subprocess.run(command, cwd=REPO, env=env)
        drill.write_text(json.dumps(dict(commit=commit, intentional_failure=True, returncode=result.returncode))+"\n")
        raise SystemExit(result.returncode or 24)
    verification = root/f"smoke_{commit[:10]}"
    passed = verification/"passed.json"
    if not passed.exists():
        verify(args, verification, commit)
    else:
        report = json.loads(passed.read_text())
        if report["commit"] != commit or report["gpus"] != args.gpus:
            raise RuntimeError("verification identity differs")
    if not (production/"production.start").exists():
        print("Verification passed; guardian will retain both GPUs until production is armed.", flush=True)
        return
    os.execv(sys.executable, [sys.executable, str(REPO/"scripts/launch_bt_lowrate_c.py"),
                              "--root", str(production), "--gpus", args.gpus])


if __name__ == "__main__":
    main()
