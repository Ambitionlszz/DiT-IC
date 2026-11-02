"""Validate matched ablations and launch exactly two ranks with batch4/accum1."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.experiment import load_config


VARIANTS = {"c": (False, False, False), "c_d": (True, False, False),
            "c_sf": (False, True, True), "c_d_sf": (True, True, True),
            "c_p": (False, True, False), "c_d_p": (True, True, False)}


def ablation_configs(seed=903):
    configs = {name: load_config(REPO / f"configs/bt_self_flow/spatial_sf/{name}.yaml") for name in VARIANTS}
    signatures = []
    for name, config in configs.items():
        config["train"]["seed"] = seed
        config["train"]["output"] = str(Path(config["train"]["output"]).parent / f"seed_{seed}" / name)
        d, p, sf = VARIANTS[name]
        settings = config["spatial_self_flow"]
        if ((config["decoder"]["method"] != "native", settings["projector_enabled"], settings["enabled"])
                != (d, p, sf)):
            raise ValueError(f"{name} disagrees with its ablation switches")
        train = config["train"]
        if (train["world_size"], train["batch_size"], train["accumulation_steps"], train["effective_batch"]) != (2, 4, 1, 8):
            raise ValueError("all spatial ablations require two ranks, batch4, accumulation1, global batch8")
        shared = deepcopy(config)
        shared.pop("decoder")
        for key in ("output", "variant"):
            shared["train"].pop(key)
        for key in ("enabled", "projector_enabled"):
            shared["spatial_self_flow"].pop(key)
        signatures.append(shared)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("ablation shared model/data/loss/optimizer/training settings differ")
    if configs["c_d"]["decoder"] != configs["c_d_sf"]["decoder"] or configs["c_d"]["decoder"] != configs["c_d_p"]["decoder"]:
        raise ValueError("D implementations differ across paired ablations")
    return configs


def validate_initialization(checkpoint):
    if (checkpoint.get("phase") != "backend_init" or checkpoint.get("backend") != "shared"
            or checkpoint.get("config", {}).get("decoder", {}).get("method", "native") != "native"
            or any(name in checkpoint.get("raw", {}) for name in ("pixel_adapter", "spatial_projector", "projector"))):
        raise ValueError("use the common pure-C backend_init; trained C+D/SF checkpoints bias the ablation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--gpus", default="0,1", help="exactly two CUDA device indices or UUIDs")
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--run-root", type=Path, default=Path("checkpoints/spatial_sf"))
    parser.add_argument("--preflight", action="store_true", help="two real updates in a separate preflight directory")
    parser.add_argument("--dry-run", action="store_true", help="validate configs/checkpoint and print without launching")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(gpu_ids):
        parser.error("--gpus must select exactly two distinct devices")
    if args.preflight and args.resume:
        parser.error("preflight starts from common initialization; resume the actual run separately")
    configs = ablation_configs(args.seed)
    config = configs[args.variant]
    root = args.run_root.resolve() / f"seed_{args.seed}"
    output = root / ("preflight" if args.preflight else "runs") / args.variant
    config["train"]["output"] = str(output)
    source_path = (args.init_checkpoint or args.resume).resolve()
    checkpoint = read_checkpoint(source_path)
    if args.init_checkpoint:
        validate_initialization(checkpoint)
    elif checkpoint.get("config", {}).get("train", {}).get("variant") != args.variant:
        raise ValueError("resume variant differs; each ablation keeps its own optimizer and EMA")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
               str(REPO / "train_bt_self_flow.py"), "--config", str(output / "launch_config.json"),
               "--resume" if args.resume else "--init-checkpoint", str(source_path)]
    if args.preflight:
        command += ["--max-updates", "2"]
    manifest = dict(variant=args.variant, devices=gpu_ids, world_size=2, batch_per_rank=4, accumulation=1,
                    effective_batch=8, seed=args.seed, source=str(source_path), source_sha256=sha256_file(source_path),
                    preflight=args.preflight, command=command, output=str(output))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    # Fail before creating an apparently started run if CUDA/selected devices
    # cannot be initialized. The child inherits the exact same visibility.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu_ids), PYTHONUNBUFFERED="1")
    probe = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                            "assert torch.cuda.device_count()==2; "
                            "[torch.empty(1, device=f'cuda:{i}') for i in range(2)]"], env=env, cwd=REPO)
    if probe.returncode:
        raise RuntimeError("two-device CUDA preflight failed; no training was started")
    if (output / "resolved.json").exists() and not args.resume:
        raise ValueError("run already exists; resume or select another run root")
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raise SystemExit(subprocess.call(command, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
