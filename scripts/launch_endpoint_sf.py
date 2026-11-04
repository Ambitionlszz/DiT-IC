"""Plan matched endpoint-SF ablations; --launch explicitly starts two GPU ranks."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.experiment import (asset_identity, check_stage_model, load_config,
                                            validate_endpoint_initialization, AE_HYBRID_PIXEL_METHOD)


PRIMARY_VARIANTS = ("c", "c_d", "c_sf", "c_d_sf")
CONTROL_VARIANTS = ("c_d_same_rate_sf", "c_d_gt_sf")
VARIANTS = {
    "c": (False, False, "adjacent_ema"),
    "c_d": (True, False, "adjacent_ema"),
    "c_sf": (False, True, "adjacent_ema"),
    "c_d_sf": (True, True, "adjacent_ema"),
    "c_d_same_rate_sf": (True, True, "same_rate_ema"),
    "c_d_gt_sf": (True, True, "clean_latent"),
}
D_METHOD = "precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4"
FAMILIES = {"legacy": ("endpoint_sf", D_METHOD),
            "hybrid": ("endpoint_sf_hybrid", AE_HYBRID_PIXEL_METHOD)}


def shared_signature(config):
    """Remove exactly the declared factorial switches and storage locations."""
    shared = deepcopy(config)
    shared.pop("decoder")
    for key in ("output", "variant", "seed"):
        shared["train"].pop(key)
    for key in ("enabled", "target"):
        shared["endpoint_self_flow"].pop(key)
    return shared


def ablation_configs(seed=903, family="legacy"):
    if family not in FAMILIES:
        raise ValueError("unknown endpoint decoder family")
    directory, d_method = FAMILIES[family]
    configs = {name: load_config(REPO / f"configs/bt_self_flow/{directory}/{name}.yaml")
               for name in VARIANTS}
    signatures, d_settings = [], []
    for name, config in configs.items():
        config["train"]["seed"] = seed
        config["train"]["output"] = str(Path(config["train"]["output"]).parent / f"seed_{seed}" / name)
        d, sf, target = VARIANTS[name]
        settings = config["endpoint_self_flow"]
        expected = {"enabled": sf, "coefficient": 1., "epsilon": 1e-6,
                    "decoder_capture_block": 3, "target": target}
        if settings != expected or config["decoder"]["method"] != (d_method if d else "native"):
            raise ValueError(f"{name} disagrees with its endpoint ablation contract")
        if config["self_flow"] != {"enabled": False, "mixed_sampling": False} or "spatial_self_flow" in config:
            raise ValueError("endpoint recipes must contain no legacy SF or spatial projector settings")
        train = config["train"]
        if tuple(train[key] for key in ("world_size", "batch_size", "accumulation_steps", "effective_batch")) != (2, 4, 1, 8):
            raise ValueError("all endpoint ablations require two ranks, batch4, accumulation1, global batch8")
        if d:
            d_settings.append(config["decoder"])
        signatures.append(shared_signature(config))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("endpoint ablations differ outside the declared D/SF/target switches")
    if any(settings != d_settings[0] for settings in d_settings[1:]):
        raise ValueError("D implementation and initialization must match across all D ablations")
    return configs


def validate_initialization(checkpoint):
    """Reject trained-backend ancestry and uncalibrated common initializations."""
    validate_endpoint_initialization(checkpoint)


def source_identity(repo=REPO):
    """Hash tracked and unignored local source, including not-yet-committed SF files."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo).split(b"\0")
    suffixes = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt", ".sh"}
    files = {}
    for raw_name in sorted(set(names)):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        path = repo / name
        if path.is_file() and path.suffix in suffixes:
            files[name] = sha256_file(path)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": files}


def freeze_source(identity, destination, repo=REPO):
    """Launch from immutable-by-convention copied source, verified before every run."""
    destination = Path(destination)
    manifest_path = destination / "source_identity.json"
    if destination.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != identity:
            raise ValueError("source snapshot differs; use the original frozen launcher or a new run root")
        for name, expected in identity["files"].items():
            if sha256_file(destination / name) != expected:
                raise ValueError(f"frozen source was modified: {name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".endpoint_source_", dir=destination.parent))
    try:
        for name, expected in identity["files"].items():
            source, target = repo / name, temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise ValueError(f"source changed while freezing: {name}")
        (temporary / "source_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            freeze_source(identity, destination, repo)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def check_shared_identity(path, expected, write=False):
    """Pin one shared initializer and source/config contract across all six runs."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("run root already pins a different common initialization or source/config identity")
        return
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as target:
                target.write(json.dumps(expected, indent=2) + "\n")
        except FileExistsError:
            check_shared_identity(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="legacy",
                        help="hybrid selects two AE-conditioned SA+CA blocks; legacy retains mu-QK v4")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--gpus", default="0,1", help="exactly two distinct CUDA device indices or UUIDs")
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("ENDPOINT_SF_RUN_ROOT", "checkpoints/endpoint_sf")),
                        help="hybrid runs use a separate hybrid/ namespace below this root")
    parser.add_argument("--preflight", action="store_true", help="two real updates in an isolated preflight output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print the plan; this is the default")
    mode.add_argument("--launch", action="store_true", help="explicitly probe CUDA and start torchrun")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(gpu_ids):
        parser.error("--gpus must select exactly two distinct devices")
    if args.preflight and args.resume:
        parser.error("preflight starts from common initialization; resume formal runs separately")
    configs = ablation_configs(args.seed, args.family)
    config = configs[args.variant]
    root = args.run_root.resolve()
    if args.family == "hybrid":
        root = root / "hybrid"
    output = root / f"seed_{args.seed}" / ("preflight" if args.preflight else "runs") / args.variant
    config["train"]["output"] = str(output)
    source_path = (args.init_checkpoint or args.resume).resolve()
    checkpoint = read_checkpoint(source_path)
    if args.init_checkpoint:
        validate_initialization(checkpoint)
        check_stage_model(config, checkpoint["config"], checkpoint.get("assets"))
        if checkpoint.get("assets") != asset_identity(config):
            raise ValueError("common initializer and endpoint run use different pretrained assets")
        initialization_hash = sha256_file(source_path)
    else:
        source_config = checkpoint.get("config", {})
        if (checkpoint.get("phase") != "joint"
                or source_config.get("train", {}).get("variant") != args.variant
                or source_config.get("decoder", {"method": "native"}) != config["decoder"]
                or source_config.get("endpoint_self_flow") != config["endpoint_self_flow"]):
            raise ValueError("resume must preserve the endpoint decoder, variant and exact target/loss settings")
        initialization_hash = checkpoint.get("source_sha256")
        if not initialization_hash:
            raise ValueError("resume checkpoint has no common initialization identity")
    # A frozen launch can be resumed from its own launcher without requiring git.
    frozen_manifest = REPO / "source_identity.json"
    identity = (json.loads(frozen_manifest.read_text()) if frozen_manifest.is_file()
                else source_identity())
    code = root / "source"
    signature = json.dumps(shared_signature(config), sort_keys=True, separators=(",", ":")).encode()
    shared = {"initialization_sha256": initialization_hash, "source_sha256": identity["sha256"],
              "shared_config_sha256": hashlib.sha256(signature).hexdigest()}
    if args.family == "hybrid":
        # Preserve existing legacy identities while making the additive family explicit.
        shared["decoder_family"] = args.family
    identity_path = root / "ablation_identity.json"
    check_shared_identity(identity_path, shared)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
               str(code / "train_bt_self_flow.py"), "--config", str(output / "launch_config.json"),
               "--resume" if args.resume else "--init-checkpoint", str(source_path)]
    if args.preflight:
        command += ["--max-updates", "2"]
    manifest = dict(family=args.family, variant=args.variant, target=config["endpoint_self_flow"]["target"], devices=gpu_ids,
                    world_size=2, batch_per_rank=4, accumulation=1, effective_batch=8, seed=args.seed,
                    source=str(source_path), checkpoint_sha256=sha256_file(source_path),
                    initialization_sha256=initialization_hash, code_sha256=identity["sha256"],
                    preflight=args.preflight, dry_run=not args.launch, command=command, output=str(output))
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.launch:
        return
    if (output / "resolved.json").exists() and not args.resume:
        raise ValueError("run already exists; resume or select another run root")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu_ids), PYTHONUNBUFFERED="1")
    probe = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                            "assert torch.cuda.device_count()==2; "
                            "[torch.empty(1, device=f'cuda:{i}') for i in range(2)]"], env=env, cwd=REPO)
    if probe.returncode:
        raise RuntimeError("two-device CUDA probe failed; no training was started")
    freeze_source(identity, code)
    check_shared_identity(identity_path, shared, write=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raise SystemExit(subprocess.call(command, cwd=code, env=env))


if __name__ == "__main__":
    main()
