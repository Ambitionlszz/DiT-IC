"""Check local resources, then optionally run full-size paired GPU smoke runs.

No CUDA: write a blocked report and exit 2. --smoke uses the real pretrained
backbones, codec, perceptual losses, optimizers, packet validation and resume.
Smoke checkpoints are engineering checks, never the C/D research result.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.experiment import load_config


def run_smoke(args, report, visible):
    root = Path(args.output).resolve()/"smoke"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.setdefault("TORCH_HOME", "/data2/ls/checkpoints/torch-hub")
    env.setdefault("HF_HOME", "/data2/ls/checkpoints/hf-cache")
    env["PYTHONPATH"] = str(REPO)+os.pathsep+env.get("PYTHONPATH", "")
    world = min(2, torch.cuda.device_count())
    report["commands"] = []

    def execute(name, command, selected=None):
        started = time.monotonic()
        step_env = dict(env)
        if selected is not None:
            step_env["CUDA_VISIBLE_DEVICES"] = selected
        with (root/f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=REPO, env=step_env, stdout=log, stderr=subprocess.STDOUT)
        report["commands"].append({"name": name, "command": command, "returncode": result.returncode,
                                   "seconds": time.monotonic()-started, "log": str(root/f"{name}.log")})
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {root/f'{name}.log'}")

    def config(name, phase, backend, steps, accumulation):
        value = load_config(args.config)
        value["model"]["backend"] = backend
        if phase == "joint":
            value["model"]["checkpoint_blocks"] = False
        value["train"].update(phase=phase, steps=steps, accumulation_steps=accumulation,
                              batch_size=2, effective_batch=8, output=str(root/name), log_every=1,
                              eval_every=steps, save_every=steps,
                              router_uniform_steps=2, router_ramp_steps=3,
                              data_scale_samples=2, calibration_samples=4, packet_samples=1,
                              require_quality_span=False)
        value["data"].update(valid_limit=1, workers=0)
        value["optimizer"]["warmup"] = 2
        path = root/f"{name}.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return str(path)

    python = sys.executable
    common = config("common", "codec", "C", 14, 4//world)
    command = ([python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
               if world > 1 else [python])
    execute("common", command+["train_bt_self_flow.py", "--config", common])
    first = visible.split(",")[0]
    c_config = config("C_joint", "joint", "C", 14, 4)
    execute("prepare", [python, "train_bt_self_flow.py", "--config", c_config,
                         "--prepare-backend", str(root/"common/complete.pt"),
                         "--output", str(root/"backend_init.pt")], first)
    def backend_smoke(backend, selected):
        cfg = c_config if backend == "C" else config("D_joint", "joint", "D", 14, 4)
        # Exercise the real continuation path instead of doing a fresh second run.
        execute(f"{backend}_pause", [python, "train_bt_self_flow.py", "--config", cfg,
                                    "--init-checkpoint", str(root/"backend_init.pt"), "--max-updates", "2"], selected)
        execute(f"{backend}_resume", [python, "train_bt_self_flow.py", "--config", cfg,
                                     "--resume", str(root/f"{backend}_joint/paused.pt")], selected)
        times = {}
        for phase in ("joint",):
            rows = [json.loads(line) for line in (root/f"{backend}_{phase}/train.jsonl").read_text().splitlines()]
            steady = [row["seconds_per_update"] for row in rows if row["step"] > 10]
            times[phase] = {"seconds_per_effective_batch8_update": sum(steady)/len(steady),
                            "peak_memory_bytes": max(row["max_memory_bytes"] for row in rows)}
        times["nominal_200000_joint_days"] = 200000*times["joint"]["seconds_per_effective_batch8_update"]/86400
        times["note"] = "short-run estimate; excludes validation/checkpoint time"
        return backend, times

    devices = (visible.split(",")*2)[:2]
    if world > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(backend_smoke, backend, selected)
                       for backend, selected in zip(("C", "D"), devices)]
            report["throughput"] = dict(future.result() for future in futures)
    else:
        report["throughput"] = dict(backend_smoke(backend, devices[0]) for backend in ("C", "D"))
    report["smoke_output"] = str(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO/"configs/bt_self_flow/common.yaml"))
    parser.add_argument("--output", required=True, help="new report directory, outside research run outputs")
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 0,1; defaults to visible devices")
    parser.add_argument("--smoke", action="store_true", help="also run real 512px training/resume/packet checks")
    parser.add_argument("--minimum-free-gib", type=float, default=24., help="do not start full smoke below this free VRAM per selected GPU")
    args = parser.parse_args()
    if args.gpus:
        if not all(part.isdigit() for part in args.gpus.split(",")):
            parser.error("--gpus must be comma-separated GPU indices")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    output = Path(args.output)
    if (output/"preflight.json").exists():
        parser.error("report already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.executable, "gpu_smoke_requested": args.smoke,
              "packages": {name: version(name) for name in ("torch", "diffusers", "peft", "compressai", "pyiqa", "lpips")}}
    exit_code = 0
    try:
        config = load_config(args.config)
        paths = list(config["data"]["train"])+list(config["data"]["valid"])+[
            config["model"]["text_path"], str(Path(config["model"]["sana_path"])/"vae/diffusion_pytorch_model.safetensors"),
            str(Path(config["model"]["sana_path"])/"transformer/diffusion_pytorch_model.safetensors")]
        report["assets_exist"] = {path: Path(path).exists() for path in paths}
        if not all(report["assets_exist"].values()):
            raise FileNotFoundError("required assets are missing")
        report["cuda_available"] = torch.cuda.is_available()
        report["gpus"] = [{"name": torch.cuda.get_device_name(i),
                           "native_bf16": torch.cuda.get_device_capability(i)[0] >= 8 and torch.cuda.is_bf16_supported(),
                           "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())] if report["cuda_available"] else []
        free = []
        if report["cuda_available"]:
            try:
                query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                                 "--format=csv,noheader,nounits"], text=True)
                usage = {int(row.split(",")[0]): row.split(",")[1:] for row in query.splitlines()}
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(len(report["gpus"])))))
                for gpu, physical in zip(report["gpus"], visible.split(",")):
                    memory, utilization = usage[int(physical)]
                    gpu.update(free_memory_bytes=int(memory)*1024**2, utilization_percent=int(utilization))
                    free.append(gpu["free_memory_bytes"]/1024**3)
            except (OSError, subprocess.SubprocessError, KeyError, ValueError):
                # CUDA API fallback if the NVML command is unavailable.
                for i, gpu in enumerate(report["gpus"]):
                    gpu["free_memory_bytes"] = torch.cuda.mem_get_info(i)[0]
                    free.append(gpu["free_memory_bytes"]/1024**3)
        if not report["cuda_available"]:
            report.update(status="blocked", reason="CUDA is inaccessible on this host; no 512px GPU training or smoke has run")
            exit_code = 2
        elif min(free) < args.minimum_free_gib:
            report.update(status="blocked", reason=f"selected GPUs have less than {args.minimum_free_gib:g} GiB free; existing workloads were left running")
            exit_code = 2
        else:
            if args.smoke:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(torch.cuda.device_count())))
                run_smoke(args, report, visible)
            report["status"] = "passed" if args.smoke else "resources_ready_smoke_pending"
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        exit_code = 1
    (output/"preflight.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
