"""Run one two-GPU joint experiment, with an RD prefix and delayed Self-Flow."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from launch_bt_pair import PairedRun, resume_checkpoint, write_json
from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.experiment import asset_identity, check_stage_model, load_config


class JointRun(PairedRun):
    def exit_status(self, name, code):
        return "failed" if code else ("completed" if (self.root/"joint/complete.pt").exists() else "paused")


def prepare(root):
    config = load_config(REPO/"configs/bt_self_flow/joint_sf_after20k.yaml")
    config["train"]["output"] = str(root/"joint")
    source_path = Path("/data2/ls/checkpoints/bt_self_flow_v1/backend_init.pt")
    warmup_path = Path("/data2/ls/checkpoints/bt_self_flow_v1/common/complete.pt")
    source = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
    warmup = torch.load(warmup_path, map_location="cpu", weights_only=True, mmap=True)
    check_stage_model(config, source["config"], assets_verified=True)
    identity = asset_identity(config)
    if source["assets"] != identity or warmup["assets"] != identity:
        raise ValueError("warmup/shared initialization pretrained assets differ")
    if source["phase"] != "backend_init" or source["step"] != 0 or warmup["phase"] != "codec" or warmup["step"] != 50000:
        raise ValueError("unexpected initialization stage")
    warmup_hash = sha256_file(warmup_path)
    if source["source_sha256"] != warmup_hash:
        raise ValueError("shared initialization has a different warmup source")
    for part in ("codec", "distortion_scale"):
        if source["raw"][part].keys() != warmup["raw"][part].keys():
            raise ValueError(f"incompatible warmup {part}")
        for name, value in source["raw"][part].items():
            if not torch.equal(value, warmup["raw"][part][name]):
                raise ValueError(f"shared initialization is not the exact raw warmup: {part}.{name}")
    initialization = dict(initialization=str(source_path), initialization_sha256=sha256_file(source_path),
                          warmup=str(warmup_path), warmup_sha256=warmup_hash, warmup_step=50000,
                          codec_weights="raw", frontend="bt_explicit_local_step_v1", assets=identity,
                          old_run_paused_at=4333)
    (root/"configs").mkdir(parents=True, exist_ok=True)
    write_json(root/"initialization.json", initialization)
    frozen = root/"configs/joint.yaml"
    text = yaml.safe_dump(config, sort_keys=False)
    if frozen.exists() and frozen.read_text() != text:
        raise ValueError("frozen experiment configuration differs")
    frozen.write_text(text)
    smoke = load_config(frozen)
    smoke["self_flow"].update(start_step=2, ramp=2, calibrate_every=1)
    smoke["train"].update(steps=6, output=str(root/"preflight/joint"), log_every=1, eval_every=4,
                          save_every=4, packet_samples=1, full_decode_every=0, tensorboard=False)
    smoke["data"].update(workers=0, valid_limit=1, valid_crop=512)
    (root/"preflight").mkdir(exist_ok=True)
    (root/"preflight/config.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
    print(json.dumps(dict(initialization_verified=initialization, config=str(frozen))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data2/ls/checkpoints/cd_sf_after10k_b4a2_20261023"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root)
        return
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdecimal() for g in gpus):
        parser.error("two distinct GPU indices are required")
    init = json.loads((root/"initialization.json").read_text())
    if sha256_file(init["initialization"]) != init["initialization_sha256"]:
        raise ValueError("initialization checkpoint bytes changed")
    run_root = root/"preflight" if args.preflight else root
    config_path = run_root/"config.yaml" if args.preflight else root/"configs/joint.yaml"
    config = load_config(config_path)
    (run_root/"logs").mkdir(parents=True, exist_ok=True)
    with (run_root/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = sorted(set(REPO.glob("*.py")) | set((REPO/"models").rglob("*.py")) | set((REPO/"scripts").glob("*.py")))
        manifest = dict(repository=str(REPO), python=sys.executable, gpus=gpus, initialization=init,
                        config=config, code_sha256={str(p.relative_to(REPO)): sha256_file(p) for p in files})
        path = run_root/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("run code/configuration changed; use a distinct experiment directory")
        write_json(path, manifest)
        run = JointRun(run_root, gpus, sys.executable, backend="D")
        run.state["self_flow_start_step"] = config["self_flow"]["start_step"]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, run.stop)
        try:
            run.update("training")
            stages = [("warmup", 2), ("activation", 4), ("resume", None)] if args.preflight else [("joint", None)]
            for name, limit in stages:
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                           "train_bt_self_flow.py", "--config", str(config_path)]
                previous = resume_checkpoint(run_root/"joint")
                if args.preflight and limit and previous and read_checkpoint(previous)["step"] >= limit:
                    print(json.dumps(dict(stage_already_saved=name, limit=limit, checkpoint=str(previous))), flush=True)
                    continue
                command += ["--resume", str(previous)] if previous else ["--init-checkpoint", init["initialization"]]
                if limit:
                    command += ["--max-updates", str(limit)]
                run.start(name, command, gpus)
                run.wait(name)
                print(json.dumps(dict(stage_finished=name, limit=limit)), flush=True)
                if args.preflight:
                    state = read_checkpoint(run_root/"joint"/("paused.pt" if limit else "complete.pt"))
                    if state["step"] != (limit or 6) or len(state["rng_by_rank"]) != 2:
                        raise ValueError("unexpected preflight step/world size")
                    if limit == 2 and "teacher" in state:
                        raise ValueError("teacher started before the RD prefix ended")
                    if limit != 2 and (state["teacher_initialized_step"] != 2 or "teacher" not in state):
                        raise ValueError("teacher was not correctly initialized/restored at the boundary")
                    del state
            if args.preflight:
                rows = [json.loads(line) for line in (run_root/"joint/train.jsonl").read_text().splitlines()]
                if [r["sf_active"] for r in rows] != [False, False, True, True, True, True]:
                    raise ValueError("incorrect SF activation sequence")
                if [r["mixed"] for r in rows] != [False, False, True, False, True, False]:
                    raise ValueError("incorrect local-asymmetry sequence")
                if not any(r.get("sf_weight", 0) > 0 and r.get("teacher_valid", 0) > 0 for r in rows[2:]):
                    raise ValueError("no effective Self-Flow supervision in GPU preflight")
                write_json(run_root/"verification.json", dict(passed=True, updates=6, world_size=2,
                           effective_batch=config["train"]["effective_batch"], teacher_initialized_step=2,
                           peak_memory_bytes=max(r["max_memory_bytes"] for r in rows), records=rows))
            run.update("completed")
        except BaseException as error:
            run.state["error"] = f"{type(error).__name__}: {error}"
            run.update("paused" if run.stop_requested or isinstance(error, InterruptedError) else "failed")
            run.stop(None, None)
            run.reap_stopped()
            raise


if __name__ == "__main__":
    main()
