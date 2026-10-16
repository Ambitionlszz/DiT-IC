"""Scratch BT codec, paired backend/joint runs, and cross-rate Self-Flow.

Use --prepare-backend to create the single shared C/D LoRA initialization.
Use --init-checkpoint for stage handoff; --resume restores all training state.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.bt_self_flow.checkpoint import (METHOD, SCHEMA, BestCheckpoint, DeploymentEMA, capture_rng, clone_state,
                                            read_checkpoint, restore_rng, save_atomic, sha256_file)
from models.bt_self_flow.data import CounterSampler, ImageDataset, counter_seed
from models.bt_self_flow.distributed import make_control_group, rank_zero_section
from models.bt_self_flow.experiment import (asset_identity, check_stage_model, load_config, lr_factor, make_model,
                                            optimization_contract, optimizer_groups, training_contract)
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCrossRateTeacher, cross_rate_loss
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.validation import calibrate_interface, initialize_data_scale, validate


def distributed_setup(device_arg):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: run the GPU preflight before starting BT training")
    device = torch.device(f"cuda:{local_rank}" if device_arg == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_control(config, step, rank, shape, device):
    phase = config["train"]["phase"]
    stream = {"codec": 11, "backend": 12, "joint": 13, "self_flow": 14}[phase]
    rng = random.Random(counter_seed(config["train"]["seed"], step, stream))
    quality = rng.random()
    if phase == "codec":
        ramp = max(0., min(1., (step-config["train"].get("router_uniform_steps", 2000))
                           /config["train"].get("router_ramp_steps", 3000)))
        quality = .5+ramp*(quality-.5)
    sf = config["self_flow"]
    if not sf.get("mixed_sampling") or step % 2 == 0:
        return RateControl(quality)
    b, h, w = shape
    generator = torch.Generator(device="cpu").manual_seed(counter_seed(config["train"]["seed"], step, 100+rank))
    mask = torch.zeros(b, 1, h*w, dtype=torch.bool)
    count = max(1, round(h*w*sf.get("high_fraction", .25)))
    for sample in range(b):
        mask[sample, 0, torch.randperm(h*w, generator=generator)[:count]] = True
    return RateControl(quality, 1., mask.reshape(b, 1, h, w).to(device))


def initialize_from(model, checkpoint, phase):
    source_phase = checkpoint["phase"]
    if phase == "backend":
        if source_phase != "backend_init":
            raise ValueError("C/D backend adaptation requires the shared backend_init checkpoint")
        model.load_bt_state(checkpoint["raw"], initialize_conditioner=True)
    elif phase == "joint":
        if source_phase != "backend" or checkpoint["backend"] != model.backend:
            raise ValueError("joint stage requires its own completed backend checkpoint")
        if checkpoint["step"] != checkpoint["config"]["train"]["steps"]:
            raise ValueError("backend adaptation has not completed")
        model.load_bt_state(checkpoint["raw"])
    elif phase == "self_flow":
        if source_phase not in ("joint", "self_flow") or checkpoint["backend"] != "D":
            raise ValueError("cross-rate continuation requires an adapted D backend")
        if checkpoint["step"] != checkpoint["config"]["train"]["steps"]:
            raise ValueError("cross-rate continuation requires a completed source phase; use --resume otherwise")
        model.load_bt_state(checkpoint["raw"])
    else:
        raise ValueError("codec phase starts from scratch; use --resume to continue it")
    if not model.distortion_scale.calibrated.item():
        raise ValueError("stage handoff requires calibrated distortion scales")


def prepare_backend(config, source, output, device):
    if Path(output).exists():
        raise ValueError("shared backend initialization already exists; choose a new output")
    checkpoint = read_checkpoint(source)
    if checkpoint["phase"] != "codec" or not checkpoint.get("calibration", {}).get("rate_span_pass"):
        raise ValueError("common codec has not passed calibration/rate-span validation")
    if config["train"].get("require_quality_span", False) and not checkpoint["calibration"].get("quality_span_pass"):
        raise ValueError("common codec has not shown higher bitrate and lower latent distortion at high quality")
    if checkpoint["step"] != checkpoint["config"]["train"]["steps"]:
        raise ValueError("shared codec phase has not completed")
    check_stage_model(config, checkpoint["config"])
    if checkpoint["assets"] != asset_identity(config):
        raise ValueError("shared initialization pretrained assets differ")
    target = json.loads(json.dumps(config))
    target["train"]["phase"] = "backend"
    target["model"]["backend"] = "C"
    target["self_flow"] = {"enabled": False, "mixed_sampling": False}
    model = make_model(target, device)
    model.load_bt_state(checkpoint["raw"], codec_only=True)
    state = clone_state(model.export_state(), cpu=True)
    source_hash = sha256_file(source)
    save_atomic(output, {"method": METHOD, "schema": SCHEMA, "phase": "backend_init", "backend": "shared",
                         "step": 0, "raw": state, "ema": state, "config": target,
                         "assets": asset_identity(config), "source_sha256": source_hash,
                         "calibration": checkpoint["calibration"]})
    print(json.dumps({"backend_init": str(output), "source_sha256": source_hash}), flush=True)


def auxiliary_update(codec, optimizer, world):
    if optimizer is None:
        return
    optimizer.zero_grad(set_to_none=True)
    loss = codec.entropy_bottleneck_loss()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite entropy auxiliary loss")
    loss.backward()
    if world > 1:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
    optimizer.step()


def train(config, args, rank, world, device):
    control_group = make_control_group(world)
    settings = config["train"]
    phase = settings["phase"]
    seed_all(settings["seed"]+rank)
    output = Path(settings["output"])
    error = [None]
    if rank == 0:
        try:
            if (output/"resolved.json").exists() and not args.resume:
                raise ValueError("run directory already exists; use --resume or a new run root")
            output.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            error[0] = str(exc)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    validation = ImageDataset(config["data"]["valid"], config["data"].get("valid_crop", config["data"]["crop"]), False,
                               config["data"].get("valid_limit", 32))
    validation_identity = dict(dataset_fingerprint=validation.fingerprint, images=len(validation),
                               crop=validation.crop, actual_rate=settings.get("validation_actual_bpp", False))
    model = make_model(config, device)
    seed_all(settings["seed"]+rank)
    contract = training_contract(config, dataset, world)
    assets = asset_identity(config)
    effective_batch = world*settings["batch_size"]*settings["accumulation_steps"]
    if effective_batch != settings.get("effective_batch", effective_batch):
        raise ValueError("GPU count/batch/accumulation does not match the effective batch contract")
    resume = read_checkpoint(args.resume) if args.resume else None
    source_hash = None
    calibration = None
    if resume:
        if optimization_contract(resume["contract"]) != contract or resume["assets"] != assets:
            raise ValueError("resume contract, dataset, assets, batch or training settings differ")
        model.load_bt_state(resume["raw"])
        source_hash, calibration = resume.get("source_sha256"), resume.get("calibration")
    elif args.init_checkpoint:
        source = read_checkpoint(args.init_checkpoint)
        if source["assets"] != assets:
            raise ValueError("stage handoff pretrained assets differ")
        check_stage_model(config, source["config"])
        initialize_from(model, source, phase)
        source_hash, calibration = sha256_file(args.init_checkpoint), source.get("calibration")
    elif phase != "codec":
        raise ValueError("backend/joint/Self-Flow requires an explicit stage checkpoint")
    else:
        initialize_data_scale(model, dataset, settings.get("data_scale_samples", 128), settings["seed"])
    groups, auxiliary = optimizer_groups(model, config)
    # Quantiles are auxiliary-only and manually synchronized. Exclude them from
    # DDP's main reducer rather than reusing a reducer after the main backward.
    ignored = [name for name, _ in model.named_parameters() if name.endswith(".quantiles")]
    model._ddp_params_and_buffers_to_ignore = ignored
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    aux_optimizer = torch.optim.AdamW(auxiliary, lr=config["optimizer"]["aux_lr"], weight_decay=0) if auxiliary else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, settings["steps"], config["optimizer"].get("warmup", 500)))
    # Recent PyTorch reports BF16 emulation as supported even on V100. Select
    # native BF16 only on Ampere or later; V100 uses FP16 with GradScaler.
    native_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
                   and torch.cuda.is_bf16_supported())
    fp16 = device.type == "cuda" and not native_bf16
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    image_loss = None if phase == "codec" else ImageLoss(**config.get("image_loss", {})).to(device).eval()
    teacher = None
    if config["self_flow"].get("enabled"):
        teacher = BTCrossRateTeacher(model.DiT, model.text_proj, model.quality_conditioner, model.teacher_layer).to(device)
    ema = DeploymentEMA(model)
    step = micro_seen = attempts = 0
    rep_weight = 0.
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        if aux_optimizer:
            aux_optimizer.load_state_dict(resume["aux_optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        ema.state = {part: {key: value.to(device) for key, value in values.items()} for part, values in resume["ema"].items()}
        step, micro_seen, attempts = resume["step"], resume["micro_seen"], resume["attempts"]
        rep_weight = resume.get("rep_weight", 0.)
        if teacher is not None:
            teacher.load_ema_state_dict(resume["teacher"])
        restore_rng(resume["rng_by_rank"][rank])
    with rank_zero_section(rank, control_group):
        best = BestCheckpoint(output, step, resume.get("validation_best") if resume else None,
                              identity=validation_identity) if rank == 0 else None

    def validation_report(transition):
        report = validate(model, validation, image_loss, ema.state, transition,
                          packet_samples=0 if phase == "codec" else settings.get("packet_samples", 2),
                          actual_rate=settings.get("validation_actual_bpp", False))
        report["_validation_identity"] = validation_identity
        (output/f"validation_{step:07d}.json").write_text(json.dumps(report, indent=2)+"\n")
        return report

    # After an explicit validation-set migration, establish a new baseline from
    # the resumed weights. Scores from the old dataset cannot select this best.
    with rank_zero_section(rank, control_group):
        if rank == 0 and resume and best.state is None and Path(args.resume).stem.isdecimal():
            transition = min(1., max(step-1, 0)/max(settings["steps"]-1, 1)) if phase == "backend" else 1.
            best.consider(validation_report(transition), step)
            best.publish(args.resume)
    sampler = CounterSampler(len(dataset), settings["seed"], rank, world, settings["batch_size"], micro_seen,
                              stream={"codec": 21, "backend": 22, "joint": 23, "self_flow": 24}[phase])
    loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=sampler,
                        num_workers=config["data"].get("workers", 4), pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(settings["seed"]+rank+999),
                        persistent_workers=config["data"].get("workers", 4) > 0)
    iterator = iter(loader)
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                      broadcast_buffers=False) if world > 1 else model

    def save(name):
        rngs = [None]*world
        if world > 1:
            dist.all_gather_object(rngs, capture_rng(device), group=control_group)
        else:
            rngs[0] = capture_rng(device)
        with rank_zero_section(rank, control_group):
            if rank == 0:
                started_save = time.monotonic()
                print(json.dumps({"checkpoint_saving": name, "step": step}), flush=True)
                checkpoint = {"method": METHOD, "schema": SCHEMA, "phase": phase, "backend": model.backend,
                              "step": step, "micro_seen": micro_seen, "attempts": attempts,
                              "raw": clone_state(model.export_state(), cpu=True), "ema": clone_state(ema.state, cpu=True),
                              "optimizer": optimizer.state_dict(), "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng_by_rank": rngs,
                              "contract": contract, "config": config, "assets": assets, "source_sha256": source_hash,
                              "calibration": calibration, "rep_weight": rep_weight, "validation_best": best.state}
                if teacher is not None:
                    checkpoint["teacher"] = teacher.get_ema_state_dict()
                save_atomic(output/name, checkpoint)
                best.publish(output/name)
                print(json.dumps({"checkpoint_saved": name, "step": step,
                                  "seconds": time.monotonic()-started_save, "best": best.state}), flush=True)

    if rank == 0:
        try:
            commit = subprocess.check_output(["git", "-c", f"safe.directory={Path(__file__).resolve().parent}",
                                               "rev-parse", "HEAD"], text=True).strip()
        except subprocess.SubprocessError:
            commit = "unknown"
        (output/"resolved.json").write_text(json.dumps(dict(config=config, contract=contract, assets=assets, git_commit=commit), indent=2)+"\n")
        print(json.dumps({"phase": phase, "backend": model.backend, "world_size": world,
                          "effective_batch": world*settings["batch_size"]*settings["accumulation_steps"],
                          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    model.train()
    last_log = time.monotonic()
    accumulated_time, accumulated_updates = 0., 0
    pending_images = []
    stop = False
    stop_requested = [False]
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop_requested.__setitem__(0, True))
    try:
        while step < settings["steps"] and not stop:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer:
                aux_optimizer.zero_grad(set_to_none=True)
            if phase == "codec":
                model.codec.router_strength.fill_(max(0., min(1., (step-settings.get("router_uniform_steps", 2000))
                                                               /settings.get("router_ramp_steps", 3000))))
            transition = min(1., step/max(settings["steps"]-1, 1)) if phase == "backend" else 1.
            control = sample_control(config, step, rank, (settings["batch_size"], config["data"]["crop"]//64,
                                                          config["data"]["crop"]//64), device)
            metrics_sum = {}
            for micro in range(settings["accumulation_steps"]):
                if len(pending_images) <= micro:
                    pending_images.append(next(iterator))
                    micro_seen += 1
                image = pending_images[micro].to(device, non_blocking=True)
                synchronize = micro == settings["accumulation_steps"]-1
                context = wrapped.no_sync() if world > 1 and not synchronize else nullcontext()
                with context:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                        result = wrapped(image, rate_control=control, transition=transition, capture=teacher is not None)
                        loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data, image_loss)
                        if teacher is not None:
                            with torch.no_grad():
                                high = model.codec.quantize_analysis(result.source, RateControl(1.))
                                high_scale = model.distortion_scale(result.clean, high["mean"])
                                target = teacher(high["mean"], model.fixed_text_embed,
                                                 model.self_flow_timestep(image.shape[0], device), high_scale["log_rho"])
                            sf_loss, valid_fraction = cross_rate_loss(result.feature, target,
                                                                      high_scale["measured"], result.scale_info["measured"])
                            sf = config["self_flow"]
                            warmup = sf.get("warmup", 1000)
                            ramp = max(0., min(1., (step-warmup)/sf.get("ramp", 2000)))
                            if synchronize and step % sf.get("calibrate_every", 500) == 0:
                                rd_gradient = torch.autograd.grad(loss, result.coarse, retain_graph=True)[0].float().norm()
                                sf_gradient = torch.autograd.grad(sf_loss, result.coarse, retain_graph=True)[0].float().norm()
                                norms = torch.stack([rd_gradient, sf_gradient])
                                if world > 1:
                                    dist.all_reduce(norms)
                                    norms /= world
                                ratio = (norms[0]/norms[1].clamp_min(1e-12)).item()
                                candidate = min(10., sf.get("gradient_target", .05)*ratio)
                                rep_weight = min(.9*rep_weight+.1*candidate, sf.get("gradient_cap", .1)*ratio)
                            loss = loss+ramp*rep_weight*sf_loss
                            metrics.update(sf=sf_loss, teacher_valid=valid_fraction,
                                           sf_weight=sf_loss.new_tensor(ramp*rep_weight),
                                           student_feature_std=result.feature.float().std(dim=1).mean())
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("nonfinite main loss")
                    scaler.scale(loss/settings["accumulation_steps"]).backward()
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.)+value.detach()/settings["accumulation_steps"]
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]],
                                                   config["optimizer"].get("max_grad_norm", 4.))
            if not scaler.is_enabled() and not torch.isfinite(norm).item():
                raise FloatingPointError("nonfinite main gradient norm")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempts += 1
            skipped = scaler.get_scale() < previous_scale
            if skipped:
                if attempts-step > settings.get("max_amp_skips", 100):
                    raise FloatingPointError("too many AMP skipped updates")
                # Retry the same CPU batches at the lower loss scale. Otherwise
                # different overflow patterns would change paired C/D crops.
                continue
            pending_images.clear()
            auxiliary_update(model.codec, aux_optimizer, world)
            scheduler.step()
            ema.update(model)
            if teacher is not None:
                teacher.update_from_student(model.DiT, model.text_proj, model.quality_conditioner, .9999)
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            if step > 10:
                accumulated_time += elapsed
                accumulated_updates += 1
            if step % settings.get("log_every", 100) == 0 or step == 1:
                values = torch.stack(list(metrics_sum.values()))
                if world > 1:
                    dist.all_reduce(values)
                metrics_sum = dict(zip(metrics_sum, (values/world).tolist()))
                if rank == 0:
                    record = dict(step=step, attempts=attempts, lambda_low=control.value, mixed=control.mixed,
                                  transition=transition, grad_norm=float(norm), seconds_per_update=elapsed,
                                  steady_seconds_per_update=accumulated_time/max(accumulated_updates, 1),
                                  max_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                                  **metrics_sum)
                    with (output/"train.jsonl").open("a") as fd:
                        fd.write(json.dumps(record)+"\n")
                    print(json.dumps(record), flush=True)
            improved = False
            if step % settings["eval_every"] == 0 or step == settings["steps"]:
                with rank_zero_section(rank, control_group):
                    if rank == 0:
                        report = validation_report(transition)
                        improved = best.consider(report, step)
                if world > 1:
                    selected = [improved]
                    dist.broadcast_object_list(selected, src=0, group=control_group)
                    improved = selected[0]
            # Backend/joint endpoints immediately save complete.pt; do not
            # serialize the same multi-GB state twice. The codec endpoint keeps
            # a pre-calibration recovery file because calibration can be long.
            if improved or (step % settings["save_every"] == 0 and (step < settings["steps"] or phase == "codec")):
                save(f"{step:07d}.pt")
            if args.max_updates and step >= args.max_updates:
                stop = True
            request = torch.tensor(int(stop_requested[0]), device=device)
            if world > 1:
                dist.all_reduce(request, op=dist.ReduceOp.MAX)
            stop = stop or bool(request.item())
    except BaseException:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        raise
    completed = step >= settings["steps"]
    if completed and phase == "codec":
        with rank_zero_section(rank, control_group):
            if rank == 0:
                calibration = calibrate_interface(model, dataset, output/"calibration.json",
                                                  settings.get("calibration_samples", 512), settings["seed"])
        if world > 1:
            values = [calibration]
            dist.broadcast_object_list(values, src=0, group=control_group)
            calibration = values[0]
            for buffer in model.distortion_scale.buffers():
                dist.broadcast(buffer, src=0)
        # Endpoint calibration is exact shared metadata, not an EMA average.
        ema.state["distortion_scale"] = clone_state({"scale": model.distortion_scale.state_dict()})["scale"]
    save("complete.pt" if completed else "paused.pt")
    if rank == 0:
        print(json.dumps({"completed": completed, "step": step, "output": str(output),
                          "calibration": calibration}), flush=True)
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint")
    group.add_argument("--resume")
    group.add_argument("--prepare-backend", metavar="COMMON_CODEC_CHECKPOINT")
    parser.add_argument("--output", help="shared backend_init output, with --prepare-backend")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-updates", type=int, help="stop and save without changing the resume contract")
    parser.add_argument("--run-root", help="override storage root consistently for all phases")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_root:
        phase = config["train"]["phase"]
        if phase == "codec":
            suffix = "common"
        elif phase == "self_flow":
            suffix = "E" if config["self_flow"].get("enabled") else "D_mix"
        else:
            suffix = f"{config['model']['backend']}/{phase}"
        config["train"]["output"] = str(Path(args.run_root)/suffix)
    rank, world, device = distributed_setup(args.device)
    try:
        if args.prepare_backend:
            if world != 1 or not args.output:
                raise ValueError("prepare-backend requires one process and --output")
            prepare_backend(config, args.prepare_backend, args.output, device)
        else:
            train(config, args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
