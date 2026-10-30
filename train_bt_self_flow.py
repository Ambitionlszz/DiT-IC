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

from models.bt_self_flow.activation_storage import preserve_stride_cpu_offload
from models.bt_self_flow.checkpoint import (METHOD, SCHEMA, BestCheckpoint, DeploymentEMA, capture_rng, clone_state,
                                            evaluation_state, prune_checkpoints, publish_checkpoint_alias, read_checkpoint,
                                            restore_rng, save_atomic, sha256_file)
from models.bt_self_flow.data import CounterSampler, ImageDataset, counter_seed
from models.bt_self_flow.distributed import make_control_group, rank_zero_section
from models.bt_self_flow.experiment import (asset_identity, check_stage_model, load_config, make_model, schedule_factor,
                                            optimization_contract, optimizer_groups, training_contract)
from models.bt_self_flow.losses import ImageLoss, rate_beta, reconstruction_objective
from models.bt_self_flow.model import BTCrossRateTeacher, cross_rate_loss
from models.bt_self_flow.evidence_training import (average_gradients, evidence_backward,
                                                    synchronize_initial_state)
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.precision import attention_stats, reset_attention_stats
from models.bt_self_flow.training_precision import (can_retry_fp32, clip_grad_norm_safe_, finite_state,
    full_precision_recompute, nonfinite_diagnostics, nonfinite_gradient_diagnostics, resume_requires_fp32)
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
    if config.get("quality"):
        return RateControl(rng.choice(config["quality"]["values"]))
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
    extra = {"initialize_evidence": True} if hasattr(model, "evidence_projection") else {}
    if phase == "backend":
        if source_phase != "backend_init":
            raise ValueError("C/D backend adaptation requires the shared backend_init checkpoint")
        model.load_bt_state(checkpoint["raw"], initialize_conditioner=True, **extra,
                                **({"initialize_pixel_adapter": True} if hasattr(model, "pixel_adapter") else {}))
    elif phase == "joint":
        if source_phase == "backend_init" and checkpoint["backend"] == "shared":
            model.load_bt_state(checkpoint["raw"], initialize_conditioner=True, **extra,
                                **({"initialize_pixel_adapter": True} if hasattr(model, "pixel_adapter") else {}))
        elif source_phase == "backend" and checkpoint["backend"] == model.backend:
            # Keep explicit continuation of earlier staged experiments usable.
            if checkpoint["step"] != checkpoint["config"]["train"]["steps"]:
                raise ValueError("backend adaptation has not completed")
            model.load_bt_state(checkpoint["raw"])
        else:
            raise ValueError("joint stage requires shared initialization or its own completed backend checkpoint")
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
    check_stage_model(config, checkpoint["config"], checkpoint["assets"])
    if checkpoint["assets"] != asset_identity(config):
        raise ValueError("shared initialization pretrained assets differ")
    target = json.loads(json.dumps(config))
    target["train"]["phase"] = "joint"
    target["model"]["backend"] = "C"
    target["self_flow"] = {"enabled": False, "mixed_sampling": False}
    model = make_model(target, device)
    source_weights = "ema" if model.codec.precision_conditioning else "raw"
    model.load_bt_state(checkpoint[source_weights], codec_only=True)
    state = clone_state(model.export_state(), cpu=True)
    source_hash = sha256_file(source)
    save_atomic(output, {"method": METHOD, "schema": SCHEMA, "phase": "backend_init", "backend": "shared",
                         "step": 0, "raw": state, "ema": state, "config": target,
                         "assets": asset_identity(config), "source_sha256": source_hash,
                         "calibration": checkpoint["calibration"], "codec_weights": source_weights})
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


def gradient_diagnostics(model):
    """Sampled after unscale/clip; zero is distinct from a missing gradient."""
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name.startswith("codec."):
            part = name.split(".")[1]
        else:
            part = name.split(".")[0]
        buckets.setdefault(part, []).append(parameter.grad.detach().float().norm())
    return {name: torch.stack(norms).norm().item() for name, norms in buckets.items()}


def gradient_failure_diagnostics(model):
    """Per-parameter gradient state when the global gradient norm is nonfinite."""
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = torch.isfinite(grad)
        count = int(finite.sum())
        row = {"name": name, "numel": grad.numel(), "finite": count == grad.numel()}
        if count:
            row["max_abs"] = float(grad[finite].abs().max())
            row["norm"] = float(torch.linalg.vector_norm(grad[finite], dtype=torch.float64))
        else:
            row["max_abs"] = float("inf")
        if count != grad.numel():
            row["nonfinite_elements"] = grad.numel() - count
        rows.append(row)
    rows.sort(key=lambda row: row["max_abs"], reverse=True)
    return {"parameters": rows[:40], "nonfinite": [row for row in rows if not row["finite"]][:40]}


def register_module_trace(model):
    """Record activation and activation-gradient magnitudes per module."""
    forward, backward, inputs_max, handles = {}, {}, {}, []

    def value_absmax(value):
        if not torch.is_tensor(value) or value.numel() == 0:
            return None
        finite = torch.isfinite(value)
        return float(value.detach()[finite].abs().max()) if finite.any() else float("inf")

    def collect(values):
        if not isinstance(values, (tuple, list)):
            values = (values,)
        found = [value_absmax(value) for value in values]
        found = [value for value in found if value is not None]
        return max(found) if found else None

    def wanted(name):
        if name.startswith("quality_conditioner."):
            return True
        if not name.startswith("DiT."):
            return False
        tail = name.rsplit(".", 1)[-1]
        if tail in {"norm_out", "proj_out", "patch_embed", "caption_norm", "time_embed",
                    "attn1", "attn2", "ff", "norm1", "norm2"}:
            return True
        parts = name.split(".")
        return "transformer_blocks" in parts and parts[-1].isdigit()

    for name, module in model.named_modules():
        if not wanted(name):
            continue

        def forward_hook(module, inputs, output, name=name):
            forward[name] = collect(output)
            inputs_max[name] = [value_absmax(value) for value in (inputs if isinstance(inputs, (tuple, list)) else (inputs,))]

        def backward_hook(module, grad_input, grad_output, name=name):
            in_slots = [value_absmax(value) for value in (grad_input if isinstance(grad_input, (tuple, list)) else (grad_input,))]
            out_slots = [value_absmax(value) for value in (grad_output if isinstance(grad_output, (tuple, list)) else (grad_output,))]
            backward[name] = {"grad_in": collect(grad_input), "grad_out": collect(grad_output),
                              "in_slots": in_slots, "out_slots": out_slots}

        try:
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))
        except Exception:
            continue
    return forward, backward, inputs_max, handles


def train(config, args, rank, world, device):
    control_group = make_control_group(world)
    settings = config["train"]
    phase = settings["phase"]
    if os.environ.get("BT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
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
        check_stage_model(config, source["config"], source["assets"])
        initialize_from(model, source, phase)
        source_hash, calibration = sha256_file(args.init_checkpoint), source.get("calibration")
    elif phase != "codec":
        raise ValueError("backend/joint/Self-Flow requires an explicit stage checkpoint")
    else:
        initialize_data_scale(model, dataset, settings.get("data_scale_samples", 128), settings["seed"])
    groups, auxiliary = optimizer_groups(model, config)
    evidence_method = bool(config.get("evidence_self_flow"))
    main_parameters = [p for group in groups for p in group["params"]]
    codec_parameters = [p for group in groups if group["name"] == "codec" for p in group["params"]]
    backend_parameters = [p for group in groups if group["name"] != "codec" for p in group["params"]]
    if evidence_method:
        synchronize_initial_state(model, world)
    # Quantiles are auxiliary-only and manually synchronized. Exclude them from
    # DDP's main reducer rather than reusing a reducer after the main backward.
    ignored = [name for name, _ in model.named_parameters() if name.endswith(".quantiles")]
    model._ddp_params_and_buffers_to_ignore = ignored
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    aux_optimizer = torch.optim.AdamW(auxiliary, lr=config["optimizer"]["aux_lr"], weight_decay=0) if auxiliary else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: schedule_factor(s, settings["steps"], config["optimizer"]))
    # Recent PyTorch reports BF16 emulation as supported even on V100. Select
    # native BF16 only on Ampere or later; V100 uses FP16 with GradScaler.
    native_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
                   and torch.cuda.is_bf16_supported())
    fp16 = device.type == "cuda" and not native_bf16
    amp_dtype = torch.float16 if fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    image_loss = None if phase == "codec" else ImageLoss(
        **config.get("image_loss", {}),
        checkpoint_features=hasattr(model, "pixel_adapter") or evidence_method).to(device).eval()
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
        started_validation = time.monotonic()
        full_decode = settings.get("full_decode_every", 0)
        samples = len(validation) if full_decode and step % full_decode == 0 else settings.get("packet_samples", 2)
        report = validate(model, validation, image_loss, ema.state, transition,
                          packet_samples=0 if phase == "codec" else samples,
                          actual_rate=settings.get("validation_actual_bpp", False),
                          objective=config.get("objective"), quality=config.get("quality"))
        report["_validation_identity"] = validation_identity
        (output/f"validation_{step:07d}.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(validation_completed=step, seconds=time.monotonic()-started_validation,
                              parallel=False, images=len(validation))), flush=True)
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
                                      broadcast_buffers=False) if world > 1 and not evidence_method else model
    module_trace = register_module_trace(model) if os.environ.get("BT_DIAG_TRACE") == "1" else None

    last_saved = {}

    def save(name, resume_micro_seen=None, resume_rng=None, reuse_last=False):
        version = (step, micro_seen, attempts, rep_weight, full_precision_retry)
        if reuse_last:
            reuse = [rank == 0 and last_saved.get("version") == version]
            if world > 1:
                dist.broadcast_object_list(reuse, src=0, group=control_group)
            if reuse[0]:
                with rank_zero_section(rank, control_group):
                    if rank == 0:
                        started_save = time.monotonic()
                        publish_checkpoint_alias(last_saved["path"], output/name)
                        print(json.dumps(dict(checkpoint_saved=name, step=step,
                                               seconds=time.monotonic()-started_save,
                                               reused_checkpoint=str(last_saved["path"]), best=best.state)), flush=True)
                return
        rngs = [None]*world
        rng = capture_rng(device) if resume_rng is None else resume_rng
        if world > 1:
            dist.all_gather_object(rngs, rng, group=control_group)
        else:
            rngs[0] = rng
        with rank_zero_section(rank, control_group):
            if rank == 0:
                started_save = time.monotonic()
                print(json.dumps({"checkpoint_saving": name, "step": step}), flush=True)
                checkpoint = {"method": METHOD, "schema": SCHEMA, "phase": phase, "backend": model.backend,
                              "step": step, "micro_seen": micro_seen if resume_micro_seen is None else resume_micro_seen,
                              "attempts": attempts,
                              "raw": clone_state(model.export_state(), cpu=True), "ema": clone_state(ema.state, cpu=True),
                              "optimizer": optimizer.state_dict(), "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng_by_rank": rngs,
                              "contract": contract, "config": config, "assets": assets, "source_sha256": source_hash,
                              "calibration": calibration, "rep_weight": rep_weight, "validation_best": best.state,
                              "full_precision_retry": full_precision_retry}
                if teacher is not None:
                    checkpoint["teacher"] = teacher.get_ema_state_dict()
                save_atomic(output/name, checkpoint)
                best.publish(output/name)
                prune_checkpoints(output, settings.get("keep_last_checkpoints"), best.state)
                last_saved.update(version=version, path=output/name)
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
                          "decoder_method": getattr(model, "decoder_method", "native"),
                          "precision_consistent": model.codec.precision_consistent,
                          "saved_tensors": config.get("execution", {}).get("saved_tensors", "device"),
                          "checkpoint_perceptual_loss": bool(image_loss and image_loss.checkpoint_features),
                          "evidence_self_flow": config.get("evidence_self_flow"),
                          "gradient_synchronization": "explicit_mean" if evidence_method else "native_DDP",
                          "per_rank_batch": settings["batch_size"],
                          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    model.train()
    writer = None
    if rank == 0 and settings.get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output/"tensorboard"))
    last_log = time.monotonic()
    accumulated_time, accumulated_updates = 0., 0
    pending_images = []
    update_rng = None
    update_rep_weight = rep_weight
    full_precision_retry = resume_requires_fp32(resume, output)
    if resume and resume.get("full_precision_retry") and not full_precision_retry and rank == 0:
        print(json.dumps({"finite_gradient_norm_resume": step,
                          "precision": str(amp_dtype), "stable_clipping": True}), flush=True)

    def recover_nonfinite(diagnostic, kind):
        nonlocal attempts, rep_weight, full_precision_retry
        attempts += 1
        rep_weight = update_rep_weight
        optimizer.zero_grad(set_to_none=True)
        if aux_optimizer:
            aux_optimizer.zero_grad(set_to_none=True)
        state_ok = torch.tensor(int(finite_state(model)), device=device)
        if world > 1:
            dist.all_reduce(state_ok, op=dist.ReduceOp.MIN)
        diagnostic["model_state_finite"] = bool(state_ok.item())
        diagnostics = [None]*world
        if world > 1:
            dist.all_gather_object(diagnostics, diagnostic, group=control_group)
        else:
            diagnostics[0] = diagnostic
        with rank_zero_section(rank, control_group):
            if rank == 0:
                (output/f"nonfinite_{step:07d}_{attempts:07d}.json").write_text(
                    json.dumps({"kind": kind, "ranks": diagnostics}, indent=2)+"\n")
        if not state_ok.item():
            raise FloatingPointError("nonfinite model state; refusing to overwrite a valid recovery point")
        # DDP cannot reuse an abandoned synchronized forward. Single-GPU AMP
        # replays the same accumulated batches once, including unscaled BF16.
        retry = can_retry_fp32(device, world, full_precision_retry, scaler.is_enabled())
        full_precision_retry = full_precision_retry or retry
        save("paused.pt", micro_seen-len(pending_images), update_rng)
        if not retry:
            raise FloatingPointError(
                f"nonfinite main {kind} persists; saved the last completed update for diagnosis")
        restore_rng(update_rng)

    first_update_step = step + 1
    consecutive_amp_skips = 0
    stop = False
    stop_requested = [False]
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop_requested.__setitem__(0, True))
    try:
        if getattr(args, "startup_gate", None):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ready = Path(args.startup_ready)
            ready.write_text(json.dumps(dict(pid=os.getpid(), phase=phase,
                                            batch_size=settings["batch_size"],
                                            accumulation_steps=settings["accumulation_steps"],
                                            ready_for_updates=True))+"\n")
            while not Path(args.startup_gate).exists():
                if stop_requested[0]:
                    raise InterruptedError("stop requested before first update")
                parent = os.environ.get("C_V3_SUPERVISOR_PID")
                if parent and not Path(f"/proc/{parent}").exists():
                    raise InterruptedError("startup supervisor exited")
                time.sleep(.01)
        while step < settings["steps"] and not stop:
            started = time.monotonic()
            data_wait = 0.
            reset_attention_stats()
            if not pending_images:
                update_rng = capture_rng(device)
                update_rep_weight = rep_weight
            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer:
                aux_optimizer.zero_grad(set_to_none=True)
            if phase == "codec" and not model.codec.precision_conditioning:
                model.codec.router_strength.fill_(max(0., min(1., (step-settings.get("router_uniform_steps", 2000))
                                                               /settings.get("router_ramp_steps", 3000))))
            transition = min(1., step/max(settings["steps"]-1, 1)) if phase == "backend" else 1.
            control = sample_control(config, step, rank, (settings["batch_size"], config["data"]["crop"]//64,
                                                          config["data"]["crop"]//64), device)
            metrics_sum = {}
            nonfinite_forward = False
            for micro in range(settings["accumulation_steps"]):
                if len(pending_images) <= micro:
                    loading_started = time.monotonic()
                    pending_images.append(next(iterator))
                    data_wait += time.monotonic()-loading_started
                    micro_seen += 1
                image = pending_images[micro].to(device, non_blocking=True)
                synchronize = micro == settings["accumulation_steps"]-1
                context = wrapped.no_sync() if world > 1 and not evidence_method and not synchronize else nullcontext()
                storage_context = (preserve_stride_cpu_offload(pin_memory=device.type == "cuda")
                                   if config.get("execution", {}).get("saved_tensors") == "stride_preserving_cpu"
                                   else nullcontext())
                with context, storage_context, full_precision_recompute(model, full_precision_retry):
                    if evidence_method:
                        result, loss, metrics = evidence_backward(
                            model, image, control, image_loss, config, step, rank, world, scaler,
                            codec_parameters, backend_parameters, amp_dtype, full_precision_retry)
                    else:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                            enabled=device.type == "cuda" and not full_precision_retry,
                                            cache_enabled=not hasattr(model, "pixel_adapter")):
                            result = wrapped(image, rate_control=control, transition=transition, capture=teacher is not None)
                            loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                                                     image_loss, objective=config.get("objective"),
                                                                     quality=config.get("quality"))
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
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        nonfinite_forward = True
                        diagnostic = dict(step=step, micro=micro, rank=rank, backend=model.backend,
                                          precision="fp32" if full_precision_retry or device.type != "cuda" else str(amp_dtype),
                                          **nonfinite_diagnostics(result, metrics))
                        print(json.dumps({"nonfinite_forward": diagnostic, "rank": rank}), flush=True)
                        break
                    if not evidence_method:
                        scaler.scale(loss/settings["accumulation_steps"]).backward()
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.)+value.detach()/settings["accumulation_steps"]
            if nonfinite_forward:
                del result, loss, metrics
                recover_nonfinite(diagnostic, "loss")
                continue
            if evidence_method:
                average_gradients(main_parameters, world)
            scaler.unscale_(optimizer)
            try:
                norm = clip_grad_norm_safe_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    config["optimizer"].get("max_grad_norm", 4.),
                    error_if_nonfinite=not scaler.is_enabled())
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                diagnostic = dict(step=step, rank=rank, backend=model.backend,
                                  precision="fp32" if full_precision_retry or device.type != "cuda" else str(amp_dtype),
                                  **nonfinite_gradient_diagnostics(model))
                diagnostic["detail"] = gradient_failure_diagnostics(model)
                diagnostic["attention_stats"] = attention_stats()
                if module_trace is not None:
                    diagnostic["module_trace"] = dict(forward=module_trace[0], backward=module_trace[1],
                                                     inputs=module_trace[2])
                print(json.dumps({"nonfinite_gradient": diagnostic, "rank": rank}), flush=True)
                del result, loss, metrics
                recover_nonfinite(diagnostic, "gradient")
                continue
            if norm.dtype == torch.float64 and torch.isfinite(norm) and rank == 0:
                print(json.dumps({"stable_gradient_norm": float(norm), "step": step,
                                  "precision": "fp32" if full_precision_retry else str(amp_dtype)}), flush=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempts += 1
            skipped = scaler.get_scale() < previous_scale
            if skipped:
                consecutive_amp_skips += 1
                if consecutive_amp_skips > settings.get("max_amp_skips", 100):
                    raise FloatingPointError("too many AMP skipped updates")
                # Retry the same CPU batches at the lower loss scale. Otherwise
                # different overflow patterns would change paired C/D crops.
                rep_weight = update_rep_weight
                restore_rng(update_rng)
                continue
            pending_images.clear()
            replayed_in_fp32 = full_precision_retry
            full_precision_retry = False
            consecutive_amp_skips = 0
            auxiliary_update(model.codec, aux_optimizer, world)
            scheduler.step()
            ema.update(model)
            if teacher is not None:
                teacher.update_from_student(model.DiT, model.text_proj, model.quality_conditioner, .9999)
            step += 1
            if replayed_in_fp32 and rank == 0:
                print(json.dumps({"fp32_replay_completed": True, "step": step, "attempts": attempts}), flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            if step > 10:
                accumulated_time += elapsed
                accumulated_updates += 1
            if step % settings.get("log_every", 100) == 0 or step == first_update_step:
                values = torch.stack(list(metrics_sum.values()))
                if world > 1:
                    dist.all_reduce(values)
                metrics_sum = dict(zip(metrics_sum, (values/world).tolist()))
                if rank == 0:
                    record = dict(step=step, attempts=attempts, lambda_low=control.value, mixed=control.mixed,
                                  transition=transition, grad_norm=float(norm), seconds_per_update=elapsed,
                                  data_wait_seconds=data_wait,
                                  steady_seconds_per_update=accumulated_time/max(accumulated_updates, 1),
                                  max_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                                  q=control.value, rate_coefficient=rate_beta(control, config.get("quality")),
                                  learning_rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                                  **metrics_sum)
                    if evidence_method:
                        record["gradient_clip_factor"] = min(1., config["optimizer"].get("max_grad_norm", 4.)
                                                             /max(float(norm), 1e-12))
                    if model.codec.precision_conditioning:
                        with torch.no_grad():
                            delta = model.codec.steps(control, result.source.allocation)
                            record["precision"] = dict(min_step=delta.min().item(), max_step=delta.max().item(),
                                                       geometric_mean_step=delta.log().mean().exp().item(),
                                                       last_micro_allocation_histogram=torch.bincount(
                                                           result.allocation_codes.flatten().long(), minlength=8).tolist())
                        record["gradient_norms_after_clip"] = gradient_diagnostics(model)
                    with (output/"train.jsonl").open("a") as fd:
                        fd.write(json.dumps(record)+"\n")
                    print(json.dumps(record), flush=True)
                    if writer is not None:
                        for name, value in record.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(f"train/{name}", value, step)
                        for name, value in record["learning_rates"].items():
                            writer.add_scalar(f"lr/{name}", value, step)
                        for name, value in record.get("gradient_norms_after_clip", {}).items():
                            writer.add_scalar(f"grad_after_clip/{name}", value, step)
            improved = False
            if step % settings["eval_every"] == 0 or step == settings["steps"]:
                with rank_zero_section(rank, control_group):
                    if rank == 0:
                        report = validation_report(transition)
                        improved = best.consider(report, step)
                        if writer is not None:
                            for q, values in report.items():
                                if not q.startswith("_"):
                                    for name, value in values.items():
                                        writer.add_scalar(f"validation/q{q}/{name}", value, step)
                            writer.flush()
                if world > 1:
                    selected = [improved]
                    dist.broadcast_object_list(selected, src=0, group=control_group)
                    improved = selected[0]
            # Backend/joint endpoints immediately save complete.pt; do not
            # serialize the same multi-GB state twice. The codec endpoint keeps
            # a pre-calibration recovery file because calibration can be long.
            if improved or step in getattr(args, "checkpoint_steps", []) or (step % settings["save_every"] == 0 and (step < settings["steps"] or phase == "codec")):
                save(f"{step:07d}.pt")
            if args.max_updates and step >= args.max_updates:
                stop = True
            request = torch.tensor(int(stop_requested[0]), device=device)
            if world > 1:
                dist.all_reduce(request, op=dist.ReduceOp.MAX)
            stop = stop or bool(request.item())
    except BaseException:
        if writer is not None:
            writer.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        raise
    completed = step >= settings["steps"]
    if completed and phase == "codec":
        with rank_zero_section(rank, control_group):
            if rank == 0:
                if model.codec.precision_conditioning:
                    # Calibrate the exact EMA frontend used for both e2e arms.
                    with evaluation_state(model, ema.state):
                        calibration = calibrate_interface(model, dataset, output/"calibration.json",
                                                          settings.get("calibration_samples", 512), settings["seed"])
                        calibrated = clone_state({"scale": model.distortion_scale.state_dict()}, cpu=True)["scale"]
                    model.distortion_scale.load_state_dict(calibrated)
                    calibration["weights"] = "deployment_ema"
                    (output/"calibration.json").write_text(json.dumps(calibration, indent=2)+"\n")
                else:
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
    # Calibration changes buffers; its endpoint needs a fresh snapshot.
    save("complete.pt" if completed else "paused.pt", reuse_last=not (completed and phase == "codec"))
    if rank == 0:
        print(json.dumps({"completed": completed, "step": step, "output": str(output),
                          "calibration": calibration}), flush=True)
    for sig, handler in old_handlers.items():
        signal.signal(sig, handler)
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint")
    group.add_argument("--resume")
    group.add_argument("--prepare-backend", metavar="COMMON_CODEC_CHECKPOINT")
    parser.add_argument("--output", help="shared backend_init output, with --prepare-backend")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[],
                        help="additional early recovery checkpoints, without changing the resume contract")
    parser.add_argument("--max-updates", type=int, help="stop and save without changing the resume contract")
    parser.add_argument("--run-root", help="override storage root consistently for all phases")
    parser.add_argument("--startup-ready", help="publish readiness after model and optimizer initialization")
    parser.add_argument("--startup-gate", help="wait for this file before the first update")
    args = parser.parse_args()
    if bool(args.startup_ready) != bool(args.startup_gate):
        parser.error("startup-ready and startup-gate must be supplied together")
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
