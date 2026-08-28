import argparse
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderDC
import lpips
import pyiqa
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import yaml

from datasets.image import ImageFolder
from models.latent_codec import LatentCodec
from train_nogan_ddp import (
    ResizeIfSmall,
    my_worker_init_fn,
    sample_shared_quality_index,
    set_seed,
    setup_ddp,
    cleanup_ddp,
)


CHECKPOINT_SCHEMA_VERSION = 8
OBJECTIVE_VERSION = "distortion_plus_lambda_rate_v2"
VARIABLE_RATE_MECHANISM = "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
STAGE_NAME = "variable_quality_codec"
PHASE_A = "latent_rd"
PHASE_B = "pixel_perceptual_rd"


def load_config(path, calibration_confirmed=False):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if calibration_confirmed:
        config.setdefault("quality_control", {})["calibration_confirmed"] = True
    quality = config.get("quality_control") or {}
    if int(quality.get("num_levels", 0)) != 6:
        raise ValueError("Stage 1 requires exactly 6 quality levels")
    lic_config = config.get("lic_train") or {}
    phase_a = lic_config.get("phase_a") or {}
    phase_b = lic_config.get("phase_b") or {}
    for phase_name, phase_config in (("phase_a", phase_a), ("phase_b", phase_b)):
        steps = int(phase_config.get("steps", 0))
        if phase_name == "phase_a" and steps < 0:
            raise ValueError("lic_train.phase_a.steps must be non-negative")
        if phase_name == "phase_b" and steps < 1:
            raise ValueError("lic_train.phase_b.steps must be positive")
        if steps == 0:
            continue
        lambda_values = [
            float(value) for value in phase_config.get("lambda_values", [])
        ]
        if len(lambda_values) != 6:
            raise ValueError(
                f"lic_train.{phase_name}.lambda_values must contain 6 values"
            )
        if any(value <= 0 for value in lambda_values) or any(
            left <= right for left, right in zip(lambda_values, lambda_values[1:])
        ):
            raise ValueError(
                f"lic_train.{phase_name}.lambda_values must be positive and "
                "strictly decreasing from low-quality q0 to high-quality q5"
            )
    if (
        int(phase_a.get("steps", 0)) > 0
        and float(phase_a.get("latent_mse_weight", 0.0)) <= 0
    ):
        raise ValueError("lic_train.phase_a.latent_mse_weight must be positive")
    for weight_name in ("pixel_mse_weight", "lpips_weight", "dists_weight"):
        if float(phase_b.get(weight_name, 0.0)) <= 0:
            raise ValueError(
                f"lic_train.phase_b.{weight_name} must be positive"
            )
    total_steps = int(phase_a["steps"]) + int(phase_b["steps"])
    if int(config["train"]["max_steps"]) != total_steps:
        raise ValueError(
            "train.max_steps must equal phase_a.steps + phase_b.steps"
        )
    if (
        int(config["train"]["max_steps"]) > 5000
        and not quality.get("calibration_confirmed", False)
    ):
        raise ValueError(
            "A main Stage 1 run requires quality_control.calibration_confirmed=true"
        )
    model_config = config["model"]
    if model_config.get("codec_initialization") != "scratch":
        raise ValueError("Stage 1 requires model.codec_initialization=scratch")
    if model_config.get("source_checkpoint"):
        raise ValueError("scratch Stage 1 must not define model.source_checkpoint")
    if (model_config.get("vae_params") or {}).get("vae_lora", False):
        raise ValueError("scratch Stage 1 uses the frozen pretrained VAE without LoRA")
    return config


def training_phase(global_step, phase_a_steps):
    return PHASE_A if global_step < phase_a_steps else PHASE_B


def phase_step(global_step, phase_a_steps):
    if global_step < phase_a_steps:
        return global_step
    return global_step - phase_a_steps


def objective_contract(config):
    phase_a = config["lic_train"]["phase_a"]
    phase_b = config["lic_train"]["phase_b"]
    if int(phase_a["steps"]) == 0:
        phase_a_contract = {"disabled": True}
    else:
        phase_a_contract = {
            "lambda_values": [float(value) for value in phase_a["lambda_values"]],
            "latent_mse_weight": float(phase_a["latent_mse_weight"]),
        }
    return {
        "version": OBJECTIVE_VERSION,
        "loss_form": "distortion_plus_rate_lambda",
        "variable_rate_mechanism": VARIABLE_RATE_MECHANISM,
        "codec_initialization": "scratch",
        "phase_a": phase_a_contract,
        "phase_b": {
            "lambda_values": [float(value) for value in phase_b["lambda_values"]],
            "pixel_mse_weight": float(phase_b["pixel_mse_weight"]),
            "lpips_weight": float(phase_b["lpips_weight"]),
            "dists_weight": float(phase_b["dists_weight"]),
            "lpips_input_range": "minus_one_to_one",
            "dists_input_range": "minus_one_to_one",
        },
    }


def initialization_contract(config):
    return {
        "codec": "random",
        "vae": "pretrained_frozen",
        "vae_source": str(config["model"]["dit_path"]),
    }


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def quality_lambda(quality_index, lambda_values):
    device = (
        quality_index.device
        if isinstance(quality_index, torch.Tensor)
        else None
    )
    values = torch.as_tensor(lambda_values, device=device, dtype=torch.float32)
    index = torch.as_tensor(quality_index, device=values.device, dtype=torch.long)
    return values[index]


def rate_distortion_loss(distortion, rate, rate_lambda):
    """FlowCodec/baseline RD loss: distortion plus lambda-weighted rate."""
    return distortion + rate_lambda * rate


def likelihood_bpp(likelihoods, num_pixels):
    return sum(
        torch.log(values).sum() / (-math.log(2) * num_pixels)
        for values in likelihoods.values()
    )


class VariableRateLICStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config["model"]
        self.vae = AutoencoderDC.from_pretrained(
            model_config["dit_path"], subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.codec = LatentCodec(**model_config["codec_param"])

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def encode_clean_latent(self, image):
        with torch.no_grad():
            return self.vae.encode(image).latent * self.vae.config.scaling_factor

    def forward(self, image, quality_index, decode_image=False):
        clean_latent = self.encode_clean_latent(image)
        codec_output = self.codec(clean_latent, quality_index)
        reconstructed_latent = codec_output["mean"]
        reconstruction = None
        if decode_image:
            reconstruction = self.vae.decode(
                reconstructed_latent / self.vae.config.scaling_factor,
                return_dict=False,
            )[0].clamp(-1, 1)
        return {
            "clean_latent": clean_latent,
            "reconstructed_latent": reconstructed_latent,
            "reconstruction": reconstruction,
            "likelihoods": codec_output["likelihoods"],
        }


class PhaseBPerceptualLosses(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net="alex")
        self.dists = pyiqa.create_metric("dists", as_loss=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, reference, reconstruction):
        lpips_loss = self.lpips(reference, reconstruction).mean()
        dists_loss = self.dists(reference, reconstruction).mean()
        return lpips_loss, dists_loss


def create_logger(output_dir, rank):
    logger = logging.getLogger(f"variable_rate_lic_rank{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    if rank == 0:
        stream = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(Path(output_dir) / "log.txt")
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_codec_ema(ema_state, codec, decay=0.999):
    current = codec.state_dict()
    for name, value in current.items():
        source = value.detach()
        if name not in ema_state or ema_state[name].shape != source.shape:
            ema_state[name] = source.clone()
        elif torch.is_floating_point(ema_state[name]):
            ema_state[name].mul_(decay).add_(source, alpha=1 - decay)
        else:
            ema_state[name].copy_(source)


@torch.no_grad()
def evaluate(model, loader, device, config, phase, perceptual_losses=None):
    model.eval()
    phase_key = "phase_a" if phase == PHASE_A else "phase_b"
    phase_config = config["lic_train"][phase_key]
    if phase == PHASE_B and perceptual_losses is None:
        raise ValueError("Phase B evaluation requires perceptual losses")
    totals = torch.zeros(6, 8, device=device, dtype=torch.float64)
    for quality_index in range(6):
        for image in loader:
            image = image.to(device)
            outputs = model(
                image, quality_index, decode_image=phase == PHASE_B
            )
            batch_size = image.shape[0]
            num_pixels = image.shape[0] * image.shape[2] * image.shape[3]
            rate = likelihood_bpp(outputs["likelihoods"], num_pixels)
            latent_mse = F.mse_loss(
                outputs["clean_latent"], outputs["reconstructed_latent"]
            )
            pixel_mse = torch.zeros((), device=device)
            pixel_l1 = torch.zeros((), device=device)
            lpips_loss = torch.zeros((), device=device)
            dists_loss = torch.zeros((), device=device)
            if phase == PHASE_A:
                distortion = (
                    float(phase_config["latent_mse_weight"]) * latent_mse
                )
            else:
                pixel_mse = F.mse_loss(image, outputs["reconstruction"])
                pixel_l1 = F.l1_loss(image, outputs["reconstruction"])
                lpips_loss, dists_loss = perceptual_losses(
                    image, outputs["reconstruction"]
                )
                distortion = (
                    float(phase_config["pixel_mse_weight"]) * pixel_mse
                    + float(phase_config["lpips_weight"]) * lpips_loss
                    + float(phase_config["dists_weight"]) * dists_loss
                )
            quality_tensor = torch.tensor(quality_index, device=device)
            current_lambda = quality_lambda(
                quality_tensor, phase_config["lambda_values"]
            )
            loss = rate_distortion_loss(distortion, rate, current_lambda)
            totals[quality_index] += torch.tensor(
                [
                    float(batch_size),
                    float(loss),
                    float(rate),
                    float(latent_mse),
                    float(pixel_mse),
                    float(pixel_l1),
                    float(lpips_loss),
                    float(dists_loss),
                ],
                device=device,
                dtype=torch.float64,
            ) * torch.tensor(
                [1.0] + [batch_size] * 7,
                device=device,
                dtype=torch.float64,
            )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()

    metrics = {}
    for quality_index in range(6):
        count = totals[quality_index, 0].item()
        metrics[f"q{quality_index}"] = {
            "loss": totals[quality_index, 1].item() / count,
            "likelihood_bpp": totals[quality_index, 2].item() / count,
            "latent_mse": totals[quality_index, 3].item() / count,
            "pixel_mse": totals[quality_index, 4].item() / count,
            "pixel_l1": totals[quality_index, 5].item() / count,
            "lpips": totals[quality_index, 6].item() / count,
            "dists": totals[quality_index, 7].item() / count,
        }
    score = sum(value["loss"] for value in metrics.values()) / 6
    return score, metrics


@torch.no_grad()
def evaluate_with_codec_state(
    model,
    codec_state,
    loader,
    device,
    config,
    phase,
    perceptual_losses=None,
):
    live_state = {
        name: value.detach().cpu().clone()
        for name, value in model.module.codec.state_dict().items()
    }
    model.module.codec.load_state_dict(codec_state, strict=True)
    try:
        score, metrics = evaluate(
            model,
            loader,
            device,
            config,
            phase,
            perceptual_losses,
        )
        scale_diagnostics = model.module.codec.quality_controller.diagnostics()
        return score, metrics, scale_diagnostics
    finally:
        model.module.codec.load_state_dict(live_state, strict=True)


def train(rank, local_rank, world_size, config):
    device = torch.device(f"cuda:{local_rank}")
    train_config = config["train"]
    output_dir = Path(train_config["output_dir"]) / train_config["exp_name"]
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = create_logger(output_dir, rank)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None

    stage_model = VariableRateLICStage(config).to(device)
    codec_parameters = [
        parameter
        for name, parameter in stage_model.codec.named_parameters()
        if parameter.requires_grad and not name.endswith(".quantiles")
    ]
    entropy_parameters = [
        parameter
        for name, parameter in stage_model.codec.named_parameters()
        if parameter.requires_grad and name.endswith(".quantiles")
    ]
    if {id(value) for value in codec_parameters} & {
        id(value) for value in entropy_parameters
    }:
        raise RuntimeError("codec and entropy optimizer parameters overlap")

    phase_a_steps = int(config["lic_train"]["phase_a"]["steps"])
    phase_b_steps = int(config["lic_train"]["phase_b"]["steps"])
    max_steps = phase_a_steps + phase_b_steps

    def build_training_state(phase):
        optimizer = torch.optim.AdamW(
            codec_parameters,
            lr=float(config["optimizer"]["lr"]),
            weight_decay=0,
            betas=(0.9, float(config["optimizer"]["beta2"])),
        )
        entropy_optimizer = torch.optim.AdamW(
            entropy_parameters,
            lr=float(config["optimizer"]["aux_lr"]),
            weight_decay=0,
        )
        offset = 0 if phase == PHASE_A else phase_a_steps
        phase_end = phase_a_steps if phase == PHASE_A else max_steps
        milestones = [
            int(step) - offset
            for step in config["optimizer"]["step_lr"]
            if offset < int(step) <= phase_end
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.5,
        )
        return optimizer, entropy_optimizer, scheduler

    initial_phase = training_phase(0, phase_a_steps)
    optimizer, entropy_optimizer, scheduler = build_training_state(initial_phase)

    model = DDP(stage_model, device_ids=[local_rank])
    model.train()

    perceptual_losses = None

    def get_perceptual_losses():
        nonlocal perceptual_losses
        if perceptual_losses is None:
            perceptual_losses = PhaseBPerceptualLosses().to(device).eval()
        return perceptual_losses

    train_transform = transforms.Compose(
        [
            ResizeIfSmall(config["data"]["path_size"]),
            transforms.RandomCrop(config["data"]["path_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_dataset = ImageFolder(
        config["data"]["data_path"],
        config["data"]["im_exts"],
        transform=train_transform,
    )
    configured_per_gpu_batch = train_config.get("per_gpu_batch_size")
    if configured_per_gpu_batch is not None:
        batch_size = int(configured_per_gpu_batch)
        if batch_size < 1:
            raise ValueError("train.per_gpu_batch_size must be positive")
        train_config["global_batch_size"] = batch_size * world_size
    else:
        if int(train_config["global_batch_size"]) % world_size:
            raise ValueError(
                "train.global_batch_size must be divisible by world size"
            )
        batch_size = int(train_config["global_batch_size"]) // world_size
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=my_worker_init_fn,
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
            transforms.CenterCrop(config["data"]["path_size"]),
        ]
    )
    valid_dataset = ImageFolder(
        config["data"]["valid_path"],
        config["data"]["im_exts"],
        transform=valid_transform,
    )
    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    quality_generator = torch.Generator(device="cpu")
    quality_generator.manual_seed(int(train_config["global_seed"]) + 1701)
    phase_quality_counts = torch.zeros(2, 6, dtype=torch.long)
    ema_codec = deepcopy(stage_model.codec.state_dict())
    train_steps = 0
    best_scores = {PHASE_A: float("inf"), PHASE_B: float("inf")}

    if train_config.get("resume", False):
        resume_path = train_config.get("resume_checkpoint")
        if not resume_path:
            raise ValueError("train.resume requires train.resume_checkpoint")
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("stage") != STAGE_NAME:
            raise ValueError(f"cannot resume Stage 1 from {resume.get('stage')!r}")
        if int(resume.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "scratch variable-rate LIC training requires a schema-version 8 "
                "checkpoint"
            )
        if resume.get("objective_contract") != objective_contract(config):
            raise ValueError(
                "resume checkpoint objective/lambda contract does not match config"
            )
        if resume.get("initialization_contract") != initialization_contract(config):
            raise ValueError(
                "resume checkpoint initialization contract does not match config"
            )
        if (
            int(resume.get("phase_a_steps", -1)) != phase_a_steps
            or int(resume.get("phase_b_steps", -1)) != phase_b_steps
        ):
            raise ValueError(
                "resume checkpoint phase boundaries do not match the config"
            )
        model.module.codec.load_state_dict(
            resume["model"]["state_dict_codec"], strict=True
        )
        optimizer.load_state_dict(resume["optimizer"])
        entropy_optimizer.load_state_dict(resume["entropy_optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        quality_generator.set_state(resume["quality_generator_state"])
        phase_quality_counts.copy_(resume["phase_quality_counts"])
        ema_codec = {
            name: value.to(device)
            for name, value in resume["ema"]["state_dict_codec"].items()
        }
        train_steps = int(resume["global_step"])
        expected_phase = training_phase(train_steps, phase_a_steps)
        if resume.get("training_phase") != expected_phase:
            raise ValueError(
                "resume checkpoint training_phase does not match global_step"
            )
        best_scores = {
            PHASE_A: float(resume["best_scores"][PHASE_A]),
            PHASE_B: float(resume["best_scores"][PHASE_B]),
        }

    def make_checkpoint():
        codec_state = model.module.codec.state_dict()
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "model": {"state_dict_codec": codec_state},
            "ema": {"state_dict_codec": ema_codec},
            "optimizer": optimizer.state_dict(),
            "entropy_optimizer": entropy_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": train_steps,
            "training_phase": training_phase(train_steps, phase_a_steps),
            "phase_step": phase_step(train_steps, phase_a_steps),
            "phase_a_steps": phase_a_steps,
            "phase_b_steps": phase_b_steps,
            "best_scores": dict(best_scores),
            "phase_quality_counts": phase_quality_counts.clone(),
            "quality_counts": phase_quality_counts.sum(dim=0),
            "quality_generator_state": quality_generator.get_state(),
            "resolved_config": config,
            "git_commit": current_git_commit(),
            "initialization_contract": initialization_contract(config),
            "objective_contract": objective_contract(config),
            "codec_sha256": state_dict_sha256(codec_state),
            "bitstream_format_version": 1,
            "variable_rate_mechanism": VARIABLE_RATE_MECHANISM,
            "trainable_parameter_names": [
                name
                for name, parameter in model.module.codec.named_parameters()
                if parameter.requires_grad
            ],
        }

    accumulation_steps = int(train_config.get("accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("train.accumulation_steps must be at least 1")
    optimizer.zero_grad(set_to_none=True)
    entropy_optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    quality_index = None
    running = torch.zeros(8, device=device, dtype=torch.float64)
    log_micro_steps = 0
    started = time()
    if rank == 0:
        logger.info(
            "starting phase=%s global_step=%d phase_step=%d phase_a_steps=%d "
            "phase_b_steps=%d",
            training_phase(train_steps, phase_a_steps),
            train_steps,
            phase_step(train_steps, phase_a_steps),
            phase_a_steps,
            phase_b_steps,
        )

    while train_steps < max_steps:
        epoch = (train_steps * accumulation_steps) // max(1, len(train_loader))
        train_sampler.set_epoch(epoch)
        for image in train_loader:
            if train_steps >= max_steps:
                break
            image = image.to(device)
            if accum_count == 0:
                quality_index = sample_shared_quality_index(
                    6, device, generator=quality_generator
                )

            current_phase = training_phase(train_steps, phase_a_steps)
            phase_index = 0 if current_phase == PHASE_A else 1
            phase_key = "phase_a" if current_phase == PHASE_A else "phase_b"
            phase_config = config["lic_train"][phase_key]
            outputs = model(
                image,
                quality_index,
                decode_image=current_phase == PHASE_B,
            )
            num_pixels = image.shape[0] * image.shape[2] * image.shape[3]
            rate = likelihood_bpp(outputs["likelihoods"], num_pixels)
            latent_mse = F.mse_loss(
                outputs["clean_latent"], outputs["reconstructed_latent"]
            )
            pixel_mse = torch.zeros((), device=device)
            pixel_l1 = torch.zeros((), device=device)
            lpips_loss = torch.zeros((), device=device)
            dists_loss = torch.zeros((), device=device)
            if current_phase == PHASE_A:
                distortion = (
                    float(phase_config["latent_mse_weight"]) * latent_mse
                )
            else:
                pixel_mse = F.mse_loss(image, outputs["reconstruction"])
                pixel_l1 = F.l1_loss(image, outputs["reconstruction"])
                lpips_loss, dists_loss = get_perceptual_losses()(
                    image, outputs["reconstruction"]
                )
                distortion = (
                    float(phase_config["pixel_mse_weight"]) * pixel_mse
                    + float(phase_config["lpips_weight"]) * lpips_loss
                    + float(phase_config["dists_weight"]) * dists_loss
                )
            current_lambda = quality_lambda(
                quality_index, phase_config["lambda_values"]
            )
            loss = rate_distortion_loss(distortion, rate, current_lambda)
            (loss / accumulation_steps).backward()
            accum_count += 1
            running += torch.stack(
                [
                    loss.detach().double(),
                    rate.detach().double(),
                    latent_mse.detach().double(),
                    pixel_mse.detach().double(),
                    pixel_l1.detach().double(),
                    lpips_loss.detach().double(),
                    dists_loss.detach().double(),
                    current_lambda.detach().double(),
                ]
            )
            log_micro_steps += 1

            if accum_count != accumulation_steps:
                continue

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["optimizer"]["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            entropy_loss = model.module.codec.entropy_bottleneck_loss()
            if not torch.isfinite(entropy_loss).item():
                raise FloatingPointError("non-finite entropy-bottleneck loss")
            entropy_optimizer.zero_grad(set_to_none=True)
            entropy_loss.backward()
            entropy_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_codec_ema(ema_codec, model.module.codec)

            phase_quality_counts[phase_index, int(quality_index.item())] += 1
            quality_index = None
            accum_count = 0
            train_steps += 1

            if train_steps % int(train_config["log_every"]) == 0:
                dist.all_reduce(running, op=dist.ReduceOp.SUM)
                averages = running / (log_micro_steps * world_size)
                if rank == 0:
                    elapsed = max(time() - started, 1e-6)
                    completed_phase_steps = (
                        train_steps
                        if current_phase == PHASE_A
                        else train_steps - phase_a_steps
                    )
                    logger.info(
                        "phase=%s step=%d phase_step=%d loss=%.6f rate=%.6f "
                        "latent_mse=%.6f pixel_mse=%.6f pixel_l1=%.6f "
                        "lpips=%.6f dists=%.6f lambda=%.6f steps/s=%.3f q_counts=%s",
                        current_phase,
                        train_steps,
                        completed_phase_steps,
                        *averages.tolist(),
                        int(train_config["log_every"]) / elapsed,
                        phase_quality_counts[phase_index].tolist(),
                    )
                    writer.add_scalar("Train/loss", averages[0], train_steps)
                    writer.add_scalar("Train/rate", averages[1], train_steps)
                    writer.add_scalar("Train/latent_mse", averages[2], train_steps)
                    writer.add_scalar("Train/pixel_mse", averages[3], train_steps)
                    writer.add_scalar("Train/pixel_l1", averages[4], train_steps)
                    writer.add_scalar("Train/lpips", averages[5], train_steps)
                    writer.add_scalar("Train/dists", averages[6], train_steps)
                running.zero_()
                log_micro_steps = 0
                started = time()

            if train_steps % int(train_config["eval_every"]) == 0:
                evaluation_phase = current_phase
                phase_losses = (
                    get_perceptual_losses()
                    if evaluation_phase == PHASE_B
                    else None
                )
                score, metrics, scale_diagnostics = evaluate_with_codec_state(
                    model,
                    ema_codec,
                    valid_loader,
                    device,
                    config,
                    evaluation_phase,
                    phase_losses,
                )
                if rank == 0:
                    bpp_values = [
                        metrics[f"q{quality}"]["likelihood_bpp"]
                        for quality in range(6)
                    ]
                    logger.info(
                        "validation_phase=%s metrics=%s",
                        evaluation_phase,
                        json.dumps(metrics),
                    )
                    logger.info(
                        "validation_quality_scales=%s",
                        json.dumps(scale_diagnostics),
                    )
                    logger.info(
                        "validation_bpp_span=%.8f strictly_increasing=%s",
                        bpp_values[-1] - bpp_values[0],
                        all(
                            left < right
                            for left, right in zip(bpp_values, bpp_values[1:])
                        ),
                    )
                    is_best = score < best_scores[evaluation_phase]
                    best_scores[evaluation_phase] = min(
                        best_scores[evaluation_phase], score
                    )
                    if is_best and train_steps != phase_a_steps:
                        best_name = (
                            "best_phase_a_ckpt.pt"
                            if evaluation_phase == PHASE_A
                            else "best_ckpt.pt"
                        )
                        torch.save(make_checkpoint(), checkpoint_dir / best_name)
                dist.barrier()

            if phase_a_steps > 0 and train_steps == phase_a_steps:
                optimizer, entropy_optimizer, scheduler = build_training_state(
                    PHASE_B
                )
                optimizer.zero_grad(set_to_none=True)
                entropy_optimizer.zero_grad(set_to_none=True)
                ema_codec = deepcopy(model.module.codec.state_dict())
                if rank == 0:
                    torch.save(
                        make_checkpoint(), checkpoint_dir / "phase_a_complete.pt"
                    )
                    logger.info(
                        "phase A complete; reset optimizer, scheduler, and EMA; "
                        "automatically starting phase B"
                    )
                running.zero_()
                log_micro_steps = 0
                started = time()
                dist.barrier()

            if train_steps % int(train_config["ckpt_every"]) == 0:
                if rank == 0:
                    torch.save(
                        make_checkpoint(),
                        checkpoint_dir / f"{train_steps:07d}.pt",
                    )
                dist.barrier()

    if rank == 0:
        torch.save(make_checkpoint(), checkpoint_dir / "last_ckpt.pt")
        writer.close()
        logger.info("variable-rate LIC training complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/variable_rate_lic_pilot.yaml"
    )
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--calibration_confirmed", action="store_true")
    args = parser.parse_args()
    config = load_config(
        args.config,
        calibration_confirmed=args.calibration_confirmed,
    )
    if args.resume_checkpoint is not None:
        config["train"]["resume"] = True
        config["train"]["resume_checkpoint"] = args.resume_checkpoint
    if args.global_batch_size is not None and args.per_gpu_batch_size is not None:
        raise ValueError(
            "use only one of --global_batch_size and --per_gpu_batch_size"
        )
    if args.global_batch_size is not None:
        if args.global_batch_size < 1:
            raise ValueError("--global_batch_size must be positive")
        config["train"]["global_batch_size"] = args.global_batch_size
    if args.per_gpu_batch_size is not None:
        if args.per_gpu_batch_size < 1:
            raise ValueError("--per_gpu_batch_size must be positive")
        config["train"]["per_gpu_batch_size"] = args.per_gpu_batch_size
    if args.accumulation_steps is not None:
        if args.accumulation_steps < 1:
            raise ValueError("--accumulation_steps must be positive")
        config["train"]["accumulation_steps"] = args.accumulation_steps
    set_seed(int(config["train"]["global_seed"]))
    rank, world_size, local_rank = setup_ddp()
    try:
        train(rank, local_rank, world_size, config)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
