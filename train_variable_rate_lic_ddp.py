import argparse
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderDC
from peft import LoraConfig, get_peft_model
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


CHECKPOINT_SCHEMA_VERSION = 2
STAGE_NAME = "variable_quality_codec"


def filter_supported_vae_modules(model):
    pattern = re.compile(
        r"^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv_inverted|"
        r"conv_point|to_k|to_q|to_v|to_out\.0)$"
    )
    supported = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return [
        name
        for name, module in model.named_modules()
        if pattern.match(name) and isinstance(module, supported)
    ]


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    quality = config.get("quality_control") or {}
    if int(quality.get("num_levels", 0)) != 6:
        raise ValueError("Stage 1 requires exactly 6 quality levels")
    lambda_min = float(quality.get("lambda_min", 0.0))
    lambda_max = float(quality.get("lambda_max", 0.0))
    if not 0 < lambda_min < lambda_max:
        raise ValueError("quality_control requires 0 < lambda_min < lambda_max")
    if (
        int(config["train"]["max_steps"]) > 5000
        and not quality.get("calibration_confirmed", False)
    ):
        raise ValueError(
            "A main Stage 1 run requires quality_control.calibration_confirmed=true"
        )
    source_checkpoint = config["model"].get("source_checkpoint")
    if not source_checkpoint or source_checkpoint.startswith("/path/to/"):
        raise ValueError("model.source_checkpoint must point to the M0 checkpoint")
    return config


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def log_interpolated_lambda(quality_index, lambda_min, lambda_max, num_levels=6):
    quality = quality_index.to(dtype=torch.float32)
    normalized = quality / float(num_levels - 1)
    return torch.exp(
        math.log(lambda_min)
        + normalized * (math.log(lambda_max) - math.log(lambda_min))
    )


def likelihood_bpp(likelihoods, num_pixels):
    return sum(
        torch.log(values).sum() / (-math.log(2) * num_pixels)
        for values in likelihoods.values()
    )


def load_source_checkpoint(path, use_ema):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_key = "ema" if use_ema else "model"
    if state_key not in checkpoint:
        raise KeyError(f"source checkpoint lacks {state_key!r}: {path}")
    source_state = checkpoint[state_key]
    required = {
        "state_dict_codec",
        "state_dict_vae",
        "state_dict_DiT",
        "state_dict_text_proj",
    }
    missing = sorted(required.difference(source_state))
    if missing:
        raise KeyError(f"source checkpoint lacks Pure SANA state: {missing}")
    return checkpoint, source_state


class VariableRateLICStage(nn.Module):
    def __init__(self, config, source_state):
        super().__init__()
        model_config = config["model"]
        self.vae = AutoencoderDC.from_pretrained(
            model_config["dit_path"], subfolder="vae"
        )
        vae_params = model_config.get("vae_params") or {}
        if vae_params.get("vae_lora", False):
            lora_config = LoraConfig(
                r=int(vae_params["lora_rank_vae"]),
                lora_alpha=int(vae_params["lora_alpha_vae"]),
                target_modules=filter_supported_vae_modules(self.vae),
                bias="none",
                init_lora_weights="gaussian",
            )
            self.vae = get_peft_model(self.vae, lora_config)
            expected_lora = {
                name for name in self.vae.state_dict() if "lora" in name
            }
            missing_lora = sorted(
                expected_lora.difference(source_state["state_dict_vae"])
            )
            if missing_lora:
                raise RuntimeError(
                    "M0 checkpoint lacks configured VAE LoRA weights: "
                    + ", ".join(missing_lora[:5])
                )
            load_result = self.vae.load_state_dict(
                source_state["state_dict_vae"], strict=False
            )
            if load_result.unexpected_keys:
                raise RuntimeError(
                    "M0 VAE LoRA keys do not match the configured VAE: "
                    + ", ".join(load_result.unexpected_keys[:5])
                )
        elif source_state["state_dict_vae"]:
            raise RuntimeError(
                "M0 contains VAE LoRA weights but model.vae_params.vae_lora is false"
            )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.codec = LatentCodec(**model_config["codec_param"])

        load_result = self.codec.load_state_dict(
            source_state["state_dict_codec"], strict=False
        )
        allowed_missing = {
            "quality_controller.encoder_scale_table",
            "quality_controller.decoder_scale_table",
        }
        active_missing = set(load_result.missing_keys).difference(allowed_missing)
        if active_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "M0 codec is incompatible with the variable-rate adapter; "
                f"missing={sorted(active_missing)}, "
                f"unexpected={sorted(load_result.unexpected_keys)}"
            )
        if set(load_result.missing_keys) != allowed_missing:
            raise RuntimeError(
                "M0 initialization must miss exactly the identity quality tables; "
                f"got {sorted(load_result.missing_keys)}"
            )

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def encode_clean_latent(self, image):
        with torch.no_grad():
            return self.vae.encode(image).latent * self.vae.config.scaling_factor

    def forward(self, image, quality_index):
        clean_latent = self.encode_clean_latent(image)
        codec_output = self.codec(clean_latent, quality_index)
        reconstructed_latent = codec_output["mean"]
        reconstruction = self.vae.decode(
            reconstructed_latent / self.vae.config.scaling_factor,
            return_dict=False,
        )[0].clamp(-1, 1)
        return clean_latent, reconstructed_latent, reconstruction, codec_output[
            "likelihoods"
        ]


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
def evaluate(model, loader, device, config):
    model.eval()
    quality_config = config["quality_control"]
    lic_config = config["lic_train"]
    totals = torch.zeros(6, 5, device=device, dtype=torch.float64)
    for quality_index in range(6):
        for image in loader:
            image = image.to(device)
            clean, reconstructed, output, likelihoods = model(
                image, quality_index
            )
            batch_size = image.shape[0]
            num_pixels = image.shape[0] * image.shape[2] * image.shape[3]
            rate = likelihood_bpp(likelihoods, num_pixels)
            latent_mse = F.mse_loss(clean, reconstructed)
            image_mse = F.mse_loss(image, output)
            distortion = (
                float(lic_config["latent_mse_weight"]) * latent_mse
                + float(lic_config["image_mse_weight"]) * image_mse
            )
            quality_tensor = torch.tensor(quality_index, device=device)
            current_lambda = log_interpolated_lambda(
                quality_tensor,
                float(quality_config["lambda_min"]),
                float(quality_config["lambda_max"]),
            )
            loss = rate + current_lambda * distortion
            totals[quality_index] += torch.tensor(
                [
                    float(batch_size),
                    float(loss),
                    float(rate),
                    float(latent_mse),
                    float(image_mse),
                ],
                device=device,
                dtype=torch.float64,
            ) * torch.tensor(
                [1.0, batch_size, batch_size, batch_size, batch_size],
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
            "image_mse": totals[quality_index, 4].item() / count,
        }
    score = sum(value["loss"] for value in metrics.values()) / 6
    return score, metrics


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

    source_path = config["model"]["source_checkpoint"]
    source_checkpoint, source_state = load_source_checkpoint(
        source_path, bool(config["model"].get("source_use_ema", False))
    )
    stage_model = VariableRateLICStage(config, source_state).to(device)
    del source_checkpoint
    source_payload = source_state if rank == 0 else None
    if rank != 0:
        del source_state
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config["optimizer"]["step_lr"]),
        gamma=0.5,
    )

    model = DDP(stage_model, device_ids=[local_rank], static_graph=True)
    model.train()

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
    if int(train_config["global_batch_size"]) % world_size:
        raise ValueError("train.global_batch_size must be divisible by world size")
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
    quality_counts = torch.zeros(6, dtype=torch.long)
    ema_codec = deepcopy(stage_model.codec.state_dict())
    train_steps = 0
    best_score = float("inf")

    if train_config.get("resume", False):
        resume_path = train_config.get("resume_checkpoint")
        if not resume_path:
            raise ValueError("train.resume requires train.resume_checkpoint")
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume.get("stage") != STAGE_NAME:
            raise ValueError(f"cannot resume Stage 1 from {resume.get('stage')!r}")
        model.module.codec.load_state_dict(
            resume["model"]["state_dict_codec"], strict=True
        )
        optimizer.load_state_dict(resume["optimizer"])
        entropy_optimizer.load_state_dict(resume["entropy_optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        quality_generator.set_state(resume["quality_generator_state"])
        quality_counts.copy_(resume["quality_counts"])
        ema_codec = {
            name: value.to(device)
            for name, value in resume["ema"]["state_dict_codec"].items()
        }
        train_steps = int(resume["global_step"])
        best_score = float(resume.get("best_score", best_score))

    source_sha256 = file_sha256(source_path) if rank == 0 else None

    def make_checkpoint():
        codec_state = model.module.codec.state_dict()
        model_payload = deepcopy(source_payload)
        model_payload["state_dict_codec"] = codec_state
        ema_payload = deepcopy(source_payload)
        ema_payload["state_dict_codec"] = ema_codec
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": STAGE_NAME,
            "model": model_payload,
            "ema": ema_payload,
            "optimizer": optimizer.state_dict(),
            "entropy_optimizer": entropy_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": train_steps,
            "best_score": best_score,
            "quality_counts": quality_counts.clone(),
            "quality_generator_state": quality_generator.get_state(),
            "resolved_config": config,
            "git_commit": current_git_commit(),
            "source_checkpoint_sha256": source_sha256,
            "codec_sha256": state_dict_sha256(codec_state),
            "bitstream_format_version": 1,
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
    running = torch.zeros(5, device=device, dtype=torch.float64)
    log_micro_steps = 0
    started = time()

    while train_steps < int(train_config["max_steps"]):
        epoch = (train_steps * accumulation_steps) // max(1, len(train_loader))
        train_sampler.set_epoch(epoch)
        for image in train_loader:
            if train_steps >= int(train_config["max_steps"]):
                break
            image = image.to(device)
            if accum_count == 0:
                quality_index = sample_shared_quality_index(
                    6, device, generator=quality_generator
                )

            clean, reconstructed, output, likelihoods = model(
                image, quality_index
            )
            num_pixels = image.shape[0] * image.shape[2] * image.shape[3]
            rate = likelihood_bpp(likelihoods, num_pixels)
            latent_mse = F.mse_loss(clean, reconstructed)
            image_mse = F.mse_loss(image, output)
            distortion = (
                float(config["lic_train"]["latent_mse_weight"]) * latent_mse
                + float(config["lic_train"]["image_mse_weight"]) * image_mse
            )
            current_lambda = log_interpolated_lambda(
                quality_index,
                float(config["quality_control"]["lambda_min"]),
                float(config["quality_control"]["lambda_max"]),
            )
            loss = rate + current_lambda * distortion
            (loss / accumulation_steps).backward()
            accum_count += 1
            running += torch.stack(
                [
                    loss.detach().double(),
                    rate.detach().double(),
                    latent_mse.detach().double(),
                    image_mse.detach().double(),
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

            quality_counts[int(quality_index.item())] += 1
            quality_index = None
            accum_count = 0
            train_steps += 1

            if train_steps % int(train_config["log_every"]) == 0:
                dist.all_reduce(running, op=dist.ReduceOp.SUM)
                averages = running / (log_micro_steps * world_size)
                if rank == 0:
                    elapsed = max(time() - started, 1e-6)
                    logger.info(
                        "step=%d loss=%.6f rate=%.6f latent_mse=%.6f "
                        "image_mse=%.6f lambda=%.6f steps/s=%.3f q_counts=%s",
                        train_steps,
                        *averages.tolist(),
                        int(train_config["log_every"]) / elapsed,
                        quality_counts.tolist(),
                    )
                    writer.add_scalar("Train/loss", averages[0], train_steps)
                    writer.add_scalar("Train/rate", averages[1], train_steps)
                    writer.add_scalar("Train/latent_mse", averages[2], train_steps)
                    writer.add_scalar("Train/image_mse", averages[3], train_steps)
                running.zero_()
                log_micro_steps = 0
                started = time()

            if train_steps % int(train_config["eval_every"]) == 0:
                score, metrics = evaluate(model, valid_loader, device, config)
                if rank == 0:
                    logger.info("validation=%s", json.dumps(metrics))
                    is_best = score < best_score
                    best_score = min(best_score, score)
                    if is_best:
                        torch.save(make_checkpoint(), checkpoint_dir / "best_ckpt.pt")
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
        logger.info("Stage 1 complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/variable_rate_lic_pilot.yaml"
    )
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume_checkpoint is not None:
        config["train"]["resume"] = True
        config["train"]["resume_checkpoint"] = args.resume_checkpoint
    if args.global_batch_size is not None:
        if args.global_batch_size < 1:
            raise ValueError("--global_batch_size must be positive")
        config["train"]["global_batch_size"] = args.global_batch_size
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
