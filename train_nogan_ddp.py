import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.optim as optim
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


import os
import sys
import json
import hashlib
import random
import logging
import argparse
import importlib
import subprocess
import numpy as np
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict

from datasets.image import ImageFolder
from models.losses import DISTS_INPUT_CONTRACT
from config_compat import load_pure_sana_config
from self_flow_training import (
    resolve_self_flow_checkpoint_contract,
    all_token_representation_loss,
    calibrate_representation_weight,
    deterministic_self_flow_seed,
    likelihood_bpp,
    sample_iid_uniform_pairs,
    sample_rate_anchored_pairs,
    select_post_burnin_best,
    self_flow_schedule,
    validate_self_flow_config,
    zero_connect_feature,
)


MULTIRATE_CHECKPOINT_SCHEMA_VERSION = 3
VARIABLE_RATE_MECHANISM = "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"


# Bump when entropy-gradient ordering, scaling, or clipping semantics change.
# Version 4 follows the official DiT-IC NoGAN order: main backward/clip/step,
# then recompute and update the entropy-bottleneck quantiles.
ENTROPY_OPTIMIZER_VERSION = 4
HISTORICAL_ENTROPY_OPTIMIZER_VERSION = 3
OFFICIAL_ENTROPY_UPDATE = "official_post_main"
HISTORICAL_ENTROPY_UPDATE = "historical_pre_main"
SUPPORTED_ENTROPY_UPDATES = {
    OFFICIAL_ENTROPY_UPDATE,
    HISTORICAL_ENTROPY_UPDATE,
}


def configured_entropy_update(config):
    mode = config.get("optimizer", {}).get(
        "entropy_update_mode", OFFICIAL_ENTROPY_UPDATE
    )
    if mode not in SUPPORTED_ENTROPY_UPDATES:
        raise ValueError(
            "optimizer.entropy_update_mode must be one of "
            f"{sorted(SUPPORTED_ENTROPY_UPDATES)}, got {mode!r}"
        )
    return mode


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


def sample_shared_quality_index(num_levels, device, generator=None):
    """Sample one step-level quality on rank 0 and broadcast it to all ranks."""
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    broadcast_device = device if dist.get_backend() == "nccl" else "cpu"
    quality_index = torch.zeros((), dtype=torch.long, device=broadcast_device)
    if dist.get_rank() == 0:
        sampled = torch.randint(
            low=0,
            high=num_levels,
            size=(),
            generator=generator,
            device="cpu",
        )
        quality_index.copy_(sampled.to(broadcast_device))
    dist.broadcast(quality_index, src=0)
    return quality_index.to(device)




def codec_models(model, device, **kwargs):
    model_map = {
        'Codec': 'models.DiT_IC.Codec',
    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(device, model, **kwargs)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict) or isinstance(obj, torch.nn.modules.container.OrderedDict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def do_train(rank, local_rank, world_size, train_config):
    device = f'cuda:{local_rank}'
    entropy_update_mode = configured_entropy_update(train_config)
    entropy_optimizer_version = (
        ENTROPY_OPTIMIZER_VERSION
        if entropy_update_mode == OFFICIAL_ENTROPY_UPDATE
        else HISTORICAL_ENTROPY_OPTIMIZER_VERSION
    )
    quality_config = train_config.get("quality_control", {})
    multirate_plain = bool(quality_config.get("enabled", False))
    num_quality_levels = int(quality_config.get("num_levels", 0))
    if multirate_plain and num_quality_levels != 6:
        raise ValueError("multi-rate Plain training requires exactly 6 quality levels")
    self_flow_config = validate_self_flow_config(train_config)
    self_flow_enabled = bool(self_flow_config.get("enabled", False))
    self_flow_checkpoint_contract = (
        resolve_self_flow_checkpoint_contract(self_flow_config)
        if self_flow_enabled
        else None
    )
    rate_anchored_enabled = (
        self_flow_enabled
        and self_flow_config.get(
            "training_distribution", "fully_asymmetric"
        ) == "rate_anchored"
    )
    legal_anchor_ratio = float(
        self_flow_config.get("legal_anchor_ratio", 0.5)
    )
    self_flow_teacher_enabled = bool(
        self_flow_config.get("teacher_enabled", False)
    )
    self_flow_representation_enabled = bool(
        self_flow_config.get("representation_enabled", False)
    )
    deployment_ema_decay = (
        float(self_flow_config.get("deployment_ema_decay", 0.9999))
        if self_flow_enabled
        else 0.999
    )
    teacher_ema_decay = float(
        self_flow_config.get("teacher_ema_decay", 0.9999)
    )
    self_flow_pair_counts = (
        torch.zeros(
            (num_quality_levels, num_quality_levels), dtype=torch.long
        )
        if self_flow_enabled
        else None
    )
    self_flow_anchor_quality_counts = (
        torch.zeros(num_quality_levels, dtype=torch.long)
        if rate_anchored_enabled
        else None
    )
    self_flow_partition_counts = (
        torch.zeros(2, dtype=torch.long)
        if rate_anchored_enabled
        else None
    )
    quality_generator = None
    quality_counts = None
    quality_rate_weights = quality_config.get("rate_weights")
    if quality_rate_weights is not None:
        quality_rate_weights = [float(value) for value in quality_rate_weights]
        if len(quality_rate_weights) != num_quality_levels:
            raise ValueError(
                "quality_control.rate_weights must match num_levels"
            )
    if multirate_plain:
        quality_generator = torch.Generator(device="cpu")
        quality_generator.manual_seed(
            int(train_config["train"]["global_seed"]) + 1701
        )
        if not self_flow_enabled:
            quality_counts = torch.zeros(num_quality_levels, dtype=torch.long)

    # Compute the experiment name consistently across all ranks.
    experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
    model_string_name = train_config['model']['model_type'].replace("/", "-")
    if train_config['train']['exp_name'] is None:
        exp_name = f'{experiment_index:03d}-{model_string_name}'
    else:
        exp_name = train_config['train']['exp_name']
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = os.path.abspath(f"{train_config['train']['output_dir']}/tensorboard_logs/{exp_name}")
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)

    checkpoint_dir = f"{train_config['train']['output_dir']}/{exp_name}/checkpoints"

    # Create model:
    params = dict(train_config.get("model", {}).get("params", {}))
    params["training"] = True
    model = codec_models(train_config['model']['model_type'], device, **params).to(device)
    ema_exact_parts = (
        ("state_dict_codec",) if model.freeze_codec else ()
    )
    source_checkpoint_sha256 = (
        file_sha256(params["codec_path"])
        if rank == 0 and multirate_plain
        else None
    )
    loss_module = model.loss
    model.loss = None
    main_parameters = model.get_trainable_params()
    self_flow_teacher = (
        model.build_self_flow_teacher().to(device)
        if self_flow_teacher_enabled
        else None
    )
    resolved_representation_weight = (
        None if self_flow_representation_enabled else 0.0
    )
    teacher_resynced_after_warmup = False

    if rank==0:
        ema = deepcopy(model.get_state_dicts_for_save())

    if rank == 0:
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Trainable Parameters: {sum(p.numel() for p in main_parameters) / 1e6:.2f}M")
        trainable_codec_params = sum(
            parameter.numel()
            for parameter in model.codec.parameters()
            if parameter.requires_grad
        )
        logger.info(
            f"Trainable Codec Parameters: {trainable_codec_params / 1e6:.2f}M"
        )
        relay_parameters = (
            sum(parameter.numel() for parameter in model.decoder_relay.parameters())
            if model.decoder_relay is not None
            else 0
        )
        logger.info(
            f"Decoder Relay: {model.decoder_relay_mode}, "
            f"trainable={relay_parameters / 1e6:.3f}M"
        )
        if self_flow_enabled:
            projector_parameters = (
                sum(
                    parameter.numel()
                    for parameter in model.representation_projector.parameters()
                )
                if model.representation_projector is not None
                else 0
            )
            teacher_parameters = (
                sum(parameter.numel() for parameter in self_flow_teacher.parameters())
                if self_flow_teacher is not None
                else 0
            )
            logger.info(
                "Self-Flow full path: " + json.dumps(self_flow_config, sort_keys=True)
            )
            logger.info(
                f"Self-Flow projector={projector_parameters / 1e6:.3f}M, "
                f"truncated_teacher={teacher_parameters / 1e6:.3f}M"
            )
        if quality_rate_weights is not None:
            logger.info(
                "Per-quality rate weights: "
                + json.dumps(quality_rate_weights)
            )
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
    entropy_parameters = model.get_entropy_parameters()
    main_parameter_ids = {id(parameter) for parameter in main_parameters}
    entropy_parameter_ids = {id(parameter) for parameter in entropy_parameters}
    if len(main_parameter_ids) != len(main_parameters):
        raise RuntimeError("Main optimizer parameter list contains duplicates")
    if len(entropy_parameter_ids) != len(entropy_parameters):
        raise RuntimeError("Entropy optimizer parameter list contains duplicates")
    if main_parameter_ids & entropy_parameter_ids:
        raise RuntimeError(
            "Main and entropy optimizer parameter sets must be disjoint"
        )
    model_named_parameters = list(model.named_parameters())
    model_parameter_names = {
        id(parameter): name
        for name, parameter in model_named_parameters
    }
    trainable_parameter_ids = {
        id(parameter)
        for _, parameter in model_named_parameters
        if parameter.requires_grad
    }
    trainable_parameter_names = {
        id(parameter): name
        for name, parameter in model_named_parameters
        if parameter.requires_grad
    }
    optimizer_parameter_ids = main_parameter_ids | entropy_parameter_ids
    if optimizer_parameter_ids != trainable_parameter_ids:
        missing = sorted(
            trainable_parameter_names[parameter_id]
            for parameter_id in trainable_parameter_ids - optimizer_parameter_ids
        )
        unexpected = sorted(
            model_parameter_names.get(parameter_id, f"<id={parameter_id}>")
            for parameter_id in optimizer_parameter_ids - trainable_parameter_ids
        )
        raise RuntimeError(
            "Optimizer parameter partition does not match trainable model "
            f"parameters; missing={missing}, unexpected={unexpected}"
        )
    entropy_parameter_signature = [
        (name, tuple(parameter.shape))
        for name, parameter in model.codec.named_parameters()
        if name.endswith(".quantiles") and parameter.requires_grad
    ]
    optimizer_parameters = (
        model.get_self_flow_optimizer_groups(train_config["optimizer"])
        if self_flow_enabled
        else main_parameters
    )
    opt = torch.optim.AdamW(
        optimizer_parameters,
        lr=train_config["optimizer"]["lr"],
        weight_decay=0,
        betas=(0.9, train_config["optimizer"]["beta2"]),
    )
    opt_disc = torch.optim.AdamW(loss_module.discriminator.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=train_config['optimizer']['beta_vae'])
    opt_entropy = (
        optim.AdamW(
            entropy_parameters,
            lr=train_config['optimizer']['aux_lr'],
            weight_decay=0,
        )
        if entropy_parameters
        else None
    )
    if rank == 0:
        if entropy_update_mode == OFFICIAL_ENTROPY_UPDATE:
            logger.info(
                "Entropy updates: official DiT-IC NoGAN order "
                "(global model clip, post-main auxiliary step)"
            )
        else:
            logger.info(
                "Entropy updates: historical pre-fix order "
                "(pre-main accumulated auxiliary backward, global model clip, "
                "then main and auxiliary steps)"
            )
    
    milestones = train_config["optimizer"]["step_lr"]
    gammas = train_config["optimizer"].get(
        "step_lr_gammas", [0.5, 0.2, 0.1]
    )
    if len(milestones) != len(gammas):
        raise ValueError(
            "optimizer.step_lr and optimizer.step_lr_gammas must have "
            "the same length"
        )
    if any(float(gamma) <= 0.0 or float(gamma) > 1.0 for gamma in gammas):
        raise ValueError("optimizer.step_lr_gammas values must be in (0, 1]")

    def lr_lambda(epoch):
        factor = 1.0
        for m, g in zip(milestones, gammas):
            if epoch >= m:
                factor *= g
        return factor

    # scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)
    scheduler_disc = LambdaLR(opt_disc, lr_lambda=lr_lambda)
    
    # scheduler = MultiStepLR(opt, milestones=train_config['optimizer']['step_lr'], gamma=0.5)
    # scheduler_disc = MultiStepLR(opt_disc, milestones=train_config['optimizer']['step_lr'], gamma=0.5)
    
    model = DDP(model, device_ids=[local_rank], static_graph=True)
    discriminator = (
        DDP(loss_module.discriminator, device_ids=[local_rank], broadcast_buffers=False)
        if world_size > 1
        else loss_module.discriminator
    )
    # Setup data
    train_transforms = transforms.Compose([
            ResizeIfSmall(train_config['data']['path_size']),
            transforms.RandomCrop(train_config['data']['path_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    dataset = ImageFolder(train_config['data']['data_path'], train_config['data']['im_exts'], transform=train_transforms)
    
    # batch_size_per_gpu = int(train_config['train']['global_batch_size'] / torch.cuda.device_count())
    per_gpu_batch_size = train_config['train'].get('per_gpu_batch_size')
    if per_gpu_batch_size is not None:
        batch_size_per_gpu = int(per_gpu_batch_size)
        if batch_size_per_gpu < 1:
            raise ValueError("train.per_gpu_batch_size must be positive")
        global_batch_size = batch_size_per_gpu * world_size
        train_config['train']['global_batch_size'] = global_batch_size
    else:
        global_batch_size = train_config['train']['global_batch_size']
        if global_batch_size % world_size != 0:
            raise ValueError(
                "train.global_batch_size must be divisible by the DDP world size"
            )
        batch_size_per_gpu = global_batch_size // world_size
    if rate_anchored_enabled:
        anchor_count_float = legal_anchor_ratio * batch_size_per_gpu
        local_anchor_count = int(round(anchor_count_float))
        if abs(anchor_count_float - local_anchor_count) > 1e-9:
            raise ValueError(
                "legal_anchor_ratio must produce an integer per-GPU anchor count"
            )
        if not 0 < local_anchor_count < batch_size_per_gpu:
            raise ValueError(
                "each GPU batch must contain legal and asymmetric samples"
            )
        if rank == 0:
            logger.info(
                "Rate-Anchored partition per GPU: "
                f"{local_anchor_count} legal + "
                f"{batch_size_per_gpu - local_anchor_count} asymmetric"
            )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=my_worker_init_fn,
        sampler=sampler,
    )

    if rank == 0: 
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {train_config['train']['global_batch_size']} global batch size")
    
    if 'valid_path' in train_config['data']:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.CenterCrop(train_config['data']['path_size']),
        ])
        
        valid_dataset = ImageFolder(train_config['data']['valid_path'], train_config['data']['im_exts'], transform=val_transforms)
    
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False
        )
        if rank == 0: 
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")
    
    # Prepare models for training:
    if rank==0:
        update_ema(
            ema,
            model.module.get_state_dicts_for_save(),
            decay=deployment_ema_decay,
            exact_parts=ema_exact_parts,
        )
    model.train()
    model.module.assert_frozen_codec_contract()
    model.module.assert_reconstruction_training_contract()
    loss_module.train()

    train_config['train']['resume'] if 'resume' in train_config['train'] else False
    best_loss = float("inf")

    if train_config['train']['resume']:
        # check if the checkpoint exists
        resume_checkpoint = train_config['train'].get('resume_checkpoint')
        checkpoint_files = (
            [resume_checkpoint]
            if resume_checkpoint is not None
            else glob(f"{checkpoint_dir}/*.pt")
        )
        if checkpoint_files:
            if resume_checkpoint is not None:
                latest_checkpoint = resume_checkpoint
                if not os.path.isfile(latest_checkpoint):
                    raise FileNotFoundError(
                        f"Configured resume checkpoint does not exist: {latest_checkpoint}"
                    )
            else:
                checkpoint_files.sort(key=os.path.getmtime)
                latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            
            model.module.load_ckpt_dict(latest_checkpoint, use_ema=False)
            model.train()
            loss_module.train()
            if self_flow_enabled:
                if checkpoint.get("stage") != self_flow_checkpoint_contract["stage"]:
                    raise ValueError(
                        "Self-Flow resume requires a Self-Flow training checkpoint"
                    )
                if rate_anchored_enabled and not np.isclose(
                    float(checkpoint.get("legal_anchor_ratio", -1.0)),
                    legal_anchor_ratio,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Rate-Anchored resume checkpoint ratio does not match config"
                    )
                if self_flow_teacher_enabled:
                    teacher_state = checkpoint.get("self_flow_teacher")
                    if teacher_state is None:
                        raise KeyError(
                            "Self-Flow resume checkpoint lacks teacher state"
                        )
                    self_flow_teacher.load_ema_state_dict(teacher_state)
                resolved_representation_weight = checkpoint.get(
                    "resolved_representation_weight",
                    resolved_representation_weight,
                )
                teacher_resynced_after_warmup = bool(
                    checkpoint.get("teacher_resynced_after_warmup", False)
                )
                saved_pair_counts = checkpoint.get("self_flow_pair_counts")
                if saved_pair_counts is None:
                    raise KeyError(
                        "Self-Flow resume checkpoint lacks pair-count state"
                    )
                self_flow_pair_counts.copy_(saved_pair_counts.cpu())
                if rate_anchored_enabled:
                    saved_anchor_counts = checkpoint.get(
                        "self_flow_anchor_quality_counts"
                    )
                    saved_partition_counts = checkpoint.get(
                        "self_flow_partition_counts"
                    )
                    if saved_anchor_counts is None or saved_partition_counts is None:
                        raise KeyError(
                            "Rate-Anchored resume checkpoint lacks partition state"
                        )
                    self_flow_anchor_quality_counts.copy_(
                        saved_anchor_counts.cpu()
                    )
                    self_flow_partition_counts.copy_(
                        saved_partition_counts.cpu()
                    )

            restored_entropy_loss = float(
                model.module.codec.entropy_bottleneck_loss().detach().item()
            )
            if not np.isfinite(restored_entropy_loss):
                raise RuntimeError(
                    "Checkpoint restored a non-finite entropy-bottleneck loss: "
                    f"{restored_entropy_loss}"
                )
            saved_entropy_loss = checkpoint.get("entropy_bottleneck_loss")
            if saved_entropy_loss is not None and not np.isclose(
                restored_entropy_loss,
                float(saved_entropy_loss),
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Entropy-bottleneck state changed while restoring the checkpoint: "
                    f"saved={float(saved_entropy_loss):.6f}, "
                    f"restored={restored_entropy_loss:.6f}"
                )
            
            opt.load_state_dict(checkpoint['opt'])
            if "loss" in checkpoint:
                loss_state = checkpoint["loss"]
                loss_module.discriminator.load_state_dict(
                    loss_state["discriminator"]
                )
                if "logvar" in loss_state:
                    loss_module.logvar.data.copy_(loss_state["logvar"])
                opt_disc.load_state_dict(checkpoint['opt_disc'])
            elif rank == 0:
                logger.warning(
                    "Legacy checkpoint has no discriminator weights; "
                    "starting the discriminator from its fresh initialization."
                )
            saved_entropy_update_mode = checkpoint.get(
                "entropy_update_mode", OFFICIAL_ENTROPY_UPDATE
            )
            if saved_entropy_update_mode != entropy_update_mode:
                raise ValueError(
                    "resume checkpoint entropy update mode does not match config: "
                    f"checkpoint={saved_entropy_update_mode!r}, "
                    f"config={entropy_update_mode!r}"
                )
            entropy_state = checkpoint.get("opt_entropy", checkpoint.get("opt_aux"))
            if opt_entropy is not None and entropy_state is not None:
                saved_entropy_version = checkpoint.get("entropy_optimizer_version")
                saved_entropy_signature = checkpoint.get(
                    "entropy_parameter_signature"
                )
                # Legacy moments were accumulated while entropy gradients
                # participated in model-wide clipping.
                if (
                    saved_entropy_version == entropy_optimizer_version
                    and saved_entropy_signature == entropy_parameter_signature
                ):
                    opt_entropy.load_state_dict(entropy_state)
                elif rank == 0:
                    logger.warning(
                        "Resetting legacy/incompatible entropy optimizer moments "
                        "while preserving the checkpoint's entropy-model weights "
                        "and quantiles."
                    )

            ema = checkpoint['ema']
            ema = move_to_device(checkpoint['ema'], device)
            checkpoint_stem = os.path.splitext(os.path.basename(latest_checkpoint))[0]
            train_steps = checkpoint.get(
                "train_steps",
                int(checkpoint_stem) if checkpoint_stem.isdigit() else 0,
            )
            best_loss = float(checkpoint.get("best_loss", best_loss))
            if multirate_plain:
                saved_quality_state = checkpoint.get("quality_generator_state")
                if saved_quality_state is None:
                    raise KeyError(
                        "multi-rate resume checkpoint lacks quality_generator_state"
                    )
                quality_generator.set_state(saved_quality_state.cpu())
                if not self_flow_enabled:
                    saved_quality_counts = checkpoint.get("quality_counts")
                    if saved_quality_counts is None:
                        raise KeyError("multi-rate resume checkpoint lacks quality_counts")
                    quality_counts.copy_(saved_quality_counts.cpu())
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
                if "loss" in checkpoint and "scheduler_disc" in checkpoint:
                    scheduler_disc.load_state_dict(checkpoint["scheduler_disc"])
            else:
                scheduler = MultiStepLR(opt, milestones=train_config['optimizer']['step_lr'], gamma=0.5, last_epoch=train_steps)
                scheduler_disc = MultiStepLR(opt_disc, milestones=train_config['optimizer']['step_lr'], gamma=0.5, last_epoch=train_steps)
    
            if rank == 0: 
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
                logger.info(
                    "Restored entropy-bottleneck loss before the first update: "
                    f"{restored_entropy_loss:.6f}"
                )
        else:
            train_steps = 0
            if rank == 0: 
                logger.info("No checkpoint found. Starting training from scratch.")

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0

    def make_checkpoint():
        checkpoint = {
            "schema_version": (
                self_flow_checkpoint_contract["schema_version"]
                if self_flow_enabled
                else (MULTIRATE_CHECKPOINT_SCHEMA_VERSION if multirate_plain else 2)
            ),
            "stage": (
                self_flow_checkpoint_contract["stage"]
                if self_flow_enabled
                else ("multirate_reconstruction" if multirate_plain else "baseline")
            ),
            "model": model.module.get_state_dicts_for_save(),
            "ema": ema,
            "loss": {
                "discriminator": loss_module.discriminator.state_dict(),
                "logvar": loss_module.logvar.detach().cpu(),
            },
            "opt": opt.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_disc": scheduler_disc.state_dict(),
            "train_steps": train_steps,
            "best_loss": best_loss,
            "config": train_config,
            "entropy_bottleneck_loss": float(
                model.module.codec.entropy_bottleneck_loss().detach().item()
            ),
            "entropy_update_mode": entropy_update_mode,
            "entropy_optimizer_version": entropy_optimizer_version,
            "entropy_parameter_signature": entropy_parameter_signature,
            "dists_input_contract": DISTS_INPUT_CONTRACT,
        }
        if opt_entropy is not None:
            entropy_state = opt_entropy.state_dict()
            checkpoint["opt_entropy"] = entropy_state
            checkpoint["opt_aux"] = entropy_state
        if multirate_plain:
            checkpoint["quality_generator_state"] = quality_generator.get_state()
            if not self_flow_enabled:
                checkpoint["quality_counts"] = quality_counts.clone()
        if multirate_plain:
            codec_state = checkpoint["model"]["state_dict_codec"]
            codec_sha256 = state_dict_sha256(codec_state)
            ema_codec_sha256 = state_dict_sha256(
                checkpoint["ema"]["state_dict_codec"]
            )
            if codec_sha256 != ema_codec_sha256:
                if model.module.freeze_codec:
                    raise RuntimeError(
                        "frozen codec drifted between live and EMA checkpoint state"
                    )
            checkpoint["codec_sha256"] = codec_sha256
            checkpoint["ema_codec_sha256"] = ema_codec_sha256
            checkpoint["source_checkpoint_sha256"] = source_checkpoint_sha256
            checkpoint["git_commit"] = current_git_commit()
            checkpoint["bitstream_format_version"] = 1
            checkpoint["variable_rate_mechanism"] = VARIABLE_RATE_MECHANISM
            checkpoint["end_to_end"] = not model.module.freeze_codec
            checkpoint["quality_rate_weights"] = quality_rate_weights
            checkpoint["decoder_relay_mode"] = model.module.decoder_relay_mode
            checkpoint["decoder_relay_contract"] = (
                model.module.decoder_relay.contract()
                if model.module.decoder_relay is not None
                else None
            )
            checkpoint["trainable_parameter_names"] = sorted(
                name
                for name, parameter in model.module.named_parameters()
                if parameter.requires_grad
            )
        if self_flow_enabled:
            checkpoint["self_flow_mechanism"] = self_flow_checkpoint_contract["mechanism"]
            checkpoint["self_flow_teacher"] = (
                self_flow_teacher.get_ema_state_dict()
                if self_flow_teacher is not None
                else None
            )
            checkpoint["resolved_representation_weight"] = (
                resolved_representation_weight
            )
            checkpoint["teacher_resynced_after_warmup"] = (
                teacher_resynced_after_warmup
            )
            checkpoint["self_flow_pair_counts"] = (
                self_flow_pair_counts.clone()
            )
            if rate_anchored_enabled:
                checkpoint["self_flow_anchor_quality_counts"] = (
                    self_flow_anchor_quality_counts.clone()
                )
                checkpoint["self_flow_partition_counts"] = (
                    self_flow_partition_counts.clone()
                )
                checkpoint["legal_anchor_ratio"] = legal_anchor_ratio
            checkpoint["teacher_ema_decay"] = teacher_ema_decay
            checkpoint["deployment_ema_decay"] = deployment_ema_decay
            checkpoint["rng_contract"] = "stateless_step_rank_microstep_v1"
        return checkpoint

    log_steps = 0
    running_loss = 0
    running_entropy_loss = 0
    running_dis_loss = 0

    start_time = time()
    
    accumulation_steps = train_config['train'].get('accumulation_steps', 1)
    if accumulation_steps < 1:
        raise ValueError("train.accumulation_steps must be at least 1")
    accumulate_disc = train_config['train'].get('accumulate_disc', True)
    accum_count = 0
    disc_start = train_config['model']['params']['loss_params']['disc_start']
    opt.zero_grad(set_to_none=True)
    opt_disc.zero_grad(set_to_none=True)
    if opt_entropy is not None:
        opt_entropy.zero_grad(set_to_none=True)
    quality_index = None

    while True:
        # sampler.set_epoch(train_steps)
        epoch = (train_steps * accumulation_steps) // max(1, len(loader))  # ★DDP: ensure shuffle
        sampler.set_epoch(epoch)


        for x in loader:
            x = x.to(device)
            will_step = accum_count + 1 == accumulation_steps
            disc_active = train_steps >= disc_start
            if self_flow_enabled:
                warmup_steps = int(
                    self_flow_config.get("uniform_warmup_steps", 2000)
                )
                if (
                    self_flow_teacher is not None
                    and train_steps >= warmup_steps
                    and not teacher_resynced_after_warmup
                ):
                    self_flow_teacher.update_from_student(
                        model.module.DiT,
                        model.module.text_proj,
                        model.module.quality_conditioner,
                        decay=0.0,
                    )
                    teacher_resynced_after_warmup = True
                    if rank == 0:
                        logger.info(
                            f"Resynced EMA Teacher from Student at step {train_steps}"
                        )
                sample_seed = deterministic_self_flow_seed(
                    train_config["train"]["global_seed"],
                    0 if train_steps < warmup_steps else rank,
                    train_steps,
                    accum_count,
                )
                if train_steps < warmup_steps:
                    if accum_count == 0:
                        quality_index = sample_shared_quality_index(
                            num_quality_levels,
                            device,
                            generator=quality_generator,
                        )
                    quality_a = quality_index.expand(x.shape[0]).cpu()
                    quality_b = quality_a.clone()
                    asymmetric_samples = torch.full(
                        (x.shape[0],),
                        not rate_anchored_enabled,
                        dtype=torch.bool,
                    )
                else:
                    pair_generator = torch.Generator(device="cpu")
                    pair_generator.manual_seed(sample_seed)
                    if rate_anchored_enabled:
                        (
                            quality_a,
                            quality_b,
                            asymmetric_samples,
                        ) = sample_rate_anchored_pairs(
                            x.shape[0],
                            num_quality_levels,
                            legal_anchor_ratio,
                            pair_generator,
                        )
                        anchor_samples = ~asymmetric_samples
                        anchor_bins = torch.bincount(
                            quality_a[anchor_samples],
                            minlength=num_quality_levels,
                        )
                        self_flow_anchor_quality_counts.add_(anchor_bins)
                        self_flow_partition_counts[0] += anchor_samples.sum()
                        self_flow_partition_counts[1] += asymmetric_samples.sum()
                    else:
                        quality_a, quality_b = sample_iid_uniform_pairs(
                            x.shape[0],
                            num_quality_levels,
                            pair_generator,
                        )
                        asymmetric_samples = torch.ones(
                            x.shape[0], dtype=torch.bool
                        )
                asymmetric_quality_a = quality_a[asymmetric_samples]
                asymmetric_quality_b = quality_b[asymmetric_samples]
                pair_bins = torch.bincount(
                    asymmetric_quality_a * num_quality_levels
                    + asymmetric_quality_b,
                    minlength=num_quality_levels * num_quality_levels,
                ).reshape(num_quality_levels, num_quality_levels)
                self_flow_pair_counts.add_(pair_bins)
                quality_a = quality_a.to(device)
                quality_b = quality_b.to(device)
                asymmetric_samples = asymmetric_samples.to(device)
                schedule_values = self_flow_schedule(
                    train_steps, self_flow_config
                )
                output_image, self_flow_output = model(
                    x,
                    self_flow_batch={
                        "quality_a": quality_a,
                        "quality_b": quality_b,
                        "mask_ratio": schedule_values["mask_ratio"],
                        "mask_seed": sample_seed + 53_000_019,
                        "conditioning_scale": schedule_values[
                            "conditioning_scale"
                        ],
                        "student_input": self_flow_config["student_input"],
                        "teacher_quality": self_flow_config["teacher_quality"],
                        "use_second_view": bool(
                            self_flow_config.get("use_second_view", True)
                        ),
                        "asymmetric_samples": asymmetric_samples,
                    },
                )
                rate = self_flow_output["likelihoods_a"]
                loss_module.rate_weight = 0.0
                base_loss, log_dict = loss_module(
                    x,
                    output_image,
                    rate,
                    -1,
                    train_steps,
                    last_layer=model.module.get_last_layer(),
                    split="train",
                )
                representation_samples = self_flow_output[
                    "asymmetric_samples"
                ]
                representation_sample_count = int(
                    representation_samples.sum().item()
                )
                if (
                    self_flow_representation_enabled
                    and representation_sample_count > 0
                ):
                    with torch.no_grad():
                        teacher_feature = self_flow_teacher(
                            self_flow_output["teacher_latent"][
                                representation_samples
                            ],
                            model.module.fixed_text_embed,
                            model.module.self_flow_timestep(
                                representation_sample_count, x.device
                            ),
                            self_flow_output["teacher_quality"][
                                representation_samples
                            ],
                            schedule_values["conditioning_scale"],
                        )
                    representation_loss = all_token_representation_loss(
                        self_flow_output["student_feature"][
                            representation_samples
                        ],
                        teacher_feature,
                    )
                    if (
                        resolved_representation_weight is None
                        and train_steps >= warmup_steps
                    ):
                        (
                            resolved_representation_weight,
                            base_gradient_norm,
                            representation_gradient_norm,
                        ) = calibrate_representation_weight(
                            base_loss,
                            representation_loss,
                            [model.module.__dict__["_self_flow_probe_feature"]],
                            target_weight=float(
                                self_flow_config.get("rep_weight_target", 0.1)
                            ),
                            maximum_gradient_ratio=float(
                                self_flow_config.get(
                                    "rep_gradient_ratio_max", 0.2
                                )
                            ),
                            minimum_weight=float(
                                self_flow_config.get("rep_weight_min", 0.01)
                            ),
                        )
                        if rank == 0:
                            logger.info(
                                "Resolved representation weight: "
                                f"gamma={resolved_representation_weight:.6g}, "
                                f"base_grad={base_gradient_norm:.6g}, "
                                f"rep_grad={representation_gradient_norm:.6g}"
                            )
                    gamma = (
                        float(resolved_representation_weight)
                        if resolved_representation_weight is not None
                        else 0.0
                    ) * schedule_values["representation_scale"]
                    loss = base_loss + gamma * representation_loss
                else:
                    representation_loss = torch.zeros((), device=x.device)
                    gamma = 0.0
                    loss = (
                        zero_connect_feature(
                            base_loss, self_flow_output["student_feature"]
                        )
                        if rate_anchored_enabled
                        else base_loss
                    )
                legal_bpp_a = likelihood_bpp(
                    self_flow_output["likelihoods_a"], x
                )
                legal_bpp_b = likelihood_bpp(
                    self_flow_output["likelihoods_b"], x
                )
                token_mask = self_flow_output["mask"]
                if representation_sample_count > 0:
                    asymmetric_mask_ratio = token_mask[
                        representation_samples
                    ].float().mean()
                else:
                    asymmetric_mask_ratio = torch.zeros((), device=x.device)
                mask_has_a = (~token_mask).flatten(1).any(dim=1)
                mask_has_b = token_mask.flatten(1).any(dim=1)
                genuinely_mixed = (
                    representation_samples
                    & quality_a.ne(quality_b)
                    & mask_has_a
                    & mask_has_b
                )
                legal_student_inputs = ~genuinely_mixed
                log_dict.update({
                    "train/total_loss": loss.detach(),
                    "train/base_loss": base_loss.detach(),
                    "train/representation_loss": representation_loss.detach(),
                    "train/representation_weight": gamma,
                    "train/representation_contribution": (
                        gamma * representation_loss.detach()
                    ),
                    "train/representation_sample_count": (
                        representation_sample_count
                    ),
                    "train/mask_ratio_scheduled": schedule_values["mask_ratio"],
                    "train/mask_ratio_actual": (
                        asymmetric_mask_ratio.detach()
                    ),
                    "train/mixed_token_fraction_all": (
                        token_mask.float().mean().detach()
                    ),
                    "train/legal_anchor_fraction": (
                        (~representation_samples).float().mean().detach()
                    ),
                    "train/asymmetric_fraction": (
                        representation_samples.float().mean().detach()
                    ),
                    "train/legal_student_input_fraction_actual": (
                        legal_student_inputs.float().mean().detach()
                    ),
                    "train/q_conditioning_scale": schedule_values["conditioning_scale"],
                    "train/legal_bpp_a": legal_bpp_a.detach(),
                    "train/legal_bpp_b": legal_bpp_b.detach(),
                })
            else:
                if multirate_plain and accum_count == 0:
                    quality_index = sample_shared_quality_index(
                        num_quality_levels,
                        device,
                        generator=quality_generator,
                    )
                if quality_rate_weights is not None:
                    loss_module.rate_weight = quality_rate_weights[
                        int(quality_index.item())
                    ]
                else:
                    loss_module.rate_weight = train_config["train"]["rate_weight"]

                if multirate_plain:
                    output_image, rate = model(x, quality_index)
                else:
                    output_image, rate = model(x)

                if not disc_active:
                    loss, log_dict = loss_module(
                        x,
                        output_image,
                        rate,
                        -1,
                        train_steps,
                        last_layer=model.module.get_last_layer(),
                        split="train",
                    )
                else:
                    loss_module.discriminator.requires_grad_(False)
                    loss, log_dict = loss_module(
                        x,
                        output_image,
                        rate,
                        0,
                        train_steps,
                        last_layer=model.module.get_last_layer(),
                        split="train",
                        discriminator=loss_module.discriminator,
                    )
                    loss_module.discriminator.requires_grad_(True)
            loss = loss / accumulation_steps

            # ---- backward pass (accumulate gradients) ----
            loss.backward()

            if entropy_update_mode == HISTORICAL_ENTROPY_UPDATE:
                if opt_entropy is not None:
                    entropy_loss = (
                        model.module.codec.entropy_bottleneck_loss()
                        / accumulation_steps
                    )
                    if not torch.isfinite(entropy_loss).item():
                        raise FloatingPointError(
                            "Entropy-bottleneck loss became non-finite before "
                            f"the historical auxiliary update at step {train_steps}"
                        )
                    entropy_loss.backward()
                else:
                    entropy_loss = torch.tensor(0.0, device=x.device)
                running_entropy_loss += entropy_loss.item() * accumulation_steps
            if self_flow_enabled:
                model.module.__dict__["_self_flow_probe_feature"] = None

            # With accumulate_disc=False, use only the boundary micro-batch so
            # the discriminator still performs one update per generator step.
            run_discriminator = disc_active and (accumulate_disc or will_step)
            if run_discriminator:
                if not accumulate_disc:
                    opt_disc.zero_grad(set_to_none=True)
                discloss, log_dict_disc = loss_module(
                    x,
                    output_image,
                    rate,
                    1,
                    train_steps,
                    last_layer=model.module.get_last_layer(),
                    split="train",
                    discriminator=discriminator,
                )
                if accumulate_disc:
                    discloss = discloss / accumulation_steps
                discloss.backward()
            else:
                discloss = torch.tensor(0.0, device=x.device)

            accum_count += 1

            # Track running losses at original scale
            running_loss += loss.item() * accumulation_steps
            if run_discriminator:
                running_dis_loss += discloss.item() * accumulation_steps
            log_steps += 1

            # ---- step optimizers after accumulation ----
            if accum_count == accumulation_steps:
                # Generator step
                # Keep the previous auxiliary gradient resident until the
                # official post-main-step opt_entropy.zero_grad() below.
                if 'max_grad_norm' in train_config['optimizer']:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config['optimizer']['max_grad_norm'],
                    )
                opt.step()
                scheduler.step()
                if self_flow_teacher is not None:
                    self_flow_teacher.update_from_student(
                        model.module.DiT,
                        model.module.text_proj,
                        model.module.quality_conditioner,
                        decay=teacher_ema_decay,
                    )

                if entropy_update_mode == HISTORICAL_ENTROPY_UPDATE:
                    if opt_entropy is not None:
                        opt_entropy.step()
                else:
                    # Official DiT-IC NoGAN ordering: recompute after the main step.
                    if opt_entropy is not None:
                        entropy_loss = (
                            model.module.codec.entropy_bottleneck_loss()
                        )
                        if not torch.isfinite(entropy_loss).item():
                            raise FloatingPointError(
                                "Entropy-bottleneck loss became non-finite before "
                                f"the auxiliary update at step {train_steps}"
                            )
                        opt_entropy.zero_grad(set_to_none=True)
                        entropy_loss.backward()
                        opt_entropy.step()
                    else:
                        entropy_loss = torch.tensor(0.0, device=x.device)
                    running_entropy_loss += (
                        entropy_loss.item() * accumulation_steps
                    )

                # Discriminator step
                if run_discriminator:
                    if 'max_grad_norm' in train_config['optimizer']:
                        torch.nn.utils.clip_grad_norm_(loss_module.discriminator.parameters(), train_config['optimizer']['max_grad_norm'])
                    opt_disc.step()
                    scheduler_disc.step()

                # EMA update
                if rank == 0:
                    update_ema(
                        ema,
                        model.module.get_state_dicts_for_save(),
                        decay=deployment_ema_decay,
                        exact_parts=ema_exact_parts,
                    )

                # Zero all gradients for next accumulation cycle
                opt.zero_grad(set_to_none=True)
                if (
                    entropy_update_mode == HISTORICAL_ENTROPY_UPDATE
                    and opt_entropy is not None
                ):
                    opt_entropy.zero_grad(set_to_none=True)
                opt_disc.zero_grad(set_to_none=True)

                train_steps += 1
                if multirate_plain and not self_flow_enabled:
                    quality_counts[int(quality_index.item())] += 1
                    quality_index = None
                elif self_flow_enabled and train_steps <= warmup_steps:
                    quality_index = None
                accum_count = 0

                # ---- logging ----
                if train_steps > 0 and train_steps % train_config['train']['log_every'] == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    # d loss
                    avg_dis_loss = torch.tensor(running_dis_loss / log_steps, device=device)
                    dist.all_reduce(avg_dis_loss, op=dist.ReduceOp.SUM)
                    avg_dis_loss = avg_dis_loss.item() / dist.get_world_size()
                    # entropy-bottleneck loss
                    avg_entropy_loss = torch.tensor(running_entropy_loss / log_steps, device=device)
                    dist.all_reduce(avg_entropy_loss, op=dist.ReduceOp.SUM)
                    avg_entropy_loss = avg_entropy_loss.item() / dist.get_world_size()

                    lr = opt.param_groups[0]['lr']
                    if rank == 0:
                        logger.info(f"(step={train_steps:07d}, lr = {lr:.6e}) Train Loss: {avg_loss:.4f}, Dis: {avg_dis_loss:.4f}, Aux: {avg_entropy_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                        log = {k: float(v) for k, v in log_dict.items()}
                        logger.info(json.dumps(log, indent=2))
                        if train_steps > disc_start:
                            dis_log = {k: float(v) for k, v in log_dict_disc.items()}
                            logger.info(json.dumps(dis_log, indent=2))
                        writer.add_scalar('Loss/train', avg_loss, train_steps)
                        writer.add_scalar('Loss/entropy', avg_entropy_loss, train_steps)
                        writer.add_scalar('Loss/dis', avg_dis_loss, train_steps)
                        writer.add_scalar('Params/lr', lr, train_steps)
                        if self_flow_enabled:
                            reduced_pair_counts = self_flow_pair_counts
                            logger.info(
                                "Self-Flow q_a/q_b pair counts: "
                                + json.dumps(reduced_pair_counts.cpu().tolist())
                            )
                            if rate_anchored_enabled:
                                logger.info(
                                    "Rate-Anchored legal q counts: "
                                    + json.dumps(
                                        self_flow_anchor_quality_counts.tolist()
                                    )
                                )
                                logger.info(
                                    "Rate-Anchored [legal, asymmetric] counts: "
                                    + json.dumps(
                                        self_flow_partition_counts.tolist()
                                    )
                                )
                                writer.add_scalar(
                                    "RateAnchor/legal_samples",
                                    int(self_flow_partition_counts[0]),
                                    train_steps,
                                )
                                writer.add_scalar(
                                    "RateAnchor/asymmetric_samples",
                                    int(self_flow_partition_counts[1]),
                                    train_steps,
                                )
                            for quality_a_index in range(num_quality_levels):
                                for quality_b_index in range(num_quality_levels):
                                    writer.add_scalar(
                                        f"Quality/q{quality_a_index}_q{quality_b_index}",
                                        int(reduced_pair_counts[quality_a_index, quality_b_index]),
                                        train_steps,
                                    )
                        elif multirate_plain:
                            logger.info(
                                "Quality step counts: "
                                + json.dumps(quality_counts.tolist())
                            )
                            for quality, count in enumerate(quality_counts.tolist()):
                                writer.add_scalar(
                                    f"Quality/q{quality}_steps",
                                    count,
                                    train_steps,
                                )
                        # Log all loss components to tensorboard
                        for k, v in log_dict.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f'Train/{k}', v, train_steps)
                            elif isinstance(v, torch.Tensor):
                                writer.add_scalar(f'Train/{k}', v.item(), train_steps)
                        if train_steps > disc_start:
                            for k, v in log_dict_disc.items():
                                if isinstance(v, (int, float)):
                                    writer.add_scalar(f'Train/{k}', v, train_steps)
                                elif isinstance(v, torch.Tensor):
                                    writer.add_scalar(f'Train/{k}', v.item(), train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    running_entropy_loss = 0
                    running_dis_loss = 0
                    log_steps = 0
                    start_time = time()

                # ---- evaluation ----
                eval_every = train_config['train'].get('eval_every', 2000)
                if train_steps > 0 and train_steps % eval_every == 0 and 'valid_path' in train_config['data']:
                    if rank == 0:
                        logger.info(f"Start evaluating at step {train_steps}")

                    if multirate_plain:
                        test_loss = None
                        if rank == 0:
                            evaluation_conditioning_scale = (
                                self_flow_schedule(
                                    train_steps, self_flow_config
                                )["conditioning_scale"]
                                if self_flow_enabled
                                else 1.0
                            )
                            logger.info(
                                "Evaluation q-conditioning scale: "
                                f"{evaluation_conditioning_scale:.6f}"
                            )
                            test_loss = evaluate_multirate_ema(
                                model.module,
                                ema,
                                loss_module,
                                valid_loader,
                                device,
                                train_steps,
                                logger,
                                writer,
                                num_quality_levels,
                                quality_rate_weights,
                                conditioning_scale=evaluation_conditioning_scale,
                            )
                    else:
                        test_loss = evaluate(
                            model,
                            loss_module,
                            valid_loader,
                            device,
                            train_steps,
                            logger if rank==0 else None,
                            writer if rank==0 else None,
                        )

                    if rank == 0:
                        best_loss, is_best = select_post_burnin_best(
                            test_loss,
                            best_loss,
                            train_steps,
                            train_config["train"]["max_steps"],
                        )
                        if is_best:
                            checkpoint = make_checkpoint()
                            checkpoint_path = f"{checkpoint_dir}/best_ckpt.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved best checkpoint to {checkpoint_path} at step {train_steps}")
                    dist.barrier()
                    model.train()
                    model.module.assert_reconstruction_training_contract()
                    loss_module.train()

                # ---- save checkpoint ----
                if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = make_checkpoint()
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

                    model.train()
                if train_steps >= train_config['train']['max_steps']:
                    break
        if train_steps >= train_config['train']['max_steps']:
            break

    if rank == 0: 
        logger.info("Done!")


@torch.no_grad()
def evaluate_multirate_ema(
    model,
    ema,
    loss_module,
    test_loader,
    device,
    train_steps,
    logger,
    writer,
    num_quality_levels,
    quality_rate_weights=None,
    conditioning_scale=1.0,
):
    """Evaluate the same EMA weights that downstream P0 loading will use."""
    model.assert_reconstruction_training_contract()
    swap_saved_state_dicts(model, ema)
    try:
        return evaluate(
            model,
            loss_module,
            test_loader,
            device,
            train_steps,
            logger,
            writer,
            num_quality_levels=num_quality_levels,
            quality_rate_weights=quality_rate_weights,
            conditioning_scale=conditioning_scale,
        )
    finally:
        swap_saved_state_dicts(model, ema)


@torch.no_grad()
def evaluate(
    model,
    loss_module,
    test_loader,
    device,
    train_steps,
    logger=None,
    writer=None,
    num_quality_levels=None,
    quality_rate_weights=None,
    conditioning_scale=1.0,
):
    model.eval()
    loss_module.eval()
    loss = AverageMeter()
    
    mse_loss = AverageMeter()
    psnr_loss = AverageMeter()
    lpips_loss = AverageMeter()
    bpp_loss = AverageMeter()
    entropy_loss = AverageMeter()
    all_loss = AverageMeter()
    
    qualities = (
        range(num_quality_levels) if num_quality_levels is not None else [None]
    )
    per_quality = {}
    original_rate_weight = loss_module.rate_weight
    for quality_index in qualities:
        if quality_rate_weights is not None:
            loss_module.rate_weight = quality_rate_weights[quality_index]
        quality_loss = AverageMeter()
        quality_psnr = AverageMeter()
        quality_bpp = AverageMeter()
        for x in test_loader:
            x = x.to(device)

            if quality_index is None:
                output_image, rate = model(x)
            else:
                output_image, rate = model(
                    x,
                    quality_index,
                    conditioning_scale=conditioning_scale,
                )
            loss, rec_loss, psnr, rate_loss, p_loss = loss_module(
                x,
                output_image,
                rate,
                -1,
                -1,
                split="eval",
            )

            batch_size = x.shape[0]
            all_loss.update(loss, batch_size)
            mse_loss.update(rec_loss, batch_size)
            psnr_loss.update(psnr, batch_size)
            lpips_loss.update(p_loss, batch_size)
            bpp_loss.update(rate_loss, batch_size)
            entropy_loss.update(
                (model.module.codec if hasattr(model, 'module') else model.codec)
                .entropy_bottleneck_loss(),
                batch_size,
            )
            quality_loss.update(loss, batch_size)
            quality_psnr.update(psnr, batch_size)
            quality_bpp.update(rate_loss, batch_size)
        if quality_index is not None:
            per_quality[quality_index] = {
                "loss": float(quality_loss.avg),
                "psnr": float(quality_psnr.avg),
                "likelihood_bpp": float(quality_bpp.avg),
                "rate_weight": float(loss_module.rate_weight),
            }

    loss_module.rate_weight = original_rate_weight

    if logger is not None:
        logger.info(
            f"Test Average losses: "
            f"Loss: {all_loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"PSNR: {psnr_loss.avg:.2f} | "
            f"LPIPS: {lpips_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Entropy loss: {entropy_loss.avg:.2f}"
        )
        if per_quality:
            logger.info("Per-quality validation: " + json.dumps(per_quality))
    if writer is not None:
        writer.add_scalar('Eval/loss', all_loss.avg, train_steps)
        writer.add_scalar('Eval/mse', mse_loss.avg, train_steps)
        writer.add_scalar('Eval/psnr', psnr_loss.avg, train_steps)
        writer.add_scalar('Eval/lpips', lpips_loss.avg, train_steps)
        writer.add_scalar('Eval/bpp', bpp_loss.avg, train_steps)
        writer.add_scalar('Eval/entropy_loss', entropy_loss.avg, train_steps)
        for quality_index, metrics in per_quality.items():
            for name, value in metrics.items():
                writer.add_scalar(
                    f"Eval/q{quality_index}_{name}", value, train_steps
                )

    return all_loss.avg
    
    
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.999):
#     """
#     Step the EMA model towards the current model.
#     """
#     for part_key in ema_model.keys():
#         ema_part = ema_model[part_key]
#         model_part = model.get(part_key, {})

#         for param_key in ema_part.keys():
#             if param_key in model_part:
#                 ema_tensor = ema_part[param_key]
#                 model_tensor = model_part[param_key]
#                 if torch.is_floating_point(ema_tensor):
#                     ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
#                 else:
#                     ema_tensor.copy_(model_tensor)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999, exact_parts=()):
    """
    Step the EMA model towards the current model.
    """
    for part_key in ema_model.keys():
        ema_part = ema_model[part_key]
        model_part = model.get(part_key, {})
        copy_exactly = part_key in exact_parts

        for param_key in ema_part.keys():
            if param_key in model_part:
                ema_tensor = ema_part[param_key]
                model_tensor = model_part[param_key]

                # 如果 shape 不一致，直接替换
                if ema_tensor.shape != model_tensor.shape:
                    ema_part[param_key] = model_tensor.clone()
                    continue

                # 正常 EMA 更新
                if torch.is_floating_point(ema_tensor) and not copy_exactly:
                    ema_tensor.mul_(decay).add_(model_tensor, alpha=1 - decay)
                else:
                    ema_tensor.copy_(model_tensor)
            else:
                # 如果模型里没有这个参数，保持不变
                continue

        # 如果模型新增了参数，这里同步补上
        for param_key, model_tensor in model_part.items():
            if param_key not in ema_part:
                ema_part[param_key] = model_tensor.clone()


@torch.no_grad()
def swap_saved_state_dicts(model, saved_state):
    """Swap live and saved tensors in-place without allocating a second model."""
    live_state = model.get_state_dicts_for_save()
    if live_state.keys() != saved_state.keys():
        raise RuntimeError("live and EMA model parts do not match")
    for part_key, live_part in live_state.items():
        saved_part = saved_state[part_key]
        if live_part.keys() != saved_part.keys():
            raise RuntimeError(f"live and EMA keys differ for {part_key}")
        for tensor_key, live_tensor in live_part.items():
            saved_tensor = saved_part[tensor_key]
            temporary = live_tensor.detach().clone()
            live_tensor.copy_(saved_tensor)
            saved_tensor.copy_(temporary)



def load_config(config_path):
    return load_pure_sana_config(config_path)

class ResizeIfSmall:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if w < self.min_size[0] or h < self.min_size[1]:
            new_w = max(w, self.min_size[0])
            new_h = max(h, self.min_size[1])
            return transforms.Resize((new_h, new_w))(img)
        return img
    
def create_logger(logging_dir):
    """
    Create a logger that writes color logs to stdout and plain logs to file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if len(logger.handlers) > 0:
        return logger

    if dist.get_rank() == 0:
        os.makedirs(logging_dir, exist_ok=True)

        # Terminal (stdout) handler with color
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File handler without color
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def set_seed(seed=903):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("DiT-IC training requires at least one CUDA GPU")
    backend = "nccl"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank

def cleanup_ddp():
    dist.destroy_process_group()

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pure_sana_train.yaml')
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help='Exact training checkpoint to resume; also enables train.resume.',
    )
    parser.add_argument(
        '--codec_path',
        type=str,
        default=None,
        help='Accepted V3 LIC checkpoint used to initialize multi-rate Plain.',
    )
    parser.add_argument('--global_batch_size', type=int, default=None)
    parser.add_argument('--per_gpu_batch_size', type=int, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--self_flow_warmup_steps", type=int, default=None)
    parser.add_argument("--self_flow_ramp_steps", type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--ckpt_every', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument(
        '--decoder_relay_mode',
        choices=('none', 'decoder_self', 'bitstream'),
        default=None,
    )
    parser.add_argument('--upsample_mode', type=str, default=None,
                        help="Upsampling method: 'pixelshuffle' (default) or 'convtranspose'.")
    args = parser.parse_args()

    train_config = load_config(args.config)
    if args.upsample_mode is not None:
        train_config['model']['params']['codec_param']['upsample_mode'] = args.upsample_mode
    if args.resume_checkpoint is not None:
        train_config['train']['resume'] = True
        train_config['train']['resume_checkpoint'] = args.resume_checkpoint
    if args.codec_path is not None:
        train_config['model']['params']['codec_path'] = args.codec_path
    if args.global_batch_size is not None and args.per_gpu_batch_size is not None:
        raise ValueError(
            "use only one of --global_batch_size and --per_gpu_batch_size"
        )
    if args.global_batch_size is not None:
        if args.global_batch_size < 1:
            raise ValueError("--global_batch_size must be positive")
        train_config['train']['global_batch_size'] = args.global_batch_size
    if args.per_gpu_batch_size is not None:
        if args.per_gpu_batch_size < 1:
            raise ValueError("--per_gpu_batch_size must be positive")
        train_config['train']['per_gpu_batch_size'] = args.per_gpu_batch_size
    if args.accumulation_steps is not None:
        if args.accumulation_steps < 1:
            raise ValueError("--accumulation_steps must be positive")
        train_config['train']['accumulation_steps'] = args.accumulation_steps
    for name in ('max_steps', 'eval_every', 'ckpt_every'):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                raise ValueError(f"--{name} must be positive")
            train_config['train'][name] = value
    if args.exp_name is not None:
        train_config["train"]["exp_name"] = args.exp_name
    self_flow_overrides = train_config["model"]["params"].get(
        "self_flow_params"
    )
    if args.self_flow_warmup_steps is not None:
        if args.self_flow_warmup_steps < 0 or self_flow_overrides is None:
            raise ValueError(
                "--self_flow_warmup_steps requires Self-Flow and a non-negative value"
            )
        self_flow_overrides["uniform_warmup_steps"] = (
            args.self_flow_warmup_steps
        )
    if args.self_flow_ramp_steps is not None:
        if args.self_flow_ramp_steps < 1 or self_flow_overrides is None:
            raise ValueError(
                "--self_flow_ramp_steps requires Self-Flow and a positive value"
            )
        self_flow_overrides["ramp_steps"] = args.self_flow_ramp_steps
    if args.decoder_relay_mode is not None:
        relay_params = train_config['model']['params'].setdefault(
            'decoder_relay_params',
            {},
        )
        relay_params['mode'] = args.decoder_relay_mode
    set_seed(train_config["train"]["global_seed"])

    rank, world_size, local_rank = setup_ddp()
    do_train(rank, local_rank, world_size, train_config)
    cleanup_ddp()
