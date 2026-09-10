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
import yaml
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
from config_compat import normalize_pure_sana_config
from models.losses import DISTS_INPUT_CONTRACT


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


def end_to_end_variable_rate_contract(config):
    quality = config.get("quality_control") or {}
    rate_weights = quality.get("rate_weights")
    if rate_weights is None:
        return None
    contract = {
        "version": "e2e_variable_rate_nogan_gan_v1",
        "phase": str(config["train"]["e2e_phase"]),
        "rate_weights": [float(value) for value in rate_weights],
        "freeze_codec": bool(config["model"]["params"].get("freeze_codec")),
        "loss_form": "pixel_perceptual_plus_quality_rate",
        "variable_rate_mechanism": VARIABLE_RATE_MECHANISM,
    }
    entropy_update_mode = configured_entropy_update(config)
    if entropy_update_mode != OFFICIAL_ENTROPY_UPDATE:
        contract["entropy_update_mode"] = entropy_update_mode
    training_mode = config["train"].get("training_mode", "scratch_e2e")
    if training_mode != "scratch_e2e":
        contract.update(
            {
                "version": "phase1_warmup_e2e_nogan_gan_v1",
                "training_mode": training_mode,
                "freeze_quality_endpoints": bool(
                    config["train"].get("freeze_quality_endpoints", False)
                ),
            }
        )
    return contract


def quality_rate_weight(rate_weights, quality_index):
    if rate_weights is None:
        return None
    return float(rate_weights[int(quality_index)])


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
    e2e_contract = end_to_end_variable_rate_contract(train_config)
    training_mode = train_config["train"].get(
        "training_mode", "scratch_e2e"
    )
    freeze_quality_endpoints = bool(
        train_config["train"].get("freeze_quality_endpoints", False)
    )
    e2e_rate_weights = (
        e2e_contract["rate_weights"] if e2e_contract is not None else None
    )
    if multirate_plain and num_quality_levels != 6:
        raise ValueError("multi-rate Plain training requires exactly 6 quality levels")
    quality_generator = None
    quality_counts = None
    if multirate_plain:
        quality_generator = torch.Generator(device="cpu")
        quality_generator.manual_seed(
            int(train_config["train"]["global_seed"]) + 1701
        )
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
    if freeze_quality_endpoints:
        model.set_quality_endpoints_trainable(False)
    source_checkpoint_sha256 = (
        file_sha256(params["codec_path"])
        if rank == 0 and params.get("codec_path")
        else None
    )
    loss_module = model.loss
    model.loss = None
    main_parameters = model.get_trainable_params()
    
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
        logger.info(
            "E2E training mode: %s; freeze_quality_endpoints=%s; "
            "freeze_codec=%s",
            training_mode,
            freeze_quality_endpoints,
            model.freeze_codec,
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
    opt = torch.optim.AdamW(main_parameters, lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
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
    
    milestones = train_config['optimizer']['step_lr']
    gammas = [0.5, 0.2, 0.1]

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
            decay=0.999,
            exact_parts=("state_dict_codec",) if multirate_plain else (),
        )
    model.train()
    model.module.assert_frozen_codec_contract()
    model.module.assert_reconstruction_training_contract()
    if e2e_contract is not None:
        model.module.assert_end_to_end_variable_rate_contract(
            require_frozen_quality_endpoints=freeze_quality_endpoints,
        )
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
            if e2e_contract is not None:
                if checkpoint.get("e2e_variable_rate_contract") != e2e_contract:
                    raise ValueError(
                        "resume checkpoint end-to-end variable-rate contract "
                        "does not match config"
                    )
                if (
                    rank == 0
                    and checkpoint.get("source_checkpoint_sha256")
                    != source_checkpoint_sha256
                ):
                    raise ValueError(
                        "resume checkpoint source checkpoint does not match config"
                    )
            
            model.module.load_ckpt_dict(latest_checkpoint, use_ema=False)
            model.train()
            loss_module.train()

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
            entropy_state = checkpoint.get("opt_entropy", checkpoint.get("opt_aux"))
            if opt_entropy is not None and entropy_state is not None:
                saved_entropy_update_mode = checkpoint.get(
                    "entropy_update_mode", OFFICIAL_ENTROPY_UPDATE
                )
                if saved_entropy_update_mode != entropy_update_mode:
                    raise ValueError(
                        "resume checkpoint entropy update mode does not match config: "
                        f"checkpoint={saved_entropy_update_mode!r}, "
                        f"config={entropy_update_mode!r}"
                    )
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
                MULTIRATE_CHECKPOINT_SCHEMA_VERSION if multirate_plain else 2
            ),
            "stage": (
                "multirate_reconstruction" if multirate_plain else "baseline"
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
            checkpoint["quality_counts"] = quality_counts.clone()
            codec_state = checkpoint["model"]["state_dict_codec"]
            codec_sha256 = state_dict_sha256(codec_state)
            ema_codec_sha256 = state_dict_sha256(
                checkpoint["ema"]["state_dict_codec"]
            )
            if codec_sha256 != ema_codec_sha256:
                raise RuntimeError(
                    "frozen codec drifted between live and EMA checkpoint state"
                )
            checkpoint["codec_sha256"] = codec_sha256
            checkpoint["ema_codec_sha256"] = ema_codec_sha256
            checkpoint["source_checkpoint_sha256"] = source_checkpoint_sha256
            if e2e_contract is not None:
                checkpoint["e2e_variable_rate_contract"] = e2e_contract
            checkpoint["git_commit"] = current_git_commit()
            checkpoint["bitstream_format_version"] = 1
            checkpoint["variable_rate_mechanism"] = VARIABLE_RATE_MECHANISM
            checkpoint["trainable_parameter_names"] = sorted(
                name
                for name, parameter in model.module.named_parameters()
                if parameter.requires_grad
            )
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


        if e2e_rate_weights is None:
            loss_module.rate_weight = train_config['train']['rate_weight']
        for x in loader:
            x = x.to(device)
            will_step = accum_count + 1 == accumulation_steps
            disc_active = train_steps >= disc_start
            if multirate_plain and accum_count == 0:
                quality_index = sample_shared_quality_index(
                    num_quality_levels,
                    device,
                    generator=quality_generator,
                )
            if e2e_rate_weights is not None:
                loss_module.rate_weight = quality_rate_weight(
                    e2e_rate_weights, quality_index
                )

            # ---- forward pass ----
            if multirate_plain:
                output_image, rate = model(x, quality_index)
            else:
                output_image, rate = model(x)

            # Generator loss
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

            # Pre-fix behavior from 6c43405: accumulate the current quantile
            # gradients before the shared model-wide clipping operation.
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
                # Global clipping includes the current accumulated quantile
                # gradients in historical mode and the resident previous-step
                # quantile gradients in official mode.
                if 'max_grad_norm' in train_config['optimizer']:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config['optimizer']['max_grad_norm'],
                    )
                opt.step()
                scheduler.step()

                if entropy_update_mode == HISTORICAL_ENTROPY_UPDATE:
                    # Apply the already-computed, globally clipped quantile
                    # gradients exactly as the pre-fix training loop did.
                    if opt_entropy is not None:
                        opt_entropy.step()
                else:
                    # Official DiT-IC NoGAN ordering: recompute the quantile
                    # objective after the main entropy-model parameters move.
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
                        decay=0.999,
                        exact_parts=("state_dict_codec",)
                        if multirate_plain
                        else (),
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
                if multirate_plain:
                    quality_counts[int(quality_index.item())] += 1
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
                        if multirate_plain:
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
                                rate_weights=e2e_rate_weights,
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
                        is_best = test_loss < best_loss
                        best_loss = min(test_loss, best_loss)
                        if is_best and train_steps > 0.7 * train_config['train']['max_steps']:
                            checkpoint = make_checkpoint()
                            checkpoint_path = f"{checkpoint_dir}/best_ckpt.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved best checkpoint to {checkpoint_path} at step {train_steps}")
                    dist.barrier()
                    model.train()
                    model.module.assert_reconstruction_training_contract()
                    if e2e_contract is not None:
                        model.module.assert_end_to_end_variable_rate_contract(
                            require_frozen_quality_endpoints=(
                                freeze_quality_endpoints
                            ),
                        )
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
    rate_weights=None,
):
    """Evaluate the same EMA weights that downstream P0 loading will use."""
    model.assert_reconstruction_training_contract()
    live_state = model.get_state_dicts_for_save()
    live_vae = deepcopy(live_state["state_dict_vae"])
    live_dit = deepcopy(live_state["state_dict_DiT"])
    live_text_proj = deepcopy(live_state["state_dict_text_proj"])
    model.vae.load_state_dict(ema["state_dict_vae"], strict=False)
    model.DiT.load_state_dict(ema["state_dict_DiT"], strict=False)
    model.text_proj.load_state_dict(ema["state_dict_text_proj"], strict=True)
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
            rate_weights=rate_weights,
        )
    finally:
        model.vae.load_state_dict(live_vae, strict=False)
        model.DiT.load_state_dict(live_dit, strict=False)
        model.text_proj.load_state_dict(live_text_proj, strict=True)


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
    rate_weights=None,
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
        if rate_weights is not None:
            loss_module.rate_weight = quality_rate_weight(
                rate_weights, quality_index
            )
        quality_loss = AverageMeter()
        quality_psnr = AverageMeter()
        quality_bpp = AverageMeter()
        for x in test_loader:
            x = x.to(device)

            if quality_index is None:
                output_image, rate = model(x)
            else:
                output_image, rate = model(x, quality_index)
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



def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return normalize_pure_sana_config(config)

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
    set_seed(train_config["train"]["global_seed"])

    rank, world_size, local_rank = setup_ddp()
    do_train(rank, local_rank, world_size, train_config)
    cleanup_ddp()
