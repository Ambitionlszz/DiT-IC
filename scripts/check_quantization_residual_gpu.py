"""Two-rank capacity probe: full student, EMA20, losses, Adam and packet.

Run via torchrun. Probe weights are discarded; this never seeds research
training. At least two *successful* updates allocate the real Adam states.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.bt_self_flow.checkpoint import DeploymentEMA, read_checkpoint
from models.bt_self_flow.data import ImageDataset, counter_seed
from models.bt_self_flow.experiment import asset_identity, check_stage_model, load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.quantization_residual import CResidualTeacher
from models.bt_self_flow.residual_training import calibrate_residual, training_mask
from train_bt_self_flow import auxiliary_update, distributed_setup, initialize_from, seed_all


def probe(args):
    rank, world, device = distributed_setup("cuda")
    if world != 2:
        raise ValueError("capacity probe requires the actual two-GPU DDP layout")
    config = load_config(args.config)
    seed_all(config["train"]["seed"])
    source = read_checkpoint(args.init_checkpoint)
    if asset_identity(config) != source["assets"]:
        raise ValueError("probe assets differ from warmup")
    check_stage_model(config, source["config"], asset_verified=True)
    model = make_model(config, device)
    initialize_from(model, source, "joint")
    del source
    model.train()
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer).to(device)
    ema = DeploymentEMA(model)
    groups, auxiliary = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    aux_optimizer = torch.optim.AdamW(auxiliary, lr=config["optimizer"]["aux_lr"], weight_decay=0)
    model._ddp_params_and_buffers_to_ignore = [name for name, _ in model.named_parameters() if name.endswith(".quantiles")]
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    losses = ImageLoss(**config["image_loss"]).to(device).eval()
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    batch, accumulation = config["train"]["batch_size"], config["train"]["accumulation_steps"]
    scaler = torch.amp.GradScaler("cuda", init_scale=64.)
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    calibration_report = None
    updates = attempts = 0
    while updates < args.updates:
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            from contextlib import nullcontext
            context = wrapped.no_sync() if micro < accumulation-1 else nullcontext()
            images = torch.stack([dataset[((rank*batch+i+micro*2*batch) % len(dataset),
                                            counter_seed(903, updates*100+i+rank*batch, 1101))]
                                  for i in range(batch)]).to(device)
            mask, _ = training_mask(config, config["self_flow"]["warmup"]+config["self_flow"]["ramp"],
                                     micro, rank, batch, device)
            control = RateControl(config["quality"]["values"][updates % 6])
            with context:
                with torch.autocast("cuda", dtype=torch.float16):
                    active = model.residual_head.active.item()
                    result = wrapped(images, control, sf_teacher=teacher if active else None,
                                     sf_mask=mask if active else None)
                    rd, metrics = reconstruction_objective(result, images, control, model.distortion_scale.s_data,
                                                            losses, config["objective"], config["quality"])
                    loss = rd+.001*result.auxiliary["sf_loss"]/.5 if active else rd
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("nonfinite probe forward")
                # Match the research trainer: backward must leave autocast so
                # the FP32 codec's gradient matmuls retain FP32 semantics.
                scaler.scale(loss/accumulation).backward()
        scaler.unscale_(optimizer)
        if attempts == 0:
            names_and_gradients = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
            flags = torch.stack([torch.isfinite(g).all() for _, g in names_and_gradients]).tolist()
            bad = [n for (n, _), finite in zip(names_and_gradients, flags) if not finite]
            print(json.dumps(dict(rank=rank, first_backward_nonfinite=bad[:20], total_nonfinite=len(bad))), flush=True)
        norm = torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 4.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        if scaler.get_scale() < previous_scale:
            print(json.dumps(dict(rank=rank, amp_skip=attempts, loss_scale=scaler.get_scale())), flush=True)
            if attempts > args.updates+20:
                raise FloatingPointError("capacity probe could not obtain finite optimizer updates")
            continue
        auxiliary_update(model.codec, aux_optimizer, world)
        teacher.update_from_student(model.DiT, model.text_proj, decay=.999)
        ema.update(model)
        updates += 1
        torch.cuda.synchronize(device)
        record = dict(rank=rank, updates=updates, loss=float(loss.detach()), gradient_norm=float(norm),
                      sf_active=active, sf_valid=float(result.auxiliary["sf_valid"]) if active else 0.,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      optimizer_states=len(optimizer.state))
        print(json.dumps(record), flush=True)
        records.append(record)
        if updates == 1:
            model.activate_residual()
            teacher.update_from_student(model.DiT, model.text_proj, decay=0.)
            if args.check_calibration:
                optimizer.zero_grad(set_to_none=True)
                aux_optimizer.zero_grad(set_to_none=True)
                calibration_report = calibrate_residual(model, teacher, dataset, losses, config,
                                                         rank, world, torch.float16)
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError("transition calibration changed parameter gradients")
                print(json.dumps(dict(rank=rank, calibration_completed=calibration_report)), flush=True)
    model.eval()
    model.codec.update(force=True)
    with torch.no_grad():
        image = images[:1]
        packet = model.compress(image, RateControl(.4))
        decoded = model.decompress(packet.to_bytes())
        hard = model(image, RateControl(.4)).image
        error = float((decoded-hard).abs().max())
        if error > 2e-4 or not torch.isfinite(decoded).all().item():
            raise RuntimeError(f"active residual packet/forward mismatch {error}")
    state = dict(rank=rank, passed=True, batch_size=batch, accumulation_steps=accumulation,
                 effective_batch=world*batch*accumulation, records=records,
                 packet_forward_max_error=error,
                 calibration=calibration_report,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    gathered = [None]*world
    dist.all_gather_object(gathered, state)
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(passed=True, capacity_only=True, ranks=gathered), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--check-calibration", action="store_true",
                        help="also test the actual two-rank transition calibration before SF")
    args = parser.parse_args()
    if args.updates < 3:
        parser.error("probe requires one native and at least two active-SF updates")
    try:
        probe(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
