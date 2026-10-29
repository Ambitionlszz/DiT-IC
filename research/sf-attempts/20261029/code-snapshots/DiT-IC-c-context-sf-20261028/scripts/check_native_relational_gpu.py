"""Real C10K/512/b4 preflight, including full-weight gradients and Adam state."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models.bt_self_flow.checkpoint import DeploymentEMA, read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.native_relational import relation_distance, spatial_relation
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.quantization_residual import CResidualTeacher
from models.sana_self_flow import unwrap_sana_transformer


def gradient_norm(module):
    return sum(p.grad.float().square().sum().item() for p in module.parameters() if p.grad is not None)**.5


def has_gradient(module):
    return any(p.grad is not None for p in module.parameters())


def run(config_path, checkpoint_path, output):
    config = load_config(config_path)
    source = read_checkpoint(checkpoint_path)
    model = make_model(config, torch.device("cuda:0"))
    model.load_bt_state(source["raw"])
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer).cuda()
    ema = DeploymentEMA(model)
    ema.state = {part: {key: v.cuda() for key, v in values.items()} for part, values in source["ema"].items()}
    groups, aux = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    optimizer.load_state_dict(source["optimizer"])
    auxiliary = torch.optim.AdamW(aux, lr=config["optimizer"]["aux_lr"], weight_decay=0)
    auxiliary.load_state_dict(source["aux_optimizer"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64.)
    losses = ImageLoss(**config["image_loss"]).cuda().eval()
    dataset = ImageDataset(config["data"]["train"], config["data"]["crop"])
    images = torch.stack([dataset[(i*19, 903+i)] for i in range(config["train"]["batch_size"])]).cuda()
    control = RateControl(.2)
    report = dict(batch=len(images), crop=config["data"]["crop"], source_step=source["step"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        predictor=model.representation_projector is not None, conditioner=model.quality_conditioner is not None)
    assert not report["predictor"] and not report["conditioner"]
    report["source_state_parts_exact"] = set(model.export_state()) == set(source["raw"])
    assert report["source_state_parts_exact"]
    del source
    gc.collect()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        original = model(images, control, capture=False)
        captured = model(images, control, capture=True)
        report["capture_image_max_error"] = (original.image-captured.image).abs().max().item()
        report["capture_image_mse"] = (original.image.float()-captured.image.float()).square().mean().item()
        assert report["capture_image_mse"] < 1e-6, report
        del original, captured
        model.codec.update(force=True)
        packet = model.compress(images[:1], control)
        decoded = model.decompress(packet.to_bytes())
        direct = model(images[:1], control).image
        report["packet_image_mse"] = (decoded.float()-direct.float()).square().mean().item()
        report["packet_total_bpp"] = len(packet.to_bytes())*8/(images.shape[-1]*images.shape[-2])
        assert report["packet_image_mse"] < 1e-6
        del decoded, direct, packet
    model.train()
    blocks = unwrap_sana_transformer(model.DiT).transformer_blocks
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        result = model(images, control, capture=True)
        views = model.relational_views(result, control, teacher, config, 10000, 0, 0, quality_b=1.)
        native = relation_distance(spatial_relation(result.feature), views.target).mean()
    native.backward()
    report["native_sf_only_gradients"] = {name: gradient_norm(part) for name, part in (
        ("g_a", model.codec.g_a), ("g_s", model.codec.g_s), ("router", model.codec.router),
        ("prefix", blocks[:model.student_layer+1]), ("tail", blocks[model.student_layer+1:]), ("vae", model.vae))}
    assert all(report["native_sf_only_gradients"][name] > 0 for name in ("g_a", "g_s", "router", "prefix"))
    assert not has_gradient(blocks[model.student_layer+1:]) and not has_gradient(model.vae)
    report["initial_native_relation_loss"] = native.item()
    # Record content diagnostics; they are not a reconstruction utility gate.
    with torch.no_grad():
        target = views.target.float()
        average = target.mean(0, keepdim=True).expand_as(target)
        n = result.feature.shape[1]
        matrix = target.new_zeros((len(images), n, n))
        offdiag = ~torch.eye(n, device=target.device, dtype=torch.bool)
        matrix[:, offdiag] = target
        permutation = torch.randperm(n, device=target.device, generator=torch.Generator(device="cuda").manual_seed(3903))
        shuffled = matrix[:, permutation][:, :, permutation][:, offdiag]
        native_relation = spatial_relation(result.feature.detach())
        report["content_diagnostics"] = dict(
            teacher_spatial_std=views.teacher_spatial_std.item(),
            teacher_relation_rms=target.square().mean().sqrt().item(),
            teacher_to_wrong_image=relation_distance(target, target.roll(1, 0)).mean().item(),
            teacher_to_spatial_shuffle=relation_distance(target, shuffled).mean().item(),
            teacher_to_cross_image_average=relation_distance(target, average).mean().item(),
            native_to_correct=relation_distance(native_relation, target).mean().item(),
            native_to_wrong=relation_distance(native_relation, target.roll(1, 0)).mean().item())
    del result, native
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    with torch.autocast("cuda", dtype=torch.float16):
        mixed = model.mixed_relation(views)
    mixed.backward()
    report["mixed_sf_only_gradients"] = dict(prefix=gradient_norm(blocks[:model.student_layer+1]),
        codec=has_gradient(model.codec), tail=has_gradient(blocks[model.student_layer+1:]), vae=has_gradient(model.vae))
    assert report["mixed_sf_only_gradients"]["prefix"] > 0
    assert not any(report["mixed_sf_only_gradients"][key] for key in ("codec", "tail", "vae"))
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    del mixed, views
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report["full_weight_updates"] = []
    for index, qa in enumerate((.2, 1.)):
        control = RateControl(qa)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(images, control, capture=True)
            views = model.relational_views(result, control, teacher, config, 10000+index, 0, 0,
                                           quality_b=1. if index == 0 else 0.)
            native = relation_distance(spatial_relation(result.feature), views.target).mean()
            rd, _ = reconstruction_objective(result, images, control, model.distortion_scale.s_data,
                                              losses, config["objective"], config["quality"])
            loss = rd+.5*native
        scaler.scale(loss).backward()
        with torch.autocast("cuda", dtype=torch.float16):
            mixed = model.mixed_relation(views)
        scaler.scale(.5*mixed).backward()
        scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], config["optimizer"]["max_grad_norm"])
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= scale, "full-weight update overflowed"
        auxiliary.zero_grad(set_to_none=True)
        model.codec.entropy_bottleneck_loss().backward()
        auxiliary.step()
        ema.update(model)
        teacher.update_from_student(model.DiT, model.text_proj, decay=config["self_flow"]["teacher_decay"])
        report["full_weight_updates"].append(dict(q=qa, rd=rd.item(), sf_native=native.item(),
            sf_mixed=mixed.item(), grad_norm=norm.item(), loss_scale=scaler.get_scale()))
        del result, views, native, mixed, rd, loss
    torch.cuda.synchronize()
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        free_device_bytes=torch.cuda.mem_get_info()[0], passed=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.config, a.checkpoint, a.output)
