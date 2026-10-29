"""Bounded engineering checks for the C/PiT decoder, not a research run."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from models.bt_self_flow.checkpoint import DeploymentEMA
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.pixel_adapter import PixelDecoderAdapter
from train_bt_self_flow import auxiliary_update, gradient_diagnostics, seed_all


def setup(args):
    torch.set_num_threads(args.threads)
    seed_all(903)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.memory_limit_gib:
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 1024 ** 3 / total, device)
    return device


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if device.type == "cuda" else nullcontext()


def check_adapter(args, device):
    adapter = PixelDecoderAdapter(128, 1152).to(device)
    features = torch.randn(args.batch_size, 128, 512, 512, device=device, requires_grad=True)
    tokens = torch.randn(args.batch_size, 256, 1152, device=device, requires_grad=True)
    delta = torch.full((args.batch_size, 1, 8, 8), 1.5, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0)
    steps = []
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        features.grad = tokens.grad = delta.grad = None
        with autocast(device):
            updated = adapter(features, tokens, (16, 16), delta)
            if step == 0 and not torch.equal(updated, features):
                raise AssertionError("zero initialization changed AE features")
            loss = updated.square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 4.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite PiT gradient")
        gradients = {"block_0": float(adapter.blocks[0].attn.qkv.weight.grad.float().norm()),
                     "block_1": float(adapter.blocks[1].attn.qkv.weight.grad.float().norm()),
                     "content": float(adapter.content_projection.weight.grad.float().norm()),
                     "scale": float(adapter.scale_embedding.mlp[0].weight.grad.float().norm())}
        if step > 0 and any(value <= 0 for value in gradients.values()):
            raise AssertionError(f"missing PiT/condition gradient: {gradients}")
        optimizer.step()
        steps.append(dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(norm), gradients=gradients))
        print(json.dumps(steps[-1]), flush=True)
    return dict(scope="production_width_pixel_adapter_only", pixel_parameters=sum(p.numel() for p in adapter.parameters()),
                feature_shape=list(features.shape), steps=steps, zero_initialization_exact=True)


def check_full(args, device):
    config = load_config(args.config)
    model = make_model(config, device).eval()
    dataset = ImageDataset(config["data"]["valid"], crop=512, training=False, limit=1)
    image = dataset[0].unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device)
    control = RateControl(.4)
    with torch.no_grad():
        clean = model.encode_clean_latent(image)
        if not getattr(args, "retain_calibrated_scale", False):
            model.distortion_scale.s_data.copy_(clean.square().mean().sqrt())
        view = model.codec(clean, control)
        _, native, _ = BTCodec.repair(model, view["mean"])
        output = model.decode_received(view["mean"], view["delta"])
        initial_error = (output["image"] - native).abs().max().item()
        if initial_error > 2e-6:
            raise AssertionError(f"initial output differs from native C: {initial_error}")
        shapes = dict(mu=list(view["mean"].shape), delta=list(view["delta"].shape), image=list(output["image"].shape))
    del clean, view, native, output
    print(json.dumps(dict(stage="initial_parity", max_error=initial_error, shapes=shapes)), flush=True)
    groups, auxiliary = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    aux_optimizer = torch.optim.AdamW(auxiliary, lr=config["optimizer"]["aux_lr"], weight_decay=0)
    image_loss = ImageLoss(**config["image_loss"]).to(device).eval()
    ema = DeploymentEMA(model)
    steps = []
    model.train()
    for step in range(args.updates):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            result = model(image, control)
            loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                                     image_loss, objective=config["objective"], quality=config["quality"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]],
                                             config["optimizer"]["max_grad_norm"])
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("nonfinite full-backbone loss/gradient")
        gradients = gradient_diagnostics(model)
        optimizer.step()
        auxiliary_update(model.codec, aux_optimizer, 1)
        ema.update(model)
        steps.append(dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(norm),
                          seconds=time.monotonic() - started, gradient_norms=gradients))
        print(json.dumps(steps[-1]), flush=True)
        del result, loss, metrics
    optimizer.zero_grad(set_to_none=True)
    model.eval(); model.codec.update(force=True)
    one = image[:1]
    with torch.no_grad():
        expected = model(one, control).image
        packet = model.compress(one, control)
        def forbidden(*args, **kwargs):
            raise AssertionError("packet decoder used sender-only data")
        original_encode = model.encode_clean_latent
        original_router = model.codec.router.forward
        model.encode_clean_latent = forbidden
        model.codec.router.forward = forbidden
        calls = model.prediction_calls
        try:
            received = model.decompress(packet.to_bytes())
            error = (received - expected).abs().max().item()
        finally:
            model.encode_clean_latent = original_encode
            model.codec.router.forward = original_router
    if error > 2e-4 or model.prediction_calls - calls != 1:
        raise AssertionError(f"packet parity/NFE failed: {error}")
    return dict(scope="pretrained_sana_dc_ae_codec_perceptual_rd", steps=steps, shapes=shapes,
                pixel_parameters=sum(p.numel() for p in model.pixel_adapter.parameters()),
                ae_feature_channels=model.pixel_adapter.pixel_input.in_features,
                initial_native_max_error=initial_error, forward_packet_max_error=error,
                packet_bytes=packet.byte_breakdown(), prediction_calls_per_packet=1,
                self_flow_enabled=False, ema_includes_pixel_adapter="pixel_adapter" in ema.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("adapter", "full"), default="adapter")
    parser.add_argument("--config", default=str(REPO / "configs/bt_self_flow/pixel_adapter/joint.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--memory-limit-gib", type=float, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.updates < 2 or args.batch_size < 1:
        parser.error("use at least two updates and a positive batch size")
    target = Path(args.output)
    if target.exists():
        parser.error("report exists; choose a new output")
    try:
        device = setup(args)
        report = check_full(args, device) if args.scope == "full" else check_adapter(args, device)
        report.update(status="passed", device=str(device), batch_size=args.batch_size, research_training=False,
                      torch=torch.__version__)
        if device.type == "cuda":
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        code = 0
    except Exception as error:
        report = dict(status="failed", device=args.device, scope=args.scope,
                      error=f"{type(error).__name__}: {error}", research_training=False)
        code = 1
        import traceback
        traceback.print_exc()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
