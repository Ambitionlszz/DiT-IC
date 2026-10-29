"""One full public-backbone 512px C/D engineering check, without a long run.

Exercises the real pretrained SANA/DC-AE, perceptual objectives, main/aux
updates and serialized packet decode. This is not a trained RD experiment.
Use the paired GPU preflight separately to test two-card DDP and resume.
"""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

# pyiqa captures its cache directory during the model-module imports below.
os.environ.setdefault("TORCH_HOME", "/data2/ls/checkpoints/torch-hub")

from models.bt_self_flow.checkpoint import clone_state
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import BTImagePacket, RateControl
from train_bt_self_flow import auxiliary_update, gradient_diagnostics, seed_all


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; the GPU engineering check has not run")
    config = load_config(args.config)
    report = {"device": str(device), "crop": 512, "research_training": False,
              "torch": torch.__version__, "assets": asset_identity(config), "branches": {}}
    dataset = ImageDataset(config["data"]["valid"], crop=512, training=False, limit=1)
    image = dataset[0].unsqueeze(0).to(device)
    shared = None
    for backend in ("C", "D"):
        seed_all(config["train"]["seed"])
        cfg = deepcopy(config); cfg["model"]["backend"] = backend
        started = time.monotonic()
        model = make_model(cfg, device)
        model.eval()
        if shared is None:
            with torch.no_grad():
                clean = model.encode_clean_latent(image)
                model.distortion_scale.s_data.copy_(clean.square().mean().sqrt())
                scales = [model.distortion_scale.measure(clean, model.codec(clean, RateControl(q))["mean"])
                          for q in cfg["quality"]["values"]]
                model.distortion_scale.calibrate(clean.square().mean(), scales)
            shared = clone_state(model.export_state(), cpu=True)
        else:
            model.load_bt_state(shared, initialize_conditioner=True)
        model.train()
        image_loss = ImageLoss(**cfg["image_loss"]).to(device).eval()
        groups, auxiliary = optimizer_groups(model, cfg)
        optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
        aux_optimizer = torch.optim.AdamW(auxiliary, lr=cfg["optimizer"]["aux_lr"], weight_decay=0)
        control = RateControl(.4)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(image, control)
            loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                                     image_loss, objective=cfg["objective"], quality=cfg["quality"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 4.)
        gradients = gradient_diagnostics(model)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or gradients.get("router", 0) <= 0:
            raise RuntimeError(f"{backend}: nonfinite loss/gradient or absent Router gradient")
        optimizer.step()
        auxiliary_update(model.codec, aux_optimizer, 1)
        model.eval(); model.codec.update(force=True)
        with torch.no_grad():
            direct = model(image, control).image
            packet = model.compress(image, control)
            received = BTImagePacket.from_bytes(packet.to_bytes())
            before = model.prediction_calls
            def forbidden(*a, **kw):
                raise AssertionError("decoder used encoder-only input")
            model.codec.router.forward = forbidden
            model.encode_clean_latent = forbidden
            decoded = model.decompress(received)
            error = (direct-decoded).abs().max().item()
        if error > 2e-4 or model.prediction_calls-before != 1:
            raise RuntimeError(f"{backend}: packet parity failed: {error}")
        report["branches"][backend] = dict(seconds=time.monotonic()-started, loss=float(loss.detach()),
                                          grad_norm=float(norm), gradient_norms_after_clip=gradients,
                                          forward_packet_max_error=error, nfe=1,
                                          bytes_by_component=packet.byte_breakdown(),
                                          total_bpp=8*len(packet.to_bytes())/(512*512),
                                          trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model, optimizer, aux_optimizer, result, loss, metrics, groups, auxiliary, image_loss, direct, decoded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO/"configs/bt_self_flow/precision_v2/c_joint.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("report already exists; choose a new output file")
    try:
        report = run(args)
        code = 0
    except Exception as exc:
        report = dict(status="failed", device=args.device, error=f"{type(exc).__name__}: {exc}")
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
