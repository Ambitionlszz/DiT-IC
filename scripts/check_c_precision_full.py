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
os.environ.setdefault("TORCH_HOME", str(Path.home()/".cache/torch"))

from models.bt_self_flow.checkpoint import clone_state
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import BTImagePacket, RateControl
from train_bt_self_flow import auxiliary_update, gradient_diagnostics, seed_all


def wait_gate(path, parent_pid):
    while not Path(path).exists():
        if parent_pid and not Path(f"/proc/{parent_pid}").exists():
            raise InterruptedError("preflight supervisor exited")
        time.sleep(.01)


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; the GPU engineering check has not run")
    config = load_config(args.config)
    report = {"device": str(device), "crop": 512, "batch_size": args.batch_size, "research_training": False,
              "torch": torch.__version__, "assets": asset_identity(config), "branches": {}}
    dataset = ImageDataset(config["data"]["valid"], crop=512, training=False, limit=args.batch_size)
    image = torch.stack([dataset[index % len(dataset)] for index in range(args.batch_size)]).to(device)
    shared = None
    backends = ("C",) if config["model"]["codec"].get("precision_consistent") else ("C", "D")
    if args.hold_ready and (device.type != "cuda" or backends != ("C",)):
        raise ValueError("protected preflight requires a single C v3 CUDA branch")
    for backend in backends:
        seed_all(config["train"]["seed"])
        cfg = deepcopy(config); cfg["model"]["backend"] = backend
        started = time.monotonic()
        model = make_model(cfg, torch.device("cpu") if args.allocation_gate else device)
        if args.allocation_gate:
            probe = torch.empty(64*1024**2, dtype=torch.uint8, device=device).fill_(1)
            torch.cuda.synchronize(device)
            Path(args.allocation_ready).write_text(json.dumps(dict(model_preloaded=True, pid=os.getpid()))+"\n")
            wait_gate(args.allocation_gate, args.parent_pid)
            del probe
            torch.cuda.empty_cache()
            model.to(device)
        if args.compute_gate:
            torch.cuda.synchronize(device)
            Path(args.compute_ready).write_text(json.dumps(dict(model_resident=True, pid=os.getpid()))+"\n")
            wait_gate(args.compute_gate, args.parent_pid)
        print(json.dumps(dict(stage="model_loaded", backend=backend, batch_size=args.batch_size)), flush=True)
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
        print(json.dumps(dict(stage="backward_completed", backend=backend, loss=float(loss.detach()))), flush=True)
        norm = torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 4.)
        gradients = gradient_diagnostics(model)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or gradients.get("router", 0) <= 0:
            raise RuntimeError(f"{backend}: nonfinite loss/gradient or absent Router gradient")
        optimizer.step()
        auxiliary_update(model.codec, aux_optimizer, 1)
        model.eval(); model.codec.update(force=True)
        with torch.no_grad():
            direct = model(image[:1], control).image
            packet = model.compress(image[:1], control)
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
        report["branches"][backend] = dict(mechanism=model.codec.mechanism, packet_version=packet.version,
                                          seconds=time.monotonic()-started, loss=float(loss.detach()),
                                          grad_norm=float(norm), gradient_norms_after_clip=gradients,
                                          forward_packet_max_error=error, nfe=1,
                                          bytes_by_component=packet.byte_breakdown(),
                                          total_bpp=8*len(packet.to_bytes())/(512*512),
                                          trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if device.type == "cuda":
            report["branches"][backend]["max_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["branches"][backend]["max_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        if args.hold_ready:
            optimizer.zero_grad(set_to_none=True)
            del result, loss, metrics, direct, decoded
            gc.collect()
            torch.cuda.empty_cache()
            report["status"] = "passed"
            Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
            Path(args.hold_ready).write_text(json.dumps(dict(status="passed", pid=os.getpid(),
                                                            resident_bytes=torch.cuda.memory_reserved(device)))+"\n")
            wait_gate(args.hold_release, args.parent_pid)
            del model, optimizer, aux_optimizer, groups, auxiliary, image_loss
            gc.collect()
            torch.cuda.empty_cache()
            continue
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allocation-ready")
    parser.add_argument("--allocation-gate")
    parser.add_argument("--compute-ready")
    parser.add_argument("--compute-gate")
    parser.add_argument("--hold-ready")
    parser.add_argument("--hold-release")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    for ready, gate in ((args.allocation_ready, args.allocation_gate),
                        (args.compute_ready, args.compute_gate), (args.hold_ready, args.hold_release)):
        if bool(ready) != bool(gate):
            parser.error("preflight readiness and gate paths must be supplied together")
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
