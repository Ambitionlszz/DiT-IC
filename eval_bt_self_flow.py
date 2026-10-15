"""Real-packet RD evaluation or decoding in an independent process."""

import argparse
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torch.nn import functional as F

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, make_model
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.model import pad_image
from models.bt_self_flow.packet import BTImagePacket, RateControl, unpack_codes
from models.bt_self_flow.router import allocation_from_codes


def save_image(tensor, path):
    array = tensor[0].detach().float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU evaluation has not been validated on this host")
    checkpoint = read_checkpoint(args.checkpoint)
    if checkpoint["phase"] not in ("backend", "joint", "self_flow"):
        raise ValueError("image evaluation requires an adapted reconstruction checkpoint")
    if checkpoint["phase"] == "backend" and checkpoint["step"] != checkpoint["config"]["train"]["steps"]:
        raise ValueError("a partially transitioned backend is not a deployment model")
    config = checkpoint["config"]
    if checkpoint["assets"] != asset_identity(config):
        raise ValueError("pretrained asset identity differs from checkpoint")
    model = make_model(config, device)
    model.load_bt_state(checkpoint[args.weights])
    model.eval().requires_grad_(False)
    model.codec.update(force=True)
    if args.decode_packet:
        packet = BTImagePacket.from_bytes(Path(args.decode_packet).read_bytes())
        before = model.prediction_calls
        decoded = model.decompress(packet)
        if model.prediction_calls-before != 1:
            raise RuntimeError("deployment decoder performed more than one SANA prediction")
        save_image(decoded, args.output)
        print(json.dumps({"decoded": args.output, "nfe": 1}), flush=True)
        return
    if not args.input:
        raise ValueError("--input is required for RD evaluation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ImageDataset(args.input, crop=None, training=False, limit=args.limit)
    metric = ImageLoss().to(device).eval()
    rows = []
    for quality in args.qualities:
        control = RateControl(quality)
        directory = output/f"q{control.low_code:05d}"
        directory.mkdir(exist_ok=True)
        for index, source_path in enumerate(dataset.paths):
            image = dataset[index].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            started = time.perf_counter()
            packet = model.compress(image, control)
            synchronize(device)
            encode_seconds = time.perf_counter()-started
            data = packet.to_bytes()
            # Read the serialized representation, not a sender-side object.
            received = BTImagePacket.from_bytes(data)
            before = model.prediction_calls
            synchronize(device)
            started = time.perf_counter()
            reconstructed = model.decompress(received)
            synchronize(device)
            decode_seconds = time.perf_counter()-started
            nfe = model.prediction_calls-before
            if nfe != 1:
                raise RuntimeError(f"expected 1 NFE, got {nfe}")
            _, values = metric(reconstructed, image)
            h, w = image.shape[-2:]
            yh, yw = packet.y_shape
            allocation = torch.tensor(unpack_codes(packet.allocation, 3, yh*yw), device=device).reshape(1, 1, yh, yw)
            delta = model.codec.steps(received.rate_control(device), allocation_from_codes(allocation))
            pixel_delta = F.interpolate(delta, size=(yh*64, yw*64), mode="nearest")[..., :h, :w]
            precise = (pixel_delta <= pixel_delta.flatten().quantile(.25)).float()
            region_mse = ((reconstructed-image).square()*precise).sum()/(3*precise.sum()).clamp_min(1)
            row = {"image": str(source_path), "lambda": control.value, "lambda_code": control.low_code,
                   "height": h, "width": w, "bytes": len(data), "bpp": 8*len(data)/(h*w),
                   "bytes_by_component": packet.byte_breakdown(), "encode_seconds": encode_seconds,
                   "decode_seconds": decode_seconds, "nfe": nfe,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   "high_precision_area_fraction": precise.mean().item(),
                   "high_precision_psnr": -10*torch.log10(region_mse.clamp_min(1e-12)/4).item(),
                   **{name: float(value) for name, value in values.items()}, **model.last_compress_stats,
                   **model.last_decode_stats}
            if args.check_forward:
                direct = model(pad_image(image), rate_control=control)
                bit_error = (direct.image[..., :h, :w]-reconstructed).abs().max().item()
                if bit_error > 2e-4:
                    raise RuntimeError(f"hard forward and packet image differ by {bit_error}")
                row["forward_max_error"] = bit_error
                bits_est = sum(-v.log2().sum().item() for v in direct.likelihoods.values())+direct.side_bits
                row["bpp_estimate"] = bits_est/(h*w)
                row["bpp_coding_gap"] = row["bpp"]-row["bpp_estimate"]
            # Include the index to avoid overwriting identical names in roots.
            stem = f"{index:05d}_{source_path.stem}"
            (directory/f"{stem}.btic").write_bytes(data)
            if args.save_images:
                save_image(reconstructed, directory/f"{stem}.png")
            rows.append(row)
            with (output/"images.jsonl").open("a") as fd:
                fd.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
    summary = {}
    for quality in args.qualities:
        code = RateControl(quality).low_code
        subset = [row for row in rows if row["lambda_code"] == code]
        keys = ("bpp", "psnr", "lpips", "dists", "encode_seconds", "decode_seconds", "peak_memory_bytes")
        keys += ("base_latent_mse", "repair_latent_rms", "high_precision_psnr", "high_precision_area_fraction")
        if "scale_saturation" in subset[0]:
            keys += ("scale_saturation",)
        summary[str(code)] = {key: sum(row[key] for row in subset)/len(subset) for key in keys}
        summary[str(code)]["bytes_by_component"] = {
            name: sum(row["bytes_by_component"][name] for row in subset)/len(subset)
            for name in subset[0]["bytes_by_component"]}
    manifest = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
                "weights": args.weights, "backend": model.backend, "step": checkpoint["step"], "phase": checkpoint["phase"],
                "dataset_fingerprint": dataset.fingerprint, "images": len(dataset),
                "metric_protocol": "float RGB [-1,1], PSNR dynamic range 2, LPIPS alex, DISTS [0,1]",
                "padding": "right/bottom to 256, reflect when possible else replicate, original-pixel bpp",
                "codec_precision": "fp32", "nfe": 1, "summary": summary}
    manifest["high_precision_region"] = "nearest-upsampled transmitted step <= original-image 25th percentile, ties included"
    (output/"summary.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-packet")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--qualities", type=float, nargs="+", default=[0, .2, .4, .6, .8, 1])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--check-forward", action="store_true")
    args = parser.parse_args()
    if not args.decode_packet and Path(args.output, "images.jsonl").exists():
        parser.error("output already contains an evaluation; use a new output directory")
    evaluate(args)


if __name__ == "__main__":
    main()
