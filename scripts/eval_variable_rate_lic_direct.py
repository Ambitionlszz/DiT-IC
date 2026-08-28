import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import lpips
from PIL import Image
import pyiqa
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from codec_packet import read_codec_packet, write_codec_packet
from train_variable_rate_lic_ddp import (
    CHECKPOINT_SCHEMA_VERSION,
    STAGE_NAME,
    VariableRateLICStage,
    initialization_contract,
    load_config,
    objective_contract,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUMMARY_SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_manifest_sha256(image_paths):
    manifest = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in image_paths
    ]
    payload = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_stage(config, checkpoint_path, weights, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != STAGE_NAME:
        raise ValueError(f"checkpoint stage is not {STAGE_NAME!r}")
    if int(checkpoint.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"direct LIC evaluation requires schema {CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("objective_contract") != objective_contract(config):
        raise ValueError("checkpoint objective/lambda contract does not match config")
    if checkpoint.get("initialization_contract") != initialization_contract(config):
        raise ValueError("checkpoint initialization contract does not match config")

    stage = VariableRateLICStage(config)
    stage.codec.load_state_dict(
        checkpoint[weights]["state_dict_codec"], strict=True
    )
    stage.to(device).eval()
    stage.codec.update(force=True)
    return stage


def load_image(path, device):
    image = TF.to_tensor(Image.open(path).convert("RGB"))
    image = image.mul(2).sub(1).unsqueeze(0).to(device)
    height, width = image.shape[-2:]
    pad_h = math.ceil(height / 256) * 256 - height
    pad_w = math.ceil(width / 256) * 256 - width
    return image, F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


@torch.no_grad()
def evaluate_quality(stage, image_paths, output_dir, quality_index, metrics, device):
    quality_dir = output_dir / f"q{quality_index}"
    bin_dir = quality_dir / "bin"
    rec_dir = quality_dir / "rec"
    bin_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    per_image = []

    for image_path in image_paths:
        reference, padded = load_image(image_path, device)
        height, width = reference.shape[-2:]

        torch.cuda.synchronize(device)
        encode_started = perf_counter()
        clean_latent = stage.encode_clean_latent(padded)
        packet = stage.codec.compress(clean_latent, quality_index)
        torch.cuda.synchronize(device)
        encode_seconds = perf_counter() - encode_started

        packet_path = bin_dir / f"{image_path.stem}.bin"
        with packet_path.open("wb") as file:
            write_codec_packet(file, packet)
        with packet_path.open("rb") as file:
            stored_packet = read_codec_packet(file)

        torch.cuda.synchronize(device)
        decode_started = perf_counter()
        reconstructed_latent = stage.codec.decompress(stored_packet)
        reconstruction = stage.vae.decode(
            reconstructed_latent / stage.vae.config.scaling_factor,
            return_dict=False,
        )[0].clamp(-1, 1)
        torch.cuda.synchronize(device)
        decode_seconds = perf_counter() - decode_started
        reconstruction = reconstruction[:, :, :height, :width]

        reference_01 = reference.add(1).mul(0.5).clamp(0, 1)
        reconstruction_01 = reconstruction.add(1).mul(0.5).clamp(0, 1)
        mse = F.mse_loss(reconstruction_01, reference_01).item()
        result = {
            "image": image_path.name,
            "bytes": packet_path.stat().st_size,
            "bpp": packet_path.stat().st_size * 8.0 / (height * width),
            "psnr": -10.0 * math.log10(max(mse, 1e-12)),
            "ms_ssim": metrics["ms_ssim"](
                reconstruction_01, reference_01
            ).item(),
            "lpips": metrics["lpips"](reconstruction, reference).mean().item(),
            "dists": metrics["dists"](
                reconstruction_01, reference_01
            ).mean().item(),
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
        }
        per_image.append(result)
        TF.to_pil_image(reconstruction_01[0].cpu()).save(
            rec_dir / f"{image_path.stem}.png"
        )

    averages = {
        key: sum(item[key] for item in per_image) / len(per_image)
        for key in (
            "bpp",
            "psnr",
            "ms_ssim",
            "lpips",
            "dists",
            "encode_seconds",
            "decode_seconds",
        )
    }
    report = {
        "quality_index": quality_index,
        "num_images": len(per_image),
        "averages": averages,
        "per_image": per_image,
    }
    (quality_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--quality_indices", type=int, nargs="+", default=list(range(6))
    )
    args = parser.parse_args()
    if any(index not in range(6) for index in args.quality_indices):
        raise ValueError("quality_indices must be in [0, 5]")

    image_paths = sorted(
        path
        for path in Path(args.image_dir).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"no images found in {args.image_dir}")

    device = torch.device(f"cuda:{args.gpu}")
    config = load_config(args.config, calibration_confirmed=True)
    stage = load_stage(config, args.checkpoint, args.weights, device)
    metrics = {
        "ms_ssim": pyiqa.create_metric("ms_ssim").to(device).eval(),
        "lpips": lpips.LPIPS(net="alex").to(device).eval(),
        "dists": pyiqa.create_metric("dists").to(device).eval(),
    }
    output_dir = Path(args.output_dir)
    reports = [
        evaluate_quality(stage, image_paths, output_dir, index, metrics, device)
        for index in args.quality_indices
    ]
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "weights": args.weights,
        "image_dir": str(Path(args.image_dir).resolve()),
        "image_manifest_sha256": image_manifest_sha256(image_paths),
        "num_images": len(image_paths),
        "qualities": {
            f"q{report['quality_index']}": report["averages"]
            for report in reports
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
