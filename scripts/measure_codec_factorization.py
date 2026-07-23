"""Measure whether the Pure SANA codec has a useful factorized path.

Reports per-image latent RMS statistics for:
  transform:    g_s(y) - z_0
  quantization: g_s(y_hat) - g_s(y)
  curvature:    g_s((y + y_hat) / 2) - (g_s(y) + g_s(y_hat)) / 2

The default path obtains ``y_hat`` from real rANS compress/decompress.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from diffusers import AutoencoderDC
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from config_compat import normalize_pure_sana_config
from datasets.image import ImageFolder
from models.latent_codec import LatentCodec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Pure SANA YAML")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pure SANA checkpoint",
    )
    parser.add_argument("--data-root", action="append")
    parser.add_argument("--dit-path", help="Override model.params.dit_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--state",
        choices=("auto", "model", "ema"),
        default="auto",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--decode-source",
        choices=("rans", "forward"),
        default="rans",
        help="Real rANS decode by default; forward is an approximation.",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_models(config, args):
    params = config["model"]["params"]
    codec = LatentCodec(**dict(params.get("codec_param", {})))

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    state_name = args.state
    if state_name == "auto":
        state_name = "ema" if params.get("use_ema", False) else "model"
    if state_name not in checkpoint:
        raise KeyError(f"Checkpoint has no {state_name!r} state")
    state = checkpoint[state_name]
    if "state_dict_codec" not in state:
        raise KeyError(f"{state_name!r} has no 'state_dict_codec'")

    current = codec.state_dict()
    saved = state["state_dict_codec"]
    required_prefixes = (
        "g_a.",
        "g_s.",
        "h_a.",
        "h_s.",
        "adapter_in.",
        "g_c.",
        "adapter_out.",
        "LRP.",
        "entropy_bottleneck.",
        "gaussian_conditional.",
    )
    ignored_buffers = (
        "_quantized_cdf",
        "_cdf_length",
        "_offset",
        "scale_table",
    )
    missing = [
        key
        for key in current
        if key.startswith(required_prefixes)
        and key not in saved
        and not key.endswith(ignored_buffers)
    ]
    if missing:
        raise RuntimeError(
            "Missing active codec weights: " + ", ".join(missing[:10])
        )
    current.update(
        {
            key: value
            for key, value in saved.items()
            if key in current
        }
    )
    codec.load_state_dict(current)
    codec.update()

    vae_lora = state.get("state_dict_vae", {})
    non_decoder_lora = [
        key
        for key in vae_lora
        if "decoder" not in key.lower()
    ]
    if non_decoder_lora:
        raise RuntimeError(
            "Checkpoint changes the VAE encoder; this probe would not "
            "reproduce z_0 correctly"
        )

    dit_path = args.dit_path or params.get("dit_path")
    if not dit_path:
        raise ValueError("Provide --dit-path or model.params.dit_path")
    vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")
    return vae, codec, state_name, dit_path


def build_loader(config, args):
    data = config.get("data", {})
    roots = args.data_root or data.get("valid_path")
    roots = roots if isinstance(roots, list) else [roots]
    if not roots or roots == [None]:
        raise ValueError("Provide --data-root or data.valid_path")

    crop_size = data.get("path_size")
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    if not crop_size:
        raise ValueError("data.path_size is required")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
            transforms.CenterCrop(crop_size),
        ]
    )
    extensions = data.get(
        "im_exts",
        ["png", "jpg", "jpeg", "PNG", "JPEG"],
    )
    dataset = ImageFolder(roots, extensions, transform=transform)
    if args.max_images > 0:
        dataset = Subset(
            dataset,
            range(min(args.max_images, len(dataset))),
        )
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {roots}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    return loader, crop_size


def rms(left, right):
    return (
        (left.float() - right.float())
        .square()
        .flatten(1)
        .mean(1)
        .sqrt()
    )


@torch.no_grad()
def measure_image(vae, codec, image, decode_source):
    z_0 = vae.encode(image).latent * vae.config.scaling_factor
    captured_y = []
    captured_y_hat = []

    def save_y(_module, _inputs, output):
        captured_y.append(output.detach())

    def save_y_hat(_module, inputs):
        captured_y_hat.append(inputs[0].detach())

    if decode_source == "rans":
        handle = codec.g_a.register_forward_hook(save_y)
        try:
            packed = codec.compress(z_0)
        finally:
            handle.remove()

        handle = codec.g_s.register_forward_pre_hook(save_y_hat)
        try:
            z_1 = codec.decompress(
                packed["strings"],
                packed["shape"],
            )
        finally:
            handle.remove()
    else:
        handle_y = codec.g_a.register_forward_hook(save_y)
        handle_y_hat = codec.g_s.register_forward_pre_hook(save_y_hat)
        try:
            z_1 = codec(z_0)["mean"]
        finally:
            handle_y.remove()
            handle_y_hat.remove()

    if len(captured_y) != 1 or len(captured_y_hat) != 1:
        raise RuntimeError(
            f"Expected one y and y_hat, got {len(captured_y)} and "
            f"{len(captured_y_hat)}"
        )

    y = captured_y[0]
    y_hat = captured_y_hat[0]
    z_u = codec.g_s(y)
    codec_midpoint = codec.g_s(0.5 * (y + y_hat))
    chord_midpoint = 0.5 * (z_u + z_1)

    transform = rms(z_u, z_0)
    quantization = rms(z_1, z_u)
    curvature = rms(codec_midpoint, chord_midpoint)
    total = rms(z_1, z_0)
    eps = torch.finfo(torch.float32).eps
    return {
        "transform_rms": transform,
        "quantization_rms": quantization,
        "curvature_rms": curvature,
        "total_rms": total,
        "transform_over_total": transform / total.clamp_min(eps),
        "quantization_over_total": (
            quantization / total.clamp_min(eps)
        ),
        "curvature_over_quantization": (
            curvature / quantization.clamp_min(eps)
        ),
        "curvature_over_total": curvature / total.clamp_min(eps),
    }


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
    }


def main():
    args = parse_args()
    config = normalize_pure_sana_config(load_yaml(args.config))
    loader, crop_size = build_loader(config, args)
    vae, codec, state_name, dit_path = load_models(config, args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {args.device}")
    vae = vae.to(device).eval()
    codec = codec.to(device).eval()

    collected = {}
    for image in loader:
        metrics = measure_image(
            vae,
            codec,
            image.to(device, non_blocking=True),
            args.decode_source,
        )
        for name, value in metrics.items():
            collected.setdefault(name, []).extend(
                value.cpu().tolist()
            )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_state": state_name,
        "config": str(Path(args.config).resolve()),
        "dit_path": dit_path,
        "decode_source": args.decode_source,
        "image_count": len(next(iter(collected.values()))),
        "crop_size": crop_size,
        "metrics": {
            name: summarize(values)
            for name, values in collected.items()
        },
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
