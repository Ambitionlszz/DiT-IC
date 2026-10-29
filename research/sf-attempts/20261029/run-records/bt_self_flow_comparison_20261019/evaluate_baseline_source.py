#!/usr/bin/env python3
"""Evaluate full-image global/window decoding of identical real rANS packets."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from codec_packet import CodecPacket, read_codec_packet, write_codec_packet
from config_compat import load_pure_sana_config, prepare_inference_config


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_received_packet(packet, path, quality):
    if not isinstance(packet, CodecPacket) or packet.format_version != 1:
        raise ValueError("requires a plain E2E v1 rANS packet")
    if packet.quality_index != quality:
        raise ValueError("encoder returned the wrong quality index")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        written = write_codec_packet(stream, packet)
    with path.open("rb") as stream:
        received = read_codec_packet(stream)
        if stream.read(1) or received != packet or written != path.stat().st_size:
            raise ValueError("serialized packet round-trip failed")
    return received


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def assert_plain_e2e(net, params):
    """Older plain-E2E branches do not define AuxED/Q-Former attributes."""
    configured = (
        params.get("use_aux_encoder", False)
        or params.get("use_aux_decoder", False)
        or (params.get("query_condition_config") or {}).get("enabled", False)
        or (params.get("self_flow_params") or {}).get("enabled", False)
    )
    instantiated = any(getattr(net, flag, False) for flag in (
        "use_aux_encoder", "use_aux_decoder", "query_condition_enabled",
        "self_flow_enabled",
    ))
    if configured or instantiated:
        raise ValueError("this paired evaluation requires plain E2E without auxiliary conditioning")
    if net.decoder_relay_mode != "none":
        raise ValueError("decoder relay must be disabled")


def validate_variants(variants):
    if (not variants or len(set(variants)) != len(variants)
            or any(variant not in ("global", "window") for variant in variants)):
        raise ValueError("decoder variants must be nonempty, unique, and global/window")
    return tuple(variants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--qualities", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--expected-step", type=int, default=200000)
    parser.add_argument("--expected-weights", nargs=3, type=float, default=[4, 1, 1])
    parser.add_argument("--variants", nargs="+", choices=("global", "window"),
                        default=["global", "window"])
    args = parser.parse_args()
    variants = validate_variants(args.variants)
    paths = sorted(args.images.glob("*.png"))[:args.limit]
    if args.limit < 1 or len(paths) != args.limit:
        parser.error("image count does not match --limit")
    if not args.qualities or len(set(args.qualities)) != len(args.qualities):
        parser.error("qualities must be nonempty and unique")
    if any(q not in range(6) for q in args.qualities):
        parser.error("quality indices must be 0..5")
    if args.output.exists():
        raise FileExistsError("use a new output root; existing results are never overwritten")

    # Establish this inference process on its selected GPU before CPU weight I/O.
    torch.cuda.set_device(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    contract = checkpoint["e2e_variable_rate_contract"]
    expected = dict(zip(("pixel", "perceptual", "dists"), args.expected_weights))
    if checkpoint["train_steps"] != args.expected_step or contract["distortion_weights"] != expected:
        raise ValueError("checkpoint step or distortion weights do not match the requested model")
    config = prepare_inference_config(
        load_pure_sana_config(str(args.config)), codec_path=str(args.checkpoint), use_ema=True,
    )
    loss = config["model"]["params"]["loss_params"]
    actual = dict(zip(expected, (loss["pixelloss_weight"], loss["perceptual_weight"], loss["dists_weight"])))
    if actual != expected or list(config["data"]["path_size"]) != contract["training_crop"]:
        raise ValueError("config and checkpoint training contract differ")
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config.resolve()), "config_sha256": sha256_file(args.config),
        "checkpoint_step": checkpoint["train_steps"], "use_ema": True,
        "distortion_weights": expected, "training_contract": contract,
        "dataset": str(args.images.resolve()), "image_names": [p.name for p in paths],
        "qualities": args.qualities, "entropy_estimation": False,
        "eval_crop": "none; original full image", "padding": "reflect bottom/right to multiple of 256",
        "bitstream_root": str((args.output / "packets").resolve()),
        "comparison_contract": "both decoders receive the exact same serialized packet; all header bytes billed",
        "window_width": config["model"]["params"]["window_width"],
        "window_stride": config["model"]["params"]["window_stride"],
        "decoder_variants": list(variants),
    }
    del checkpoint
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    import pyiqa
    import torch.nn.functional as F
    from PIL import Image
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
    from torchvision.transforms.functional import to_pil_image, to_tensor
    from models.DiT_IC import Codec
    from models.local_context_decode import decode_local_context

    torch.manual_seed(903)
    net = Codec("cuda", "Codec", **config["model"]["params"]).cuda().eval().requires_grad_(False)
    assert_plain_e2e(net, config["model"]["params"])
    net.codec.update(force=True)
    metrics = {name: pyiqa.create_metric(name).cuda().eval().requires_grad_(False)
               for name in ("psnr", "ms_ssim", "dists")}
    metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    metrics["lpips"].requires_grad_(False)
    summaries = {variant: [] for variant in variants}
    with torch.no_grad():
        for quality in args.qualities:
            records = {variant: [] for variant in variants}
            for index, path in enumerate(paths, 1):
                with Image.open(path) as image:
                    original = to_tensor(image.convert("RGB"))[None].cuda()
                height, width = original.shape[-2:]
                padded = F.pad(original * 2 - 1, (0, (-width) % 256, 0, (-height) % 256), mode="reflect")
                packet, encode_seconds = timed(lambda: net.compress(padded, quality))
                packet_path = args.output / "packets" / f"q{quality}" / "bin" / path.stem
                received = save_received_packet(packet, packet_path, quality)
                bpp = packet_path.stat().st_size * 8 / (height * width)
                for variant in variants:
                    torch.cuda.reset_peak_memory_stats()
                    if variant == "global":
                        output, decode_seconds = timed(lambda: net.decompress(received))
                        detail = {"rans_decodes": 1, "dit_batched_calls": 1, "dit_forward_equivalents": 1}
                    else:
                        (output, detail), decode_seconds = timed(lambda: decode_local_context(net, received))
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    reconstruction = ((output[..., :height, :width] + 1) * 0.5).clamp(0, 1)
                    if reconstruction.shape != original.shape or not torch.isfinite(reconstruction).all():
                        raise ValueError(f"invalid {variant} reconstruction of {path.name}")
                    row = {
                        "image_name": path.stem, "quality_index": quality, "bpp": bpp,
                        "stream_bytes": packet_path.stat().st_size, "packet_sha256": sha256_file(packet_path),
                        "height": height, "width": width, "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds, "image_to_image_seconds": encode_seconds + decode_seconds,
                        "peak_allocated_gib": peak, **detail,
                    }
                    for name, metric in metrics.items():
                        row[name] = float(metric(reconstruction, original).item())
                        if not math.isfinite(row[name]):
                            raise ValueError(f"non-finite {name} metric")
                        if name == "lpips":
                            metric.reset()
                    location = args.output / variant / f"q{quality}"
                    (location / "rec").mkdir(parents=True, exist_ok=True)
                    to_pil_image(reconstruction[0].cpu()).save(location / "rec" / path.name)
                    records[variant].append(row)
                    atomic_json(location / "metrics" / "realrans" / "per_image_metrics.json", records[variant])
                    del output, reconstruction
                quality_text = " ".join(
                    f"{variant}={records[variant][-1]['psnr']:.3f}" for variant in variants
                )
                print(f"q{quality} {index:02d}/{len(paths)} {path.name} bpp={bpp:.6f} "
                      f"PSNR {quality_text}", flush=True)
            for variant, rows in records.items():
                summary = {**manifest, "variant": variant, "quality_index": quality, "num_images": len(rows),
                           "avg_bpp": sum(r["bpp"] for r in rows) / len(rows)}
                for name in (*metrics, "encode_seconds", "decode_seconds", "image_to_image_seconds", "peak_allocated_gib"):
                    summary[name] = sum(r[name] for r in rows) / len(rows)
                atomic_json(args.output / variant / f"q{quality}" / "metrics" / "realrans" / "summary_metrics.json", summary)
                summaries[variant].append(summary)
                print(f"SUMMARY q{quality} {variant}: " + json.dumps({k: summary[k] for k in ("avg_bpp", *metrics)}), flush=True)
    atomic_json(args.output / "all_summaries.json", summaries)
    print("DONE: full-image real-rANS inference completed for " + ", ".join(variants), flush=True)


if __name__ == "__main__":
    main()
