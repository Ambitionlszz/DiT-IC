"""Evaluate saved RGB images with explicit pairing and metric metadata.

Adapted from the MS-ILLM folder evaluation script. Paired evaluation retains
FID/256, which requires both image dimensions to be at least 256 pixels.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import tqdm

from evaluation_protocol import (
    dataset_content_fingerprint,
    metric_protocol,
    psnr,
    source_revision,
    structure_metrics,
)
from .image_pairs import collect_image_pairs
from .distribution_metrics import (
    FID_PATCH_SIZE,
    build_distribution_metrics,
    compute_distribution_metrics,
    distribution_metric_protocol,
)


def _build_metrics(device, paired):
    # Keep heavyweight dependencies and weight loading after input validation.
    import pyiqa

    no_reference = {name: pyiqa.create_metric(name).eval().to(device)
                    for name in ("clipiqa", "musiq", "niqe")}
    paired_metrics = {}
    fid_metric = kid_metric = update_patch_fid = None
    if paired:
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
        )
        from ._update_patch_fid import update_patch_fid

        paired_metrics = {"psnr": psnr, "dists": pyiqa.create_metric("dists").eval().to(device)}
        paired_metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True).eval().to(device)
        fid_metric, kid_metric = build_distribution_metrics(device)
    return no_reference, paired_metrics, fid_metric, kid_metric, update_patch_fid


def _read_tensor(path, device):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


@torch.inference_mode()
def evaluate(recon_dir, gt_dir=None, ntest=None, *, strip_recon_index_prefix=False,
             output_json=None, device="cuda"):
    """Return the historical list of printable results; optionally save JSON.

    Saved images are evaluated after their RGB uint8 decode. ``ms_ssim`` now
    aliases RGB MS-SSIM; the historical Y-channel value is ``ms_ssim_y``.
    """
    pairs = collect_image_pairs(recon_dir, gt_dir, ntest,
                                strip_recon_index_prefix=strip_recon_index_prefix)
    if gt_dir is not None:
        for pair in pairs:
            if min(pair.size) < FID_PATCH_SIZE:
                raise ValueError(f"Paired folder evaluation includes FID/256 and requires images of at least "
                                 f"256 x 256 pixels; {pair.reconstruction} has size {pair.size}")
    device = torch.device(device)
    metric_dict, metric_paired_dict, fid_metric, kid_metric, update_patch_fid = _build_metrics(
        device, gt_dir is not None)

    print(f"Find {len(pairs)} images in {Path(recon_dir)}")
    result = {}
    per_image = []
    patch_count = 0
    for pair in tqdm.tqdm(pairs):
        recon_tensor = _read_tensor(pair.reconstruction, device)
        values = {key: float(metric(recon_tensor).item()) for key, metric in metric_dict.items()}
        if pair.ground_truth is not None:
            gt_tensor = _read_tensor(pair.ground_truth, device)
            patch_count += update_patch_fid(gt_tensor, recon_tensor,
                                            fid_metric=fid_metric, kid_metric=kid_metric, patch_size=FID_PATCH_SIZE)
            values.update({key: float(metric(recon_tensor, gt_tensor).item())
                           for key, metric in metric_paired_dict.items()})
            values.update({key: float(value.item())
                           for key, value in structure_metrics(recon_tensor, gt_tensor).items()})
        print(pair.reconstruction)
        for key, value in values.items():
            print(key, value)
            result[key] = result.get(key, 0.0) + value
        per_image.append({"reconstruction": str(pair.reconstruction.resolve()),
                          "ground_truth": str(pair.ground_truth.resolve()) if pair.ground_truth else None,
                          "width": pair.size[0], "height": pair.size[1], **values})

    # Ordinary image metrics use an equal weight per image, regardless of size.
    result = {key: total / len(pairs) for key, total in result.items()}
    distribution_values, unavailable = compute_distribution_metrics(fid_metric, kid_metric, patch_count)
    result.update(distribution_values)

    print_results = []
    for key, value in result.items():
        digits = 2 if key == "fid" else 7 if key in ("kid_mean", "kid_std") else 5
        line = f"{key}: {value:.{digits}f}"
        print(line)
        print_results.append(line)
    for key, reason in unavailable.items():
        print(f"{key}: unavailable ({reason})")

    protocol = metric_protocol(pixel_domain="uint8_roundtrip", ms_ssim_alias="rgb")
    protocol.update({"quantization": "decoded stored RGB uint8 / 255; producer rounding is unknown",
                     "pairing": "exact_case_sensitive_stem",
                     "strip_recon_index_prefix": strip_recon_index_prefix,
                     **distribution_metric_protocol(patch_count),
                     "unavailable_metrics": unavailable})
    print("Metric protocol:", json.dumps(protocol, sort_keys=True))
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"num_images": len(pairs), "metrics": result,
                                          "metric_protocol": protocol, "images": per_image,
                                          "source_revision": source_revision(),
                                          "dataset_fingerprints": {
                                              "reconstruction": dataset_content_fingerprint(
                                                  pair.reconstruction for pair in pairs),
                                              "ground_truth": dataset_content_fingerprint(
                                                  pair.ground_truth for pair in pairs)
                                              if gt_dir is not None else None}},
                                         indent=2) + "\n")
    return print_results


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon_dir", "--recon-dir", type=str, required=True)
    parser.add_argument("--gt_dir", "--gt-dir", type=str)
    parser.add_argument("--ntest", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip-recon-index-prefix", action="store_true",
                        help="Explicitly map 00000_source.png reconstructions to source.* targets.")
    parser.add_argument("--output-json", type=str,
                        help="Save means, per-image source identities, and the evaluation protocol.")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    return evaluate(args.recon_dir, args.gt_dir, args.ntest,
                    strip_recon_index_prefix=args.strip_recon_index_prefix,
                    output_json=args.output_json, device=args.device)


if __name__ == "__main__":
    main(sys.argv[1:])
