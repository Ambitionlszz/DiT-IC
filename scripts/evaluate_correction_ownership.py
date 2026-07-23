"""Screen whether SANA corrections are image-specific enough for ISCC/BGCC.

The probe uses the real rANS path to obtain each decoded latent ``y_b``.  It
then compares four reconstructions:

* base: VAE decode of ``y_b`` without SANA;
* correct: ``y_b`` plus its own one-step SANA correction;
* random: ``y_b`` plus a norm-matched correction from a random image;
* hard: ``y_b`` plus a norm-matched correction from the most similar other
  image in the batch, selected by a frozen SigLIP2 encoder.

No model weights are changed.  The script writes per-image measurements and a
conservative summary; it does not claim that the training idea works.

Example:

    python scripts/evaluate_correction_ownership.py \
      --config configs/pure_sana_train.yaml \
      --checkpoint /path/to/pure_checkpoint.pt \
      --image_dir /path/to/validation/images \
      --siglip_path /path/to/siglip2-base-patch16-224 \
      --output_dir results/correction_ownership
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def derangement_indices(count, seed):
    """Return a deterministic permutation with no fixed positions."""
    if count < 2:
        raise ValueError("A correction permutation needs at least two images")
    if count == 2:
        return [1, 0]

    rng = random.Random(seed)
    values = list(range(count))
    for _ in range(100):
        rng.shuffle(values)
        if all(index != value for index, value in enumerate(values)):
            return list(values)
    return values[1:] + values[:1]


def _rankdata(values):
    """Assign average ranks, including ties, using only the standard library."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    return ranks


def spearman_correlation(left, right):
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if len(left) < 2:
        return None

    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def _rate(rows, predicate):
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize_rows(rows, min_images=64):
    """Summarize correction ownership and emit a conservative verdict."""
    if not rows:
        raise ValueError("No per-image rows were provided")

    vlm_hard_gaps = [
        float(row["vlm_hard"]) - float(row["vlm_correct"])
        for row in rows
    ]
    lpips_hard_gaps = [
        float(row["lpips_hard"]) - float(row["lpips_correct"])
        for row in rows
    ]
    dists_hard_gaps = [
        float(row["dists_hard"]) - float(row["dists_correct"])
        for row in rows
    ]
    vlm_lpips_agreement = spearman_correlation(
        vlm_hard_gaps,
        lpips_hard_gaps,
    )
    vlm_dists_agreement = spearman_correlation(
        vlm_hard_gaps,
        dists_hard_gaps,
    )

    summary = {
        "num_images": len(rows),
        "mean_real_bpp": _mean(rows, "real_bpp"),
        "mean_lpips_gain_correct_vs_base": (
            _mean(rows, "lpips_base") - _mean(rows, "lpips_correct")
        ),
        "mean_dists_gain_correct_vs_base": (
            _mean(rows, "dists_base") - _mean(rows, "dists_correct")
        ),
        "mean_vlm_gap_hard_minus_correct": sum(vlm_hard_gaps) / len(rows),
        "mean_lpips_gap_hard_minus_correct": (
            sum(lpips_hard_gaps) / len(rows)
        ),
        "mean_dists_gap_hard_minus_correct": (
            sum(dists_hard_gaps) / len(rows)
        ),
        "correct_wins_hard_vlm_rate": _rate(
            rows,
            lambda row: float(row["vlm_correct"]) < float(row["vlm_hard"]),
        ),
        "correct_wins_hard_lpips_rate": _rate(
            rows,
            lambda row: (
                float(row["lpips_correct"]) < float(row["lpips_hard"])
            ),
        ),
        "correct_wins_hard_dists_rate": _rate(
            rows,
            lambda row: (
                float(row["dists_correct"]) < float(row["dists_hard"])
            ),
        ),
        "hard_is_closer_than_random_vlm_rate": _rate(
            rows,
            lambda row: float(row["vlm_hard"]) < float(row["vlm_random"]),
        ),
        "spearman_vlm_gap_vs_lpips_gap": vlm_lpips_agreement,
        "spearman_vlm_gap_vs_dists_gap": vlm_dists_agreement,
        "suggested_margin_for_50pct_active": max(
            0.0,
            _median(vlm_hard_gaps),
        ),
    }

    reasons = []
    if len(rows) < min_images:
        verdict = "insufficient_sample"
        reasons.append(
            f"Need at least {min_images} images for a verdict; got {len(rows)}."
        )
    else:
        correction_improves_targets = (
            summary["mean_lpips_gain_correct_vs_base"] > 0.0
            and summary["mean_dists_gain_correct_vs_base"] > 0.0
        )
        hard_mining_works = (
            summary["hard_is_closer_than_random_vlm_rate"] >= 0.60
        )
        target_gaps_are_positive = (
            summary["mean_lpips_gap_hard_minus_correct"] > 0.0
            and summary["mean_dists_gap_hard_minus_correct"] > 0.0
        )
        target_win_rates_are_positive = (
            summary["correct_wins_hard_lpips_rate"] >= 0.55
            and summary["correct_wins_hard_dists_rate"] >= 0.55
        )
        agreement_is_positive = (
            vlm_lpips_agreement is not None
            and vlm_dists_agreement is not None
            and vlm_lpips_agreement >= 0.20
            and vlm_dists_agreement >= 0.20
        )

        if (
            correction_improves_targets
            and hard_mining_works
            and target_gaps_are_positive
            and target_win_rates_are_positive
            and agreement_is_positive
        ):
            verdict = "promising"
            reasons.append(
                "Correct corrections improve both target metrics, VLM hard "
                "mining is effective, and VLM ownership gaps agree with both "
                "LPIPS and DISTS gaps."
            )
        elif (
            not correction_improves_targets
            or (
                (vlm_lpips_agreement is not None and vlm_lpips_agreement <= 0.0)
                and (
                    vlm_dists_agreement is not None
                    and vlm_dists_agreement <= 0.0
                )
            )
        ):
            verdict = "not_supported"
            if not correction_improves_targets:
                reasons.append(
                    "The current SANA correction does not improve both LPIPS "
                    "and DISTS over the base reconstruction."
                )
            else:
                reasons.append(
                    "VLM ownership gaps do not agree with LPIPS or DISTS gaps."
                )
        else:
            verdict = "borderline"
            reasons.append(
                "Some ownership signal exists, but at least one required "
                "condition is weak. Inspect the saved images and per-image CSV."
            )

        if not hard_mining_works:
            reasons.append(
                "The VLM-selected negative is not consistently harder than a "
                "random negative."
            )
        if (
            summary["correct_wins_hard_vlm_rate"] >= 0.95
            and summary["correct_wins_hard_lpips_rate"] >= 0.95
            and summary["correct_wins_hard_dists_rate"] >= 0.95
        ):
            reasons.append(
                "Correct corrections beat hard negatives on nearly every "
                "image; negatives may be too easy for useful contrastive "
                "training."
            )

    summary["verdict"] = verdict
    summary["reasons"] = reasons
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether one-step SANA corrections are image-specific "
            "enough for correction-ownership contrastive training."
        )
    )
    parser.add_argument("--config", required=True, help="Pure SANA YAML config")
    parser.add_argument("--checkpoint", required=True, help="Pure checkpoint")
    parser.add_argument("--image_dir", required=True, help="Validation images")
    parser.add_argument(
        "--siglip_path",
        required=True,
        help=(
            "Local SigLIP2 checkpoint or Hugging Face model ID. A local path "
            "is recommended for offline server runs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="results/correction_ownership",
        help="Directory for CSV, JSON, and optional reconstructions",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=256)
    parser.add_argument("--min_images_for_verdict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true", default=None)
    parser.add_argument("--use_merge", action="store_true", default=None)
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save source/base/correct/random/hard images for visual checking",
    )
    parser.add_argument("--save_limit", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 3:
        parser.error("--batch_size must be at least 3")
    if args.image_size <= 0:
        parser.error("--image_size must be positive")
    if args.max_images < 3:
        parser.error("--max_images must be at least 3")
    return args


def _image_paths(image_dir, max_images, seed):
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(seed).shuffle(paths)
    if max_images:
        paths = paths[:max_images]
    return paths


def _nested_byte_count(value):
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_nested_byte_count(item) for item in value)
    return 0


def _match_correction_norm(source, target, epsilon=1e-8):
    source_norm = source.flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1)
    scale = target_norm / source_norm.clamp_min(epsilon)
    return source * scale[:, None, None, None]


def _to_pil_images(batch):
    from PIL import Image

    batch = (
        batch.detach()
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
    )
    return [
        Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB")
        for image in batch
    ]


def _vision_features(model, processor, images, device):
    import inspect
    import torch
    import torch.nn.functional as functional

    vision_model = getattr(model, "vision_model", model)
    processor_inputs = processor(
        images=_to_pil_images(images),
        return_tensors="pt",
    )
    forward_parameters = inspect.signature(vision_model.forward).parameters
    inputs = {}
    for name, value in processor_inputs.items():
        target_name = name
        if name not in forward_parameters:
            if (
                name == "pixel_attention_mask"
                and "attention_mask" in forward_parameters
            ):
                target_name = "attention_mask"
            else:
                continue
        inputs[target_name] = value.to(device)
    if "pixel_values" not in inputs:
        raise RuntimeError(
            "SigLIP2 processor/model interface did not provide pixel_values"
        )
    with torch.no_grad():
        outputs = vision_model(**inputs, return_dict=True)
    dense = functional.normalize(outputs.last_hidden_state.float(), dim=-1)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = dense.mean(dim=1)
    pooled = functional.normalize(pooled.float(), dim=-1)
    return dense, pooled


def _dense_distances(reference, candidates):
    return (1.0 - (reference * candidates).sum(dim=-1)).mean(dim=1)


def _hard_negative_indices(global_features):
    import torch

    similarities = global_features @ global_features.transpose(0, 1)
    similarities.fill_diagonal_(-torch.inf)
    return similarities.argmax(dim=1)


def _decode_latent(model, latent):
    return model.vae.decode(
        latent / model.vae.config.scaling_factor,
        return_dict=False,
    )[0].clamp(-1.0, 1.0)


def _pure_sana_endpoint(model, latent):
    condition = model._fixed_text_condition(latent.shape[0], latent.device)
    timestep = model.sched.base_scheduler.timesteps.expand(latent.shape[0])
    timestep = timestep * model.DiT.config.timestep_scale
    prediction = model.DiT(
        latent,
        encoder_hidden_states=condition,
        timestep=timestep.to(latent.device),
        return_dict=False,
    )[0]
    return model.sched.step(prediction, latent)


def _paired_metric_values(lpips_metric, dists_metric, reference, candidate):
    import torch

    with torch.no_grad():
        lpips_values = (
            lpips_metric(reference, candidate)
            .reshape(reference.shape[0], -1)
            .mean(dim=1)
        )
        reference_01 = reference.add(1.0).div(2.0).clamp(0.0, 1.0)
        candidate_01 = candidate.add(1.0).div(2.0).clamp(0.0, 1.0)
        dists_values = []
        for index in range(reference.shape[0]):
            value = dists_metric(
                reference_01[index : index + 1],
                candidate_01[index : index + 1],
            )
            dists_values.append(float(value.reshape(-1).mean().item()))
    return lpips_values.cpu().tolist(), dists_values


def _save_tensor_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil_images(image.unsqueeze(0))[0].save(path)


def _load_image_batch(paths, transform):
    from PIL import Image
    import torch

    images = []
    valid_paths = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
            valid_paths.append(path)
        except Exception as error:
            print(f"Skipping unreadable image {path}: {error}")
    if not images:
        return None, []
    return torch.stack(images), valid_paths


def run_probe(args):
    try:
        import lpips
        import pyiqa
        import torch
        import yaml
        from torchvision import transforms
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "The probe requires torch, torchvision, transformers, lpips, "
            "pyiqa, Pillow, and PyYAML in the server environment."
        ) from error

    from config_compat import prepare_inference_config
    from models.DiT_IC import Codec

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA for the Pure SANA model")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = prepare_inference_config(
        config,
        codec_path=str(Path(args.checkpoint).resolve()),
        use_ema=args.use_ema,
        use_merge=args.use_merge,
    )
    params = config["model"]["params"]
    if params.get("decode_mode", "sana") != "sana":
        raise ValueError("Correction ownership requires decode_mode='sana'")
    meanflow = dict(params.get("meanflow_params") or {})
    if meanflow.get("enabled") or meanflow.get("mode") == "meanflow":
        raise ValueError(
            "Run the first correction-ownership probe on Pure SANA with "
            "MeanFlow disabled"
        )
    params["freeze_codec"] = True

    print("Loading Pure SANA checkpoint...")
    model = Codec(
        device="cuda",
        model_type=config["model"]["model_type"],
        **params,
    ).to(device).eval()
    model.codec.update(force=True)

    print("Loading frozen SigLIP2 encoder...")
    processor = AutoProcessor.from_pretrained(args.siglip_path)
    vision_encoder = AutoModel.from_pretrained(args.siglip_path).to(device).eval()
    vision_encoder.requires_grad_(False)

    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    lpips_metric.requires_grad_(False)
    dists_metric = pyiqa.create_metric("dists", as_loss=True).to(device).eval()
    dists_metric.requires_grad_(False)

    transform = transforms.Compose(
        [
            transforms.Resize(
                args.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    paths = _image_paths(args.image_dir, args.max_images, args.seed)
    if len(paths) < 3:
        raise RuntimeError("The image directory contains fewer than 3 images")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = 0

    for batch_start in range(0, len(paths), args.batch_size):
        batch_paths = paths[batch_start : batch_start + args.batch_size]
        images, batch_paths = _load_image_batch(batch_paths, transform)
        if images is None or len(batch_paths) < 3:
            print(f"Skipping final/invalid batch at index {batch_start}")
            continue
        images = images.to(device)

        decoded_latents = []
        real_bpps = []
        with torch.no_grad():
            for image in images:
                compressed = model.compress(image.unsqueeze(0))
                decoded = model.codec.decompress(
                    compressed["strings"],
                    compressed["shape"],
                )
                decoded_latents.append(decoded.squeeze(0))
                byte_count = _nested_byte_count(compressed["strings"])
                real_bpps.append(
                    byte_count * 8.0 / (image.shape[-2] * image.shape[-1])
                )
            y_b = torch.stack(decoded_latents)
            y_p = _pure_sana_endpoint(model, y_b)
            corrections = y_p - y_b

            ref_dense, ref_global = _vision_features(
                vision_encoder,
                processor,
                images,
                device,
            )
            hard_indices = _hard_negative_indices(ref_global)
            random_indices = torch.tensor(
                derangement_indices(len(batch_paths), args.seed + batch_start),
                device=device,
            )

            random_corrections = _match_correction_norm(
                corrections.index_select(0, random_indices),
                corrections,
            )
            hard_corrections = _match_correction_norm(
                corrections.index_select(0, hard_indices),
                corrections,
            )

            base_images = _decode_latent(model, y_b)
            correct_images = _decode_latent(model, y_p)
            random_images = _decode_latent(model, y_b + random_corrections)
            hard_images = _decode_latent(model, y_b + hard_corrections)

            correct_dense, _ = _vision_features(
                vision_encoder,
                processor,
                correct_images,
                device,
            )
            random_dense, _ = _vision_features(
                vision_encoder,
                processor,
                random_images,
                device,
            )
            hard_dense, _ = _vision_features(
                vision_encoder,
                processor,
                hard_images,
                device,
            )
            vlm_correct = _dense_distances(ref_dense, correct_dense)
            vlm_random = _dense_distances(ref_dense, random_dense)
            vlm_hard = _dense_distances(ref_dense, hard_dense)

        metrics = {}
        for name, reconstruction in (
            ("base", base_images),
            ("correct", correct_images),
            ("random", random_images),
            ("hard", hard_images),
        ):
            lpips_values, dists_values = _paired_metric_values(
                lpips_metric,
                dists_metric,
                images,
                reconstruction,
            )
            metrics[name] = {
                "lpips": lpips_values,
                "dists": dists_values,
            }

        for index, path in enumerate(batch_paths):
            hard_path = batch_paths[int(hard_indices[index].item())]
            random_path = batch_paths[int(random_indices[index].item())]
            row = {
                "image": str(path),
                "hard_source": str(hard_path),
                "random_source": str(random_path),
                "real_bpp": real_bpps[index],
                "correction_norm": float(
                    corrections[index].flatten().norm().item()
                ),
                "vlm_correct": float(vlm_correct[index].item()),
                "vlm_random": float(vlm_random[index].item()),
                "vlm_hard": float(vlm_hard[index].item()),
            }
            for variant in ("base", "correct", "random", "hard"):
                row[f"lpips_{variant}"] = metrics[variant]["lpips"][index]
                row[f"dists_{variant}"] = metrics[variant]["dists"][index]
            rows.append(row)

            if args.save_images and saved < args.save_limit:
                image_id = f"{len(rows) - 1:04d}_{path.stem}"
                for variant, tensor in (
                    ("source", images[index]),
                    ("base", base_images[index]),
                    ("correct", correct_images[index]),
                    ("random", random_images[index]),
                    ("hard", hard_images[index]),
                ):
                    _save_tensor_image(
                        tensor,
                        output_dir / "images" / variant / f"{image_id}.png",
                    )
                saved += 1

        print(f"Processed {len(rows)} images")

    if not rows:
        raise RuntimeError("No complete batch with at least 3 valid images")

    csv_path = output_dir / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows, args.min_images_for_verdict)
    summary["config"] = str(Path(args.config).resolve())
    summary["checkpoint"] = str(Path(args.checkpoint).resolve())
    summary["siglip_path"] = args.siglip_path
    summary["image_dir"] = str(Path(args.image_dir).resolve())
    summary["seed"] = args.seed
    summary["batch_size"] = args.batch_size
    summary["image_size"] = args.image_size

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-image measurements: {csv_path}")
    print(f"Summary: {summary_path}")


def main():
    run_probe(parse_args())


if __name__ == "__main__":
    main()
