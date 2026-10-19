"""Fixed independent validation and training-only distortion calibration."""

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Subset

from .checkpoint import evaluation_state
from .data import counter_seed
from .losses import rate_beta, reconstruction_objective
from .model import pad_image
from .packet import BTImagePacket, RateControl, fixed_side_bits, pack_codes


def merge_validation_reports(parts):
    result = {}
    qualities = set().union(*(part["report"] for part in parts))
    for quality in sorted(qualities, key=float):
        result[quality] = {}
        names = set().union(*(part["report"][quality] for part in parts))
        for name in sorted(names):
            if name == "packet_forward_max_error":
                value = max(part["report"][quality].get(name, 0.) for part in parts)
            else:
                count = "packets" if name == "packet_bpp" else "images"
                total = sum(part[count] for part in parts)
                value = sum(part["report"][quality].get(name, 0.)*part[count] for part in parts)/total
            result[quality][name] = value
    return result


def validate_distributed(model, dataset, *, group=None, packet_samples=0, **kwargs):
    """Evaluate disjoint images on each rank, preserving all six quality points."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    subset = Subset(dataset, range(rank, len(dataset), world))
    packets = min(len(subset), max(0, (packet_samples-rank+world-1)//world))
    local = dict(images=len(subset), packets=packets, error=None)
    try:
        local["report"] = validate(model, subset, packet_samples=packets, **kwargs)
    except Exception as error:
        local["error"] = f"rank {rank}: {type(error).__name__}: {error}"
    gathered = [None]*world
    dist.all_gather_object(gathered, local, group=group)
    errors = [part["error"] for part in gathered if part["error"]]
    if errors:
        raise RuntimeError("distributed validation failed: "+"; ".join(errors))
    return merge_validation_reports(gathered)


@torch.no_grad()
def initialize_data_scale(model, dataset, samples=128, seed=903, batch_size=2):
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for start in range(0, samples, batch_size):
        indices = range(start, min(start+batch_size, samples))
        image = torch.stack([dataset[(index % len(dataset), counter_seed(seed, index, 701))]
                             for index in indices]).to(device)
        total += model.encode_clean_latent(image).square().flatten(1).mean(1).sum()
    model.distortion_scale.s_data.copy_((total/samples).sqrt().clamp_min(1e-8))


@torch.no_grad()
def calibrate_interface(model, dataset, output, samples=512, seed=903, packet_samples=4):
    scales, rates = [], {str(q): [] for q in (0., .2, .4, .6, .8, 1.)}
    distortion = {quality: torch.zeros((), device=next(model.parameters()).device) for quality in rates}
    device = next(model.parameters()).device
    with evaluation_state(model):
        for index in range(samples):
            image = dataset[(index % len(dataset), counter_seed(seed, index, 702))].unsqueeze(0).to(device)
            clean = model.encode_clean_latent(image)
            analysis = model.codec.analyze(clean)
            for quality in (0., .2, .4, .6, .8, 1.):
                control = RateControl(quality)
                view = model.codec.quantize_analysis(analysis, control)
                distortion[str(quality)] += (view["mean"]-clean).square().mean()/model.distortion_scale.s_data.square()
                scales.append(model.distortion_scale.measure(clean, view["mean"]).cpu())
                if index < packet_samples:
                    encoded = model.codec.compress(clean, control)
                    packet = BTImagePacket("C", tuple(image.shape[-2:]), encoded["y_shape"], encoded["hyper_shape"],
                                           control.low_code, control.high_code, encoded["hyper_stream"],
                                           encoded["main_stream"], encoded["allocation"])
                    rates[str(quality)].append(8*len(packet.to_bytes())/(image.shape[-2]*image.shape[-1]))
        model.distortion_scale.calibrate(model.distortion_scale.s_data.square(), scales)
    rates = {key: sum(values)/len(values) for key, values in rates.items()}
    distortion = {key: value.item()/samples for key, value in distortion.items()}
    report = {"samples": samples, "seed": seed, "dataset_fingerprint": dataset.fingerprint,
              "s_data": model.distortion_scale.s_data.item(),
              "log_min": model.distortion_scale.log_min.item(), "log_max": model.distortion_scale.log_max.item(),
              "range_policy": "saturate", "bits": 4, "prefix_raw_codec_actual_bpp": rates,
              "raw_codec_latent_nmse": distortion,
              "quality_span_pass": distortion["1.0"] < distortion["0.0"] and rates["1.0"] > rates["0.0"],
              "rate_span_pass": len({round(v, 9) for v in rates.values()}) >= 3}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


@torch.no_grad()
def validate(model, dataset, image_loss=None, state=None, transition=1., packet_samples=0, qualities=None,
             actual_rate=False, rate_weight=1.):
    device = next(model.parameters()).device
    qualities = qualities or (0., .2, .4, .6, .8, 1.)
    sums_by_quality = {str(quality): {} for quality in qualities}
    with evaluation_state(model, state, deployment_router=True):
        for index in range(len(dataset)):
            image = dataset[index].unsqueeze(0).to(device)
            height, width = image.shape[-2:]
            # All quality points share the frozen clean encoder and pre-quant
            # analysis. Quantization, context and repair still run per quality.
            source = model.codec.analyze(model.encode_clean_latent(pad_image(image)))
            for quality in qualities:
                control, sums = RateControl(quality), sums_by_quality[str(quality)]
                result = model.forward_analysis(source, rate_control=control, transition=transition)
                if result.image is not None:
                    result.image = result.image[..., :height, :width]
                loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                                         image_loss, rate_weight=rate_weight)
                packet = None
                if actual_rate:
                    # Encode the frontend without a second SANA forward. D's
                    # scale side information still comes from actual decoded mu.
                    encoded = model.codec.compress(source.clean, control)
                    scales = b""
                    if model.backend == "D":
                        info = model.distortion_scale(source.clean, encoded["mean"])
                        scales = pack_codes(info["codes"].flatten().tolist(), 4)
                    packet = BTImagePacket(model.backend, (height, width), encoded["y_shape"], encoded["hyper_shape"],
                                           control.low_code, control.high_code, encoded["hyper_stream"],
                                           encoded["main_stream"], encoded["allocation"], scales)
                    actual_bpp = loss.new_tensor(8*len(packet.to_bytes())/(height*width))
                    loss = loss+rate_weight*rate_beta(control)*(actual_bpp-metrics["rate_est"])
                    metrics["rate_actual"] = actual_bpp
                metrics["loss"] = loss
                if result.scale_info:
                    metrics["scale_saturation"] = result.scale_info["saturation"]
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.)+value.detach()
                if index < packet_samples and hasattr(model, "DiT") and transition == 1:
                    packet = packet if packet is not None else model.compress(image, control)
                    decoded = model.decompress(packet)
                    hard_error = (decoded-result.image).abs().max().item()
                    if not math.isfinite(hard_error) or hard_error > 2e-4:
                        raise RuntimeError(f"packet and hard eval reconstruction differ: {hard_error}")
                    sums["packet_forward_max_error"] = max(sums.get("packet_forward_max_error", 0.), hard_error)
                    sums["packet_bpp"] = sums.get("packet_bpp", 0.)+8*len(packet.to_bytes())/(image.shape[-2]*image.shape[-1])
    return {quality: {name: float(value)/(1 if name == "packet_forward_max_error" else
                                        min(packet_samples, len(dataset)) if name == "packet_bpp" else len(dataset))
                      for name, value in sums.items()} for quality, sums in sums_by_quality.items()}
