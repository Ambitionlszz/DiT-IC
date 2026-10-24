"""Fixed independent validation and training-only distortion calibration."""

import json
import math
from pathlib import Path

import torch

from .checkpoint import evaluation_state
from .data import counter_seed
from .losses import rate_beta, reconstruction_objective
from .model import pad_image
from .packet import BTImagePacket, RateControl, fixed_side_bits, pack_codes


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
            analysis = None if model.codec.precision_conditioning else model.codec.analyze(clean)
            for quality in (0., .2, .4, .6, .8, 1.):
                control = RateControl(quality)
                source = model.codec.analyze(clean, control) if analysis is None else analysis
                view = model.codec.quantize_analysis(source, control)
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
             actual_rate=False, objective=None, quality=None):
    device = next(model.parameters()).device
    qualities = qualities or (0., .2, .4, .6, .8, 1.)
    sums_by_quality = {str(quality): {} for quality in qualities}
    with evaluation_state(model, state):
        for index in range(len(dataset)):
            image = dataset[index].unsqueeze(0).to(device)
            height, width = image.shape[-2:]
            clean = model.encode_clean_latent(pad_image(image))
            # Only the frozen AE is shared across q in precision v2.
            shared = None if model.codec.precision_conditioning else model.codec.analyze(clean)
            for q in qualities:
                control, sums = RateControl(q), sums_by_quality[str(q)]
                source = model.codec.analyze(clean, control) if shared is None else shared
                result = model.forward_analysis(source, rate_control=control, transition=transition)
                if result.image is not None:
                    result.image = result.image[..., :height, :width]
                loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data,
                                                         image_loss, objective=objective, quality=quality)
                if result.image is not None and min(height, width) > 160:
                    from pytorch_msssim import ms_ssim
                    metrics["ms_ssim"] = ms_ssim((result.image+1)/2, (image+1)/2, data_range=1., size_average=True)
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
                    loss = loss+rate_beta(control, quality)*(actual_bpp-metrics["rate_est"])
                    metrics["weighted_rate"] = rate_beta(control, quality)*actual_bpp
                    metrics["rate_actual"] = actual_bpp
                    for label, payload in (("main", packet.main_stream), ("hyper", packet.hyper_stream),
                                           ("allocation", packet.allocation), ("scales", packet.scales),
                                           ("mask", packet.mask)):
                        metrics[f"rate_{label}"] = loss.new_tensor(8*len(payload)/(height*width))
                    payload_bytes = sum(len(payload) for payload in (packet.main_stream, packet.hyper_stream,
                                        packet.allocation, packet.scales, packet.mask))
                    metrics["rate_header"] = loss.new_tensor(8*(len(packet.to_bytes())-payload_bytes)/(height*width))
                metrics["loss"] = loss
                if result.scale_info:
                    metrics["scale_saturation"] = result.scale_info["saturation"]
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.)+value.detach()
                if index < packet_samples and hasattr(model, "DiT") and transition == 1:
                    packet = packet if packet is not None else model.compress(image, control)
                    decoded = model.decompress(BTImagePacket.from_bytes(packet.to_bytes()))
                    hard_error = (decoded-result.image).abs().max().item()
                    if not math.isfinite(hard_error) or hard_error > 2e-4:
                        raise RuntimeError(f"packet and hard eval reconstruction differ: {hard_error}")
                    sums["packet_forward_max_error"] = max(sums.get("packet_forward_max_error", 0.), hard_error)
                    sums["packet_bpp"] = sums.get("packet_bpp", 0.)+8*len(packet.to_bytes())/(image.shape[-2]*image.shape[-1])
    return {quality: {name: float(value)/(1 if name == "packet_forward_max_error" else
                                        min(packet_samples, len(dataset)) if name == "packet_bpp" else len(dataset))
                      for name, value in sums.items()} for quality, sums in sums_by_quality.items()}
