"""Fixed independent validation and training-only distortion calibration."""

import json
from pathlib import Path

import torch

from .checkpoint import evaluation_state
from .data import counter_seed
from .losses import reconstruction_objective
from .packet import BTImagePacket, RateControl, fixed_side_bits


@torch.no_grad()
def initialize_data_scale(model, dataset, samples=128, seed=903):
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for index in range(samples):
        image = dataset[(index % len(dataset), counter_seed(seed, index, 701))].unsqueeze(0).to(device)
        total += model.encode_clean_latent(image).square().mean()
    model.distortion_scale.s_data.copy_((total/samples).sqrt().clamp_min(1e-8))


@torch.no_grad()
def calibrate_interface(model, dataset, output, samples=512, seed=903, packet_samples=4):
    scales, rates = [], {str(q): [] for q in (0., .2, .4, .6, .8, 1.)}
    device = next(model.parameters()).device
    with evaluation_state(model):
        for index in range(samples):
            image = dataset[(index % len(dataset), counter_seed(seed, index, 702))].unsqueeze(0).to(device)
            clean = model.encode_clean_latent(image)
            analysis = model.codec.analyze(clean)
            for quality in (0., .2, .4, .6, .8, 1.):
                control = RateControl(quality)
                view = model.codec.quantize_analysis(analysis, control)
                scales.append(model.distortion_scale.measure(clean, view["mean"]).cpu())
                if index < packet_samples:
                    encoded = model.codec.compress(clean, control)
                    packet = BTImagePacket("C", tuple(image.shape[-2:]), encoded["y_shape"], encoded["hyper_shape"],
                                           control.low_code, control.high_code, encoded["hyper_stream"],
                                           encoded["main_stream"], encoded["allocation"])
                    rates[str(quality)].append(8*len(packet.to_bytes())/(image.shape[-2]*image.shape[-1]))
        model.distortion_scale.calibrate(model.distortion_scale.s_data.square(), scales)
    rates = {key: sum(values)/len(values) for key, values in rates.items()}
    report = {"samples": samples, "seed": seed, "dataset_fingerprint": dataset.fingerprint,
              "s_data": model.distortion_scale.s_data.item(),
              "log_min": model.distortion_scale.log_min.item(), "log_max": model.distortion_scale.log_max.item(),
              "range_policy": "saturate", "bits": 4, "prefix_raw_codec_actual_bpp": rates,
              "rate_span_pass": len({round(v, 9) for v in rates.values()}) >= 3}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


@torch.no_grad()
def validate(model, dataset, image_loss=None, state=None, transition=1., packet_samples=0, qualities=None):
    device = next(model.parameters()).device
    qualities = qualities or (0., .2, .4, .6, .8, 1.)
    report = {}
    with evaluation_state(model, state):
        for quality in qualities:
            control, sums = RateControl(quality), {}
            for index in range(len(dataset)):
                image = dataset[index].unsqueeze(0).to(device)
                result = model(image, rate_control=control, transition=transition)
                loss, metrics = reconstruction_objective(result, image, control, model.distortion_scale.s_data, image_loss)
                metrics["loss"] = loss
                if result.scale_info:
                    metrics["scale_saturation"] = result.scale_info["saturation"]
                for name, value in metrics.items():
                    sums[name] = sums.get(name, 0.)+float(value)
                if index < packet_samples and hasattr(model, "DiT") and transition == 1:
                    packet = model.compress(image, control)
                    decoded = model.decompress(packet)
                    hard_error = (decoded-result.image).abs().max().item()
                    if hard_error > 2e-4:
                        raise RuntimeError(f"packet and hard eval reconstruction differ: {hard_error}")
                    sums["packet_bpp"] = sums.get("packet_bpp", 0.)+8*len(packet.to_bytes())/(image.shape[-2]*image.shape[-1])
            report[str(quality)] = {name: value/(min(packet_samples, len(dataset)) if name == "packet_bpp" else len(dataset))
                                    for name, value in sums.items()}
    return report
