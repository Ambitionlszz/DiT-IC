"""Measure frozen encoder layout and EMA overhead without changing a run."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.bt_self_flow.checkpoint import DeploymentEMA
from models.bt_self_flow.model import BTCodecStage


def timed(function, count=5):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(count):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter()-started)/count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sana-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(903)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = BTCodecStage(args.sana_path, {"delta_coarse": 8., "delta_fine": 1.}).cuda().eval()
    image = torch.rand(args.batch_size, 3, 512, 512, device="cuda").mul(2).sub(1)
    reference = model.encode_clean_latent(image).detach().clone()
    baseline = timed(lambda: model.encode_clean_latent(image))
    model.vae.encoder.to(memory_format=torch.channels_last)
    actual = model.encode_clean_latent(image)
    channels_last = timed(lambda: model.encode_clean_latent(image))
    absolute = (reference-actual).abs().max().item()
    relative = (reference-actual).square().mean().sqrt().item()/reference.square().mean().sqrt().clamp_min(1e-12).item()
    old = DeploymentEMA(model)
    new = DeploymentEMA(model)

    @torch.no_grad()
    def original_update():
        for part, values in model.export_state().items():
            for name, source in values.items():
                target = old.state[part][name]
                if target.shape != source.shape:
                    old.state[part][name] = source.detach().clone()
                elif (part, name) in old.averaged:
                    target.lerp_(source.detach(), 1-old.decay)
                else:
                    target.copy_(source.detach())

    with torch.no_grad():
        for parameter in model.codec.parameters():
            parameter.add_(.001)
    old_seconds = timed(original_update, count=20)
    new_seconds = timed(lambda: new.update(model), count=20)
    ema_error = max((value.float()-new.state[part][name].float()).abs().max().item()
                    for part, values in old.state.items() for name, value in values.items() if value.numel())
    report = dict(device=torch.cuda.get_device_name(0), batch_size=args.batch_size, crop=512,
                  encoder=dict(contiguous_seconds=baseline, channels_last_seconds=channels_last,
                               speedup=baseline/channels_last, latent_max_abs_error=absolute,
                               latent_relative_rms_error=relative),
                  ema=dict(original_seconds=old_seconds, foreach_seconds=new_seconds,
                           speedup=old_seconds/new_seconds, max_abs_error=ema_error),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if ema_error != 0:
        raise RuntimeError("batched EMA changed update values")


if __name__ == "__main__":
    main()
