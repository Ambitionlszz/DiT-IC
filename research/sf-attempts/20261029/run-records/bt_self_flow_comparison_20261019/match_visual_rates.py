"""Match selected visual examples to baseline's measured per-image q2 bitrate."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "/data2/ls/code/DiT-IC-bt-self-flow")
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.experiment import make_model
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import BTImagePacket, RateControl

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("C", "D"))
    args = parser.parse_args()
    group = args.group
    checkpoint = read_checkpoint(ROOT / "checkpoints" / f"{group}_34000.pt")
    model = make_model(checkpoint["config"], torch.device("cuda"))
    model.load_bt_state(checkpoint["ema"])
    model.eval().requires_grad_(False)
    model.codec.update(force=True)
    metric = ImageLoss().cuda().eval()
    baseline = json.loads((ROOT / "baseline/global/q2/metrics/realrans/per_image_metrics.json").read_text())
    directory = ROOT / "matched_visuals" / group
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for stem in ("kodim01", "kodim13", "kodim23"):
        reference = next(row for row in baseline if row["image_name"] == stem)
        target = reference["bpp"]
        image = to_tensor(Image.open(Path("/data2/ls/data/Kodak") / f"{stem}.png").convert("RGB"))[None].cuda()*2-1
        height, width = image.shape[-2:]
        lower, upper = 0., 1.
        best = None
        for iteration in range(16):
            quality = lower if iteration == 0 else upper if iteration == 1 else (lower+upper)*.5
            packet = model.compress(image, RateControl(quality))
            data = packet.to_bytes()
            bpp = len(data)*8/(height*width)
            error = abs(bpp-target)/target
            if best is None or error < best[0]:
                best = (error, data, bpp, quality)
            if error <= .005:
                break
            if iteration >= 2:
                if bpp < target:
                    lower = quality
                else:
                    upper = quality
        error, data, bpp, quality = best
        received = BTImagePacket.from_bytes(data)
        before = model.prediction_calls
        decoded = model.decompress(received)
        assert model.prediction_calls-before == 1
        _, values = metric(decoded, image)
        packet_path = directory / f"{stem}.btic"
        packet_path.write_bytes(data)
        png_path = packet_path.with_suffix(".png")
        array = decoded[0].float().clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).cpu().numpy()
        Image.fromarray(array).save(png_path)
        results[stem] = {"target_bpp": target, "bpp": bpp, "relative_rate_error": error,
                         "lambda": quality, "iterations": iteration+1, "png": str(png_path),
                         "nfe": 1, **{k: float(values[k]) for k in ("psnr", "lpips", "dists")}}
        print(json.dumps(results[stem]), flush=True)
    (directory / "summary.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
