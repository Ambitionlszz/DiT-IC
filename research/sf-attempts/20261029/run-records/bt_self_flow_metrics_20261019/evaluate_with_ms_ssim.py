"""Real-packet evaluation with the repository's metrics plus MS-SSIM.

Adds MS-SSIM only in this evaluation process. No repository or training change.
All metrics use the same float reconstruction; MS-SSIM uses pyiqa's default
Y-channel protocol, matching compress.py. Run after resuming training, with a
bounded GPU allocator; MS-SSIM itself runs on the CPU.
"""

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path("/data2/ls/code/DiT-IC-bt-self-flow")
sys.path.insert(0, str(REPO))

import torch
import pyiqa
import eval_bt_self_flow as evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.21)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "images.jsonl").exists():
        parser.error("use a new output directory; refusing to combine evaluations")
    torch.set_num_threads(2 if args.device == "cuda" else 6)
    # C trainer's recorded peak leaves over 8 GiB on this host. Refuse to
    # compete when that margin is absent; the allocator cannot exceed 6.72 GiB.
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        budget = total * args.gpu_memory_fraction
        if free < budget + 1280 * 1024**2:
            raise RuntimeError(f"insufficient GPU headroom: free={free}, budget={budget}")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(json.dumps(dict(gpu_headroom_bytes=free, gpu_total_bytes=total,
                              allocator_limit_fraction=args.gpu_memory_fraction,
                              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))), flush=True)
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU evaluation must hide CUDA devices"
        print(json.dumps(dict(device="cpu", threads=6, gpu_allocation=False)), flush=True)
    ms_ssim = pyiqa.create_metric("ms_ssim", device="cpu").eval()
    original = evaluation.ImageLoss

    class EvaluationImageLoss(original):
        def forward(self, predicted, target):
            loss, values = super().forward(predicted, target)
            x = predicted.detach().float().cpu().add(1).mul(.5).clamp(0, 1)
            y = target.detach().float().cpu().add(1).mul(.5).clamp(0, 1)
            value = ms_ssim(x, y).mean()
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite MS-SSIM")
            values["ms_ssim"] = value
            # Release cached inference activations before the next image so
            # the resumed trainer retains the rest of the GPU's capacity.
            if args.device == "cuda":
                torch.cuda.empty_cache()
            return loss, values

    evaluation.ImageLoss = EvaluationImageLoss
    evaluation.evaluate(argparse.Namespace(
        checkpoint=args.checkpoint, input=["/data2/ls/data/Kodak"],
        output=str(output), decode_packet=None, weights="ema",
        qualities=[0., .2, .4, .6, .8, 1.], limit=args.limit,
        device=args.device, save_images=True, check_forward=False,
    ))
    rows = [json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "summary.json").read_text())
    for code, point in manifest["summary"].items():
        subset = [row for row in rows if row["lambda_code"] == int(code)]
        assert len(subset) == args.limit and len({row["image"] for row in subset}) == args.limit
        point["ms_ssim"] = sum(row["ms_ssim"] for row in subset) / len(subset)
    manifest["ms_ssim_protocol"] = {
        "implementation": "pyiqa.create_metric('ms_ssim')", "pyiqa_version": pyiqa.__version__,
        "test_y_channel": ms_ssim.net.test_y_channel, "color_space": ms_ssim.net.color_space,
        "range": "float RGB inputs mapped to [0,1], pyiqa default luminance conversion",
        "device": "cpu", "same_float_reconstruction_as_other_metrics": True,
    }
    manifest["evaluation_device"] = args.device
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(completed=True, step=manifest["step"], summary=manifest["summary"])), flush=True)


if __name__ == "__main__":
    main()
