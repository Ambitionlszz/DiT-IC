#!/usr/bin/env python3
"""Count the implemented D using fake CPU tensors, without weights or GPU work."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from models.bt_self_flow.pixel_adapter import AEConditionedHybridPixelAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=512, help="Image height after native padding")
    parser.add_argument("--width", type=int, default=512, help="Image width after native padding")
    args = parser.parse_args()
    height, width = args.height, args.width
    if min(height, width) <= 0 or height % 256 or width % 256:
        parser.error("native padded image dimensions must be positive multiples of 256")
    # The math backend exposes QK/AV contractions to FlopCounterMode. The CPU
    # fused SDPA operator is not counted by this installed PyTorch version.
    # This affects profiling only, not the adapter's execution/backend policy.
    with FakeTensorMode(), torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        adapter = AEConditionedHybridPixelAdapter(checkpoint_blocks=False).eval()
        pixels = torch.empty(1, 128, height, width)
        content = torch.empty(1, 512, height // 8, width // 8)
        precision = torch.empty(1, 128, height // 64, width // 64)
        with FlopCounterMode(display=False) as counter:
            output = adapter(pixels, content, precision)
        parameters = sum(parameter.numel() for parameter in adapter.parameters())
        counted_flops = counter.get_total_flops()
    area = height * width
    analytical_mac = 46_280 * area + area ** 2 // 32
    assert parameters == 5_478_000
    assert counted_flops == 2 * analytical_mac
    print(json.dumps({
        "method": "ae_conditioned_hybrid_pit_v1",
        "output_shape": list(output.shape),
        "parameters": parameters,
        "main_MAC": analytical_mac,
        "GMAC": analytical_mac / 1e9,
        "counted_FLOPs": counted_flops,
        "scope": "D only; existing C/AE/precision-embedding forward excluded",
        "convention": "major convolution/matrix/attention contractions; 2 FLOPs per MAC",
        "excluded": "norm, activation, pooling, softmax, bias, RoPE, elementwise and layout operations",
        "runtime_or_quality_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
