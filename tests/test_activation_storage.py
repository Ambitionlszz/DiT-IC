from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.bt_self_flow.activation_storage import (
    pack_preserving_layout, preserve_stride_cpu_offload, unpack_preserving_layout,
)


@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transposed", "expanded", "sliced"])
def test_saved_input_retains_exact_values_and_strides(layout):
    value = torch.randn(2, 4, 6, 8)
    if layout == "channels_last":
        value = value.to(memory_format=torch.channels_last)
    elif layout == "transposed":
        value = value.transpose(1, 3)
    elif layout == "expanded":
        value = value[:1].expand(3, -1, -1, -1)
    elif layout == "sliced":
        value = value[..., ::2]
    restored = unpack_preserving_layout(pack_preserving_layout(value, pin_memory=False))
    assert restored.stride() == value.stride()
    torch.testing.assert_close(restored, value, atol=0, rtol=0)


def test_bf16_channels_last_checkpoint_offload_matches_full_gradient():
    class DecoderStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1)
            self.base = nn.Linear(8, 8, bias=False).requires_grad_(False)
            self.lora = nn.Sequential(nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False))

        def forward(self, value):
            hidden = self.conv(value).movedim(1, -1)
            return (self.base(hidden) + self.lora(hidden)).tanh()

    reference = DecoderStage()
    checked = deepcopy(reference)
    source = torch.randn(2, 8, 8, 8).to(memory_format=torch.channels_last)
    results = []
    for module, recompute in ((reference, False), (checked, True)):
        value = source.detach().clone().requires_grad_()
        context = preserve_stride_cpu_offload(pin_memory=False) if recompute else nullcontext()
        with context:
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                output = checkpoint(module, value, use_reentrant=False) if recompute else module(value)
                loss = output.square().mean()
            loss.backward()
        results.append((output.detach(), value.grad))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(reference.parameters(), checked.parameters()):
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
