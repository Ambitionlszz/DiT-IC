"""Saved-tensor CPU storage that preserves activation-checkpoint input strides."""

from contextlib import contextmanager

import torch


def pack_preserving_layout(tensor, pin_memory=True):
    if tensor.layout != torch.strided:
        return None, tensor.detach()
    saved = torch.empty_like(tensor, device="cpu", pin_memory=pin_memory,
                             memory_format=torch.preserve_format)
    # Expanded/sliced views may not admit a dense copy with the same strides.
    # Keep those views on their original device instead of changing recompute.
    if saved.stride() != tensor.stride():
        return None, tensor.detach()
    saved.copy_(tensor, non_blocking=pin_memory)
    return tensor.device, saved


def unpack_preserving_layout(packed):
    device, tensor = packed
    if device is None:
        return tensor
    return tensor.to(device, non_blocking=tensor.is_pinned(), memory_format=torch.preserve_format)


@contextmanager
def preserve_stride_cpu_offload(pin_memory=True):
    """Store dense activations on CPU, retaining exact layouts for recompute.

    torch.save_on_cpu(pin_memory=True) allocates ordinary contiguous buffers;
    losing channels-last layout can change Linear's saved backward tensors.
    """
    def pack(tensor):
        return pack_preserving_layout(tensor, pin_memory)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack_preserving_layout):
        yield
