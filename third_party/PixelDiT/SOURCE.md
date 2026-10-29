# PixelDiT Reference

Upstream: https://github.com/NVlabs/PixelDiT

Revision: f7d7c35f71893d81c6e71d2576ea80601a1e8529

The PiT architecture, RMSNorm, Q/K rotary softmax attention, sinusoidal
embeddings and per-pixel modulation in models/bt_self_flow/pixel_adapter.py
are adapted from pixdit_core/modules.py and pixdit_core/pixeldit_c2i.py.
The upstream NVIDIA NSCLv1 license is reproduced in LICENSE in this directory.

Changes: AE hidden-feature input and residual output; SANA content projection;
effective quantization-scale conditions; two blocks with 256-dimensional
attention; activation checkpointing; explicit rectangular-grid interfaces;
zero output initialization with normally initialized internal gates.

No upstream weights or large Patch DiT backbone are loaded.
