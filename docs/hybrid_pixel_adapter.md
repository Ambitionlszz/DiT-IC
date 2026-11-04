# AE-conditioned hybrid PiT D

Decoder method: `ae_conditioned_hybrid_pit_v1`. This is an additive method;
existing `native` and precision mu-QK PiT v4 checkpoints and recipes retain
their original architecture. Endpoint Self-Flow, rate control, scheduler,
reconstruction losses, optimizer groups and numerical recovery rules are
unchanged. There are no new transmitted fields or inference teacher inputs.

## Deployment path and one condition

For a 512-square image, native C provides `mu [B,32,16,16]` and the actual
decoded quantization-step map. Native SANA repair and its original scheduler
produce the repaired latent. The current AE decoder runs once, capturing
`U [B,512,64,64]` immediately after `up_blocks[3]` and continuing to the final
pre-output-normalization feature `F [B,128,512,512]`.

```text
U -> average pool 2x2 -> Linear 512->256 -> LayerNorm ------------+
                                                               + -> SiLU -> c
delta -> C's existing log-step embedding -> Linear 128->256      |
                                         -> nearest alignment -+

F -> Linear 128->16 + pixel positions -> hybrid block -> hybrid block
  -> zero-initialized Linear 16->128 -> add original F -> original AE RGB head
```

Content projection is evaluated on the 32x32 region grid. Precision projection
is bias-free, zero-initialized, evaluated on its native coding grid, then
nearest-aligned to the same region grid. There is no extra entry FiLM, raw DiT
hidden condition, separate mu condition, or per-block precision projection.
The same `c [B,1024,256]` supplies both blocks' AdaLN and cross-attention K/V.
This reuses a multiscale student feature, not extra information or a target
available only during training.

U is captured whenever D is enabled, including SF-off, highest-quality,
evaluation and actual packet decompression. It remains live for ordinary RD
gradients. D reads neither the detached/normalized SF metric nor teacher U.

## Two blocks, each with one compact/expand pair

Each 16x16 region contains 256 pixels with width 16. One hybrid block performs:

```text
X -> LayerNorm + pixel AdaLN -> compact 4096->256 -> t
s = SelfAttention(LayerNorm(t))
a = CrossAttention(Q=LayerNorm(t+s), K=c, V=c)
X' = X + pixel_gate * expand_256_to_4096(s+a)
Xout = X' + mlp_gate * pixel_MLP(AdaLN(LayerNorm(X')))
```

SA and CA are sequential in region space, not two independent pixel blocks.
Only their combined update is expanded. Each attention has width 256, four
64-dimensional heads, global SDPA and the existing two-dimensional Q/K RoPE;
there is no extra Q/K RMSNorm. The main per-pixel MLP stays `16->64->16` with
GELU. Two blocks therefore retain four attention operations but only two
pixel compact/expand paths, two pixel MLPs and two six-way AdaLN generators.

The small AdaLN generator projects each region condition `256->64` once per
block. For each branch independently, the current normalized pixel state
passes through a 16-channel 3x3 depthwise convolution and bias-free `16->64`
projection. Add the broadcast region condition, apply SiLU, then project
`64->48` to pixel shift, scale and gate. The attention branch and MLP branch
share the condition projection only; the MLP uses the state after attention.
The convolution crosses region boundaries and only generates modulation
parameters; it does not replace the main pixel MLP.

All new D LayerNorm layers use `eps=1e-6`, affine scale/bias. The final output
projection is zero-initialized, the precision projection is zero-initialized,
and depthwise generator kernels start as identity. Other projections and
internal gates retain ordinary initialization. Initial output matches C;
internal task gradients become available after the zero output head updates.

The fixed recipe has 5,478,000 D parameters and approximately 14.280 GMAC
at 512x512, batch one, counting major linear/convolution/attention products.
This excludes normalization, activation, softmax and layout work and is not
a wall-time or peak-memory measurement. SF adds no deployed module/parameter;
its teacher, target prefix and selective backward remain training overhead.
The shared C precision embedding weights are reused. As in v4, the receiver
evaluates that embedding on the native grid for D; that separate embedding
call adds 0.001056768 GMAC per 512-square image outside the D-module budget.
No codec API, quantization arithmetic, or packet payload changes are needed.

Reproduce the parameter/MAC check on the actual implementation without
pretrained tensors or GPU allocation:

```bash
python scripts/audit_ae_hybrid_pit.py --height 512 --width 512
```

It uses fake CPU tensors and selects math SDPA only inside the profiling
context so all four attention operations are visible to PyTorch's counter.
Padded dimensions must be multiples of 256, matching the native codec.

## Matched experiment family

Use `configs/bt_self_flow/endpoint_sf_hybrid/` for the new C/D/SF factorial.
Its six variant names match the existing family. C, C+SF and all shared
settings inherit the original recipe; only D's architecture and output
namespace change. All groups must use the same calibrated pure-C
`backend_init.pt`, not a trained C+D checkpoint. Old D weights must not be
loaded as the new D.

```bash
python scripts/launch_endpoint_sf.py --family hybrid --variant c_d_sf \
  --init-checkpoint /absolute/path/common/backend_init.pt \
  --gpus 0,1 --seed 903 --run-root /absolute/path/endpoint_runs
```

This plans without CUDA allocation or output writes. Add `--preflight --launch`
for two engineering updates, or `--launch` for formal training. New runs live
under `endpoint_runs/hybrid/seed_903/{preflight,runs}/VARIANT`; source and shared
initialization identity are pinned under `endpoint_runs/hybrid/`. The old
launcher default remains `--family legacy` and keeps its old run layout.
Resuming a D run requires the exact decoder method and settings, so v4 and
hybrid D cannot be silently interchanged. The trainer additionally checks the
full optimization, dataset, assets and initialization contract.

SF is unchanged: compare raw live U from the actual student endpoint to the
target endpoint through the same current AE prefix, with the existing detached
student spatial scale. Only trainable SANA parameters receive extra SF
gradients. C, AE and D receive ordinary RD gradients. Reusing the same feature
location makes the paths consistent, but does not establish that D uses it or
that the new architecture improves RD. Verify actual total-bpp RD with matched
independently trained C, C+D, C+SF and C+D+SF runs before making that claim.

## Engineering validation, 2026-11-04

The endpoint/hybrid suites passed 91 checks, including both two-rank CPU/Gloo
checks with host loopback enabled. The numerical-recovery, attention, stable
perceptual, GAN contract and existing v4 suites passed another 60 checks;
four CUDA-only cases were skipped. The 16 new adapter tests include actual
packet/qmax decoding, FP32/BF16 checkpoint parity and a real endpoint-SF
backward comparison showing identical C/AE/D RD gradients with extra SANA
gradients. Independent review found no blocking issue.

The implementation retains the de13052 numerical fixes ported by 4a0e582,
including stable SANA attention, StableLPIPS and clipping/recovery. A dry-run
of `--family hybrid --variant c_d` accepted the calibrated pure-C initializer
`/data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt` (SHA256
`7286c89f56637c0c94e5d3cb8fb34801f849a44816a8977a5c579973b11efaa8`).
It neither launched a worker nor created an output run. Full-size CUDA
forward/backward, production memory/throughput and trained RD remain untested.
