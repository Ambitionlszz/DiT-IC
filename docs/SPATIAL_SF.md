# Spatial-response Self-Flow with independent C/D/SF ablations

Implementation: `cross_rate_spatial_response_self_flow_v1`.

The student receives one complete low-rate packet view. An identity-initialized
3×3 convolution P transforms SANA block-8 features; its raw output is the input
to block 9. The auxiliary target is block-20 features of a truncated, no-gradient
copy of the same deployment EMA, using a native high-rate codec forward.

For each image and channel, subtract the spatial mean and divide by the square
root of population spatial variance plus `1e-6`. Student and teacher use their
own statistics. The loss is the mean absolute difference of these standardized
responses. Student statistics remain differentiable. There is no token cosine,
channel-vector normalization, region selection, auxiliary weight calibration,
SF warmup or SF decay. The SF coefficient is exactly 1.

Only P and the trainable ancestors of its output receive the additional SF
gradient. The codec, including shared precision embedding, receives the native
RD gradient. Later SANA layers, AE and D receive the native image gradient.
All groups use the same explicit, averaged distributed gradient reduction;
this avoids mixing parameter-specific `autograd.grad` with DDP reducer hooks.
The SF denominator is the global count of active elements, including ranks
with zero active samples. Global gradient clipping follows the one reduction.

## Fixed experiment family

All recipes are in `configs/bt_self_flow/spatial_sf/`:

| Recipe | D | P | SF/teacher |
|---|---|---|---|
| `c.yaml` | off | off | off |
| `c_d.yaml` | on | off | off |
| `c_sf.yaml` | off | on | on |
| `c_d_sf.yaml` | on | on | on |
| `c_p.yaml` | off | on | off |
| `c_d_p.yaml` | on | on | off |

D is the existing mu-QK PiT with shared precision additive AdaLN, version 4.
Its output projection starts at zero. All groups start from the same pure-C
`backend_init`, not an already trained C+D model. P starts at identity. The
launcher verifies that the six configurations differ only in these switches
and output names; common data, losses, optimization and training settings match.

Every recipe fixes **two ranks, batch 4 per rank, accumulation 1, global batch 8**.
The budget is 200K successful updates. Compare at least seeds 903 and 1903,
with fixed final EMA snapshots and late-training curves. Report the teacher's
additional compute rather than describing equal steps as equal compute.

P is bypassed for a fixed 10% of training samples in all P-containing groups,
including capacity controls. Bypass and teacher-rate draws use dedicated
counter-based random streams. They do not consume the student's stochastic
quantization stream. The highest student quality uses a same-quality EMA target;
this is depth self-distillation, not extra bitrate information.

## Initialization and launch

Set `DITIC_DATA_ROOT`, `DITIC_SANA_PATH`, and `DITIC_TEXT_PATH` if the configured
local defaults differ from the target machine. Use an existing, completed,
calibrated precision-v3 **pure C** warmup. Prepare its common backend state once:

```bash
python train_bt_self_flow.py \
  --config configs/bt_self_flow/spatial_sf/c.yaml \
  --prepare-backend /path/to/common/complete.pt \
  --output /path/to/shared_backend_init.pt --device cpu
```

The handoff checks asset identity, codec configuration, calibration and quality
span. Training rejects a shared initialization containing D/P weights or a
different decoder history. All ablations load the exact same source bytes.

Print the resolved two-rank command without starting a run:

```bash
python scripts/launch_spatial_sf.py --variant c_d_sf --gpus 0,1 \
  --init-checkpoint /path/to/shared_backend_init.pt \
  --run-root /path/to/spatial_runs --dry-run
```

Before long training, run two actual updates on the intended two GPUs:

```bash
python scripts/launch_spatial_sf.py --variant c_d_sf --gpus 0,1 \
  --init-checkpoint /path/to/shared_backend_init.pt \
  --run-root /path/to/spatial_runs --preflight
```

This writes to a separate preflight directory and stops at update 2. It exercises
the real models, losses, EMA, optimizers, communication and checkpoint save; it
does not change the 200K training recipe. An unavailable driver or either
unavailable selected CUDA device aborts before creating a started run. Two tiny
CPU/Gloo ranks validate communication logic, but cannot certify CUDA memory,
kernel numerics or speed. Both kinds of evidence must be labeled separately.

Launch the actual run by omitting `--preflight`. Use the same command with
`--variant c`, `c_d`, `c_sf`, `c_p`, or `c_d_p` for matched controls. Change
`--seed` to run a second paired seed. To continue a run, replace
`--init-checkpoint` with its own `--resume /path/to/paused.pt`; do not hand off
another ablation's trained state. Raw weights, EMA, optimizer, auxiliary
optimizer, scaler, scheduler, sample counters and rank RNG are restored.
The truncated teacher is reconstructed from that exact deployment EMA.

## Inference switches and evaluation

`model.decoder_switches(sf=False, d=True)` bypasses P while retaining D;
`model.decoder_switches(sf=True, d=False)` runs P and bypasses D. Use these
contexts during evaluation on an identical received `(mu, delta)`.
Normalization and the teacher are training-only; receivers require neither
teacher inputs nor additional bits. Ordinary checkpoint loading is strict:
missing P/D or a different D version is not silently initialized.

Switching a trained full model is a dependence diagnostic. It does not replace
four independently trained core ablations, nor does a D-before image substitute
for the independently trained no-D model. Report `SF | no D` and `SF | with D`
separately, and compare each to its corresponding P-only capacity control.

Use the normal `eval_bt_self_flow.py` entry point with deployment EMA and actual
compress/decompress. Count headers and side bits, compare the shared true-bpp
range, and report PSNR, MS-SSIM, LPIPS and DISTS on Kodak and an independent set.
End-to-end groups may learn different codecs; equal q need not imply equal bpp.
For a fixed-input backend diagnostic, decode with the matching frozen codec and
cache `(mu, delta)`; never feed one learned entropy model another model's stream.

This branch also integrates the existing uncommitted evaluation-protocol fixes
from the mu-PiT checkout: consistent image domains, structural metrics,
validation identity and fixed perceptual-network evaluation mode. Their original
checkout is preserved. The protocol is shared by all six recipes; see
`docs/EVALUATION_PROTOCOL.md`.

## Validation

```bash
python -m pytest tests/test_spatial_self_flow.py tests/test_spatial_training.py -q
```

These tests use real small diffusers SANA, PEFT modules, precision-v3 codec,
entropy coding and AE decoder components. They check initial parity, spatial
statistics, teacher stop-gradient and EMA state, main-path effects, gradient
partition, packets, recomputation, exact continuation, and two Gloo processes
with batch4/accum1 including globally bypassed and locally empty SF branches.
Production CUDA preflight results are separate from these CPU tests.

### Verification record, 2026-11-02

Python: `/data1/ls/envs/cvpr/bin/python`.

| Check | Result |
|---|---|
| Spatial SF, backend, losses, existing D and precision-C regression | 44 passed, 3 CUDA tests skipped |
| Real trainer, two-process Gloo, evaluation protocol and rate gates | 63 passed, 14 subtests passed |
| Directory/distribution evaluation | 16 passed |
| Independent subagent review and repeated SF/trainer tests | No blocking findings; 15 passed, one Gloo test excluded from that repeated run |
| Real initialization and matched-config launcher dry-run | Passed |
| Full-size two-GPU CUDA preflight | Pending; existing jobs occupy GPU 0/1 at 32092/31816 MiB of 32607 MiB each |

The initialization used in the dry-run is
`/data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt`, SHA256
`7286c89f56637c0c94e5d3cb8fb34801f849a44816a8977a5c579973b11efaa8`.
This validates the launch configuration and source checkpoint, without allocating
the production model. No long training or new RD evaluation was performed.
