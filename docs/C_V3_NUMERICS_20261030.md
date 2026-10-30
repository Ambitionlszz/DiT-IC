# C v3 numerical failure and repair

Reproduction environment: RTX 5090, PyTorch `2.12.0.dev20260331+cu128`,
CUDA 12.8, diffusers 0.35.2 and LPIPS 0.1.4. The backend diagnosis below is
for this installed build; it does not establish behavior in other versions.

## Reproduced failure

On 2026-10-30 the GPU1 joint run was reproduced from its saved step 48,956,
including the optimizer, scheduler, EMA, RNG and data cursor. Both diagnostic
branches used the original LPIPS implementation, the same eight 512px crops,
quality 0.6, and accumulation 1. Diagnostic updates were written to separate
directories and never replaced the main run's checkpoint.

The default CUDA cross-attention backend produced finite outputs and finite
individual parameter gradients, but the total gradient norm was
`7.98261021382037e19`. The ordinary FP32 squared-norm reduction overflowed;
this was not evidence that the model weights or individual gradients were NaN.

Backward hooks located the amplification in cross-attention of the last SANA
transformer block, index 27. The output gradient norm was about `1.11e-7`, while
the gradient reaching the preceding self-attention output was about `1.44e13`.
The shared precision embedding received this propagated gradient. Its presence
in the failure diagnostics did not identify it as the origin of the failure.

## Attention backend isolation

The actual projected query/key/value tensors and incoming attention gradient
were captured from that backward. Replaying just these tensors selected
`aten::_scaled_dot_product_efficient_attention_backward` by default. Its output
matched the captured output exactly, confirming that this was the executed path.
All captured tensors were finite; the largest query magnitude was `27,966,120`.

| Quantity | Default CUDA efficient | FP32 math | FP64 math reference |
| --- | ---: | ---: | ---: |
| Query gradient norm | 4.13819e11 | 7.68081e-16 | 1.57833e-24 |
| Key gradient norm | 9.79173e17 | 1.25302e-10 | 3.69472e-19 |
| Value gradient norm | 1.03791e18 | 8.45505e-8 | 8.45505e-8 |
| Maximum output difference from FP64 | 1.19209e-7 | 1.49012e-7 | 0 |

The FP32 math value gradient differed from FP64 by at most `4.20194e-15`.
The query/key gradients have small absolute FP32 roundoff rather than accurate
relative error at values extremely close to zero. These results isolate the
catastrophic amplification to the efficient backward on this input.

`StableSanaCrossAttention` now selects `SDPBackend.MATH` inside its existing
FP32 scope. This changes neither the attention equation nor parameter/state
keys. It does not clip queries, detach conditioning or add a learned parameter.
The backend context is local to the cross-attention call.

The full original batch was then replayed with all cross-attention calls using
that processor:

| Full-batch quantity | Original backend | Repaired backend |
| --- | ---: | ---: |
| Total parameter gradient norm, before clipping | 7.98261e19 | 2.37866 |
| Pixel MSE | 0.020127267 | 0.020125313 |
| LPIPS | 0.121514611 | 0.121511310 |
| DISTS | 0.094414331 | 0.094414324 |
| Estimated bpp | 0.069387428 | 0.069387428 |

Forward values are close, not bitwise identical across backends. The default
probe completed only because the new overflow-safe norm reduction was present;
its enormous gradient is still incorrect and was not used to resume the main
experiment. The repaired probe completed with a normal gradient norm.

## Independent LPIPS defect

LPIPS feature normalization uses `feature / (norm + 1e-10)`. Its original
`sqrt(sum(feature**2))` implementation has a nonfinite backward at tiny positive
features whose squares underflow, including after ReLU. This is a separate
failure mechanism from the all-finite attention gradients above.

The local `StableLPIPS` subclass preserves the official initialization,
pretrained state keys, distance formula and forward options. It replaces only
the feature norm with `torch.linalg.vector_norm`, which has a defined zero-norm
derivative. Normal inputs retain matching distances and image gradients within
FP32 tolerance. No installed package or global LPIPS function is modified.

## Clipping and recovery

- The normal clipping path and configured norm threshold remain unchanged.
  If the FP32 norm alone overflows, an FP64 reduction computes the norm and
  applies the same clipping rule. Genuine NaN/Inf gradients are still rejected.
- Native BF16 can now request one full-precision replay even without GradScaler.
  Failure is diagnosed before the main optimizer or EMA update; the saved
  recovery point contains the original RNG and data cursor for that update.
- A legacy checkpoint's FP32-retry flag is cleared only when its matching
  diagnostic establishes a single-rank, finite-model, finite-gradient norm-only
  failure. Missing or incompatible diagnostics retain the conservative retry.
- FP32 replay is bounded and can require more memory. It is a recovery fallback,
  not the attention fix or a guarantee that every failure fits in GPU memory.

## Verification and continuation

CPU checks completed: **83 passed, 9 skipped** across attention, perceptual,
precision recovery, BT training/backend/CUDA/loss contracts, C v3 validation and
the GAN discriminator contract. The two new CUDA unit cases were among the
skips; the full pretrained GPU batch reproduction above was executed separately.
Recovery tests compare exact raw/EMA weights, scheduler and data position after
in-process replay and process restart. Persistent bad gradients are rejected
before an optimizer update.

An independent read-only review found no blocking issues and separately checked
normal, exceptional and nested backend-context restoration, including backward
after context exit. The new nonfinite-gradient recovery cases have not been
directly exercised with two DDP ranks.

After both diagnostic branches, the protected GPU1 worker loaded the newest
main checkpoint at **50,198** and completed more than 100 further updates with
the repaired attention and LPIPS. The main run retains batch 8, accumulation 1,
the 200K target, AdamW settings, learning-rate schedule and effective-batch-16
50K warmup lineage. No stage initialization or diagnostic update was substituted
for its complete resume checkpoint.

Local evidence is retained under
`checkpoints/c_v3_joint_gpu1_b8a1_after_eff16_20261028/numerics_recovery_20261030_probe_r4/`:
`manifest.json`, `fault_full_trace.json`, `fault_full_trace.cross.backends.json`,
`fault_math_trace.json`, both probe training logs, and `status.json`.
The manifest records the checked source hashes and the protected GPU handover.

These checks establish that the reproduced numerical failure is repaired.
They do not establish an improvement in converged rate-distortion quality or
exclude every possible future numerical failure.
