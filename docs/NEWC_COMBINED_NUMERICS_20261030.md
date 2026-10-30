# New-C PiT / evidence Self-Flow numerical repair (2026-10-30)

Ports the numerical repair from `de13052138e8061db0bd8c783fc67393344ab43a`
into both experimental branches without replacing their decoder or objectives.
The upstream reproduction and mechanism are documented in
[C_V3_NUMERICS_20261030.md](C_V3_NUMERICS_20261030.md).

Both experiments used the same affected SANA cross-attention, official LPIPS
feature normalization, and FP32 global gradient-norm reduction. Cross-attention
now selects the FP32 math SDPA backend locally; StableLPIPS changes only its
feature norm implementation; clipping retries the norm reduction in FP64 when
all individual gradients are finite. Existing complete-decoder/perceptual
activation checkpointing and GPU-resident saved tensors are preserved.

Recovery saves the last successful update, including raw/EMA, both optimizers,
scheduler, RNG by rank and data cursor. Single-GPU BF16 can replay the identical
update once in FP32. Distributed failures stop both ranks at their complete
recovery point. No batches are dropped to escape a failed gradient, and a
failed optimizer update never advances the reported successful-step counter.
The evidence method continues to average its explicit parameter-role gradients;
it is not wrapped in DDP's autograd reducer.

The working trees already contained local linear-attention denominator guards
and optional diagnostic hooks. Their original patches and source were archived
before editing. Those guards remain as in the running experiments; they are
not part of upstream de13052 and this repair does not establish them as the
cause or cure of the CUDA cross-attention defect. Optional failure diagnostics
now inspect gradients before clipping destroys evidence, using FP64 diagnostic
norms. Trace hooks are enabled only on explicit diagnostic requests.

## Protected continuation

Full graceful recovery points were preserved before migration:

- PiT GPU0: step/attempts/data cursor/main optimizer/aux optimizer/scheduler all
  **36,480**, batch 8, accumulation 1, one RNG state.
- Evidence Self-Flow GPU1+2: all corresponding counters **9,165**, per-rank batch
  4, accumulation 1, effective batch 8, two RNG states.

The older Self-Flow diagnostic recovery file labeled 9,678 has optimizer step
9,660 because the preceding temporary skip policy advanced the label without
an optimizer update. It is retained solely for diagnosis; it does not initialize
the repaired main continuation. The clean 9,165 point preserves all completed
updates of the active run.

Local audit artifacts, source patches, protected checkpoints, test output and
continuation verification are retained under:
`/data1/ls/code/runs/numerics_de13052_20261030/`.
No isolated diagnostic update initializes either main experiment.

## Validation

- PiT trainer/backend/adapter regression after the final diagnostic update:
  **41 passed**.
- PiT numerical tests: **47 passed**, including real CUDA large-query attention
  backward checks, LPIPS value/gradient/state compatibility, norm-only overflow,
  bounded FP32 replay and exact process-restart continuation.
- Recovery after additional failure-diagnostic fields: **10 passed**.
- Shared/evidence suite: **120 passed, 10 skipped** with CPU/Gloo, covering
  native C, PiT, both losses, normal/evidence parameter-specific gradients,
  main/aux updates, EMA and two-rank exact continuation.
- The added evidence numerical-recovery test corrupts only one rank's gradient,
  verifies both ranks stop before main/aux optimizer or scheduler updates, then
  checks exact raw/EMA weights and data/scheduler position against an uninterrupted
  reference. It exercises the actual evidence backward and explicit averaging.

Actual full-backbone continuation counters, devices, checkpoint consistency and
post-repair metrics are recorded in `continuation_verification.json` in that
audit directory. A successful continuation is evidence of numerical stability
for the observed updates; it does not establish final rate-distortion quality.
