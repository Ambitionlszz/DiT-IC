# C quantization residual Self-Flow

Base: `073f54eced8bff354f49f3233da9dca9aee8eccf`. Initialize joint training from
the downloaded precision-v2 lambda16 50K `backend_init.pt`, not its warmup
optimizer or the earlier D+SF experiment.

The original SF auxiliary represented the entire fine-rate teacher feature;
its projector was not consumed by reconstruction. This experiment predicts the
normalized *difference* between fine and received features of one EMA20,
and adds the same mean prediction from student8 after student20. Actual image
RD therefore trains this correction directly. This is a research hypothesis,
not an established compression gain.

## Views and gradient contract

- One global q, one precision-conditioned analysis/hyperprior/three-bit router.
- Half the images retain native allocation. On the other half, erase 25% of
  integer allocation cells to code zero and recompute centered steps, all four
  context groups and synthesis. The perturbed student is an ordinary packet.
- Centering also refines some other cells: equal geometric mean step does not
  imply equal bitrate or that every selected cell gets coarser.
- Teacher-only steps are pointwise `min(native_step, student_step)`, with the
  same analysis. This privileged view is not claimed to be a native packet or
  a nested code. Both teacher forwards reuse one frozen EMA prefix20.
- Target: RMS-normalized fine20 minus received20. Student8 and actual packet
  log-step predict mean and variance. Inject `stopgrad(RMS(h20))*mean` at20.
- Gaussian NLL uses FP32, a .05 std floor, and initially .5 std. Variance is
  training supervision; it does not gate the reconstruction mean.
- SF repeats only student1–8 with detached coarse **and log-step**. It cannot
  update the codec/router. Actual reconstruction retains full codec gradients.
- Rate estimates include native side bits; validation splits real packet bits
  into main, hyperprior, allocation, scale, mask and header.

## Training and continuation

512 input, 4 MSE + 1 LPIPS + 1 DISTS, .1 base latent loss, **zero repaired
latent loss**. Six shuffled balanced q values use beta `[16,8,4,2,1,.5]`.
Codec/DiT/AE-decoder LoRA LR 1e-4, entropy auxiliary 1e-3, gradient clipping4,
200K updates, original milestone decay.

First10K are pure RD. The zero-mean residual head exists dormant before DDP
construction; it receives no gradients or Adam states until activation. This
preserves pure-RD output and avoids adding parameters to an existing reducer.
Save10K and `pre_sf.pt` before activation. Teacher is created from current raw
student, EMA .999. A2K ramp enables both perturbation participation and SF
weight. All ranks use `find_unused_parameters=True` for dormant/variance paths.

At activation, independent training crops provide at least4 valid samples per
q. Set alpha once to `min(10,.05*median_q(||dD411/dmean||/||dNLL/dmean||))`.
Preserve the main data/RNG stream. Also inject the privileged target through
the actual decoder: if it worsens average D411 at both q endpoints and wins
on at most25% of those samples, retain that finding in `sf_transition.json`
and log a warning, then continue the scheduled ramp. This is a small-sample
test before the decoder adapts to residual injection, not a requirement for
starting training or evidence of final RD improvement. Nonfinite/degenerate
calibration remains an error. Active-SF resume requires a recorded calibration
protocol and a finite positive SF weight, including when this diagnostic fails.

Checkpoints contain raw/deployment EMA including head/activation, teacher
adapters/text, fixed alpha and calibration, Adam/aux/scheduler/scaler, crop
counter and per-rank RNG. `sf_start.pt` resumes after calibration. The CPU test
compares uninterrupted versus active-SF continuation tensor by tensor.

## Two GPUs

Each rank owns a full student, local EMA20, shared codec and AE; two teacher
views run sequentially. First probe **batch8, accumulation1** on both32GB
V100S. On OOM use **batch4, accumulation2**. Both yield global batch16.

`scripts/check_quantization_residual_gpu.py` runs real512 image losses, teacher,
full backward, entropy auxiliary, Adam states, deployment EMA and active-head
packet decoding for at least2 successful updates. Its updates are discarded.
Research training always starts fresh from the downloaded `backend_init.pt`.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/check_quantization_residual_gpu.py \
  --config configs/bt_self_flow/quantization_residual/c_sf.json \
  --init-checkpoint /path/to/backend_init.pt --output /path/to/probe.json \
  --check-calibration

python scripts/launch_quantization_residual.py \
  --root /path/to/new/run --gpus 0,1 \
  --config configs/bt_self_flow/quantization_residual/c_sf.json \
  --init-checkpoint /path/to/backend_init.pt
```

Use the b4a2 configuration only if the full probe requires it. The supervisor
freezes source/config/init identity and refuses uncommitted code. SIGTERM is
forwarded to trainer workers so they finish an update and save `paused.pt`.

`c_sf_off.json` defines the same structure, views, optimizer and budget with
zero SF objective. It is a control recipe; launching the primary experiment
does not run this control concurrently. At matched *total actual bpp*, compare
both low and high rates. With a fixed packet, decoder interventions
`residual_intervention='zero'` and `'shuffle'` test whether the mean is used.
10K→30K improvement or a falling NLL alone does not demonstrate an SF gain.

## Verification

`tests/test_quantization_residual_sf.py` covers native-SANA parity, meaningful
image influence of the head, rate/packet parity at all six q values, teacher
RNG isolation, SF versus RD gradient routing, deployment/teacher state, two
DDP ranks (including a zero-valid rank), real calibration and exact resume.
Existing precision-v2/trainer/codec/backend/loss/evaluation tests are retained.

Full V100S preflight: b8a1 completed native RD but OOMed when SF became active.
b4a2 completed native RD, real two-rank calibration, two active-SF Adam
updates and exact packet reconstruction (max error0), peak allocated22.14GiB
per GPU. The capacity probe after just one RD update failed the privileged
task gate at q0/q1; it is discarded, including its alpha10. Research runs the
agreed10K RD prefix and repeats this diagnostic before enabling SF. The first
10K transition also failed this diagnostic and stopped under the original
policy; the continuation preserves that pure-C checkpoint and treats the
diagnostic as a warning, as requested by the user.

After 973 active-SF updates, one rank's FP16 forward overflowed in SANA block27's
MLP point projection. The checkpoint at step10973 has finite model state. The
same saved crop and mask reproduce the NaN under AMP and remain finite in
FP32. SANA feed-forward modules now reuse the original weights/keys with FP32
arithmetic, preserving large finite outputs through the gated residual. This
changes numerical precision only; the data, loss, ramp, optimizer and teacher
state resume from the saved update.

## Stable continuation and bounded recovery

`c_sf_b4a2_stable.json` keeps the same optimization contract and saves every500
updates (keeping the last2 numeric snapshots and the protected pure-C state).
The existing gradient clip and GradScaler retry the same batches on gradient
overflow; skipped updates now log the loss scale explicitly.

For a nonfinite AMP forward with finite model state, both ranks save the
optimizer/EMA/teacher/RNG/crop counter **before** any update, with
`full_precision_retry=true`. Rank0 publishes a restart request only after the
checkpoint save completes. The supervisor verifies the request against the
checkpoint size, timestamp, step, attempts and both RNG states, then rebuilds
the two workers. Their first update replays the same crops/masks in FP32 with
activation checkpointing; subsequent updates return to AMP. A single-process
trainer can replay directly without rebuilding DDP.

The restart history persists across supervisor invocations with a maximum of3
numerical restarts. An old request, OOM, nonfinite model state, repeated FP32
failure or unrelated process error cannot trigger this recovery. Such errors
remain visible and stop instead of silently skipping data or looping.

The V100S recovery probe starts from the production step11332 checkpoint,
forces a rank1 NaN after a partial accumulated backward, and exercises the
real supervisor restart. It completes the same update in FP32, then the next
update in AMP with full SF weight and participation. Both GPUs fit within
25GiB reserved; all probe updates are discarded. Results and restart history
are recorded in `docs/test_reports/c_quant_residual_ddp_recovery_v100.json`.
Before continuation, every floating tensor in the student, EMA, teacher and
both optimizer states at step11332 passed a finite-value audit. Production
resumes that checkpoint with its original SF ramp and optimization contract.
