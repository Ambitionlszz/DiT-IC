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
on at most25% of those samples, retain the pure-RD checkpoint and stop for
review. This diagnostic cannot establish final RD improvement.

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
agreed10K RD prefix and repeats this diagnostic before enabling SF.
