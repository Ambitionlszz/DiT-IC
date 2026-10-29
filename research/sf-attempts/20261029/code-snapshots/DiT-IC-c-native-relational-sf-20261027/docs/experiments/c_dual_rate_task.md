# C + dual-rate task-supervised Self-Flow (2026-10-27)

This experiment restarts from the user's **pure C optimizer step 10,000**,
not the failed quantization-residual experiment. The previous run was saved
and stopped at step 21,663. Its checkpoints and evaluation history are retained.

The main branch reconstructs **every image** with the complete native C decoder.
Its loss is `beta(q_a)*R_a + 0.1*base + 4*MSE + LPIPS + DISTS`.
Repaired-latent loss is zero. The six qualities are sampled independently and
uniformly per update, as in source C. Both ranks use the same native quality.
The rate weights are `[16,8,4,2,1,0.5]` at `[0,.2,.4,.6,.8,1]`.

For a random half of each local batch, independently draw a second quality
per image. Perform a **separate precision-conditioned codec analysis** at that
quality. Mix decoded latents with exactly 25% spatial B cells, all channels
at a cell together. Do not sort the qualities or exclude equal-quality pairs.
The corresponding spatial quality map is passed to the student. A frozen
EMA teacher sees the uniformly higher-quality decoded view and supplies its
complete layer-20 features. A small student predictor maps layer 8 to these
features. Cosine alignment and an auxiliary image reconstruction both train
that predictor and the shared student prefix.

The auxiliary image uses the prediction **as the layer-20 feature** and runs
the current shared C layers 21--28, output projection, C scheduler and VAE.
The scheduler's input base is **the student's mixed latent**. The auxiliary
readout parameters are detached, while its input remains differentiable.
Functional parameter substitution takes place **inside** checkpointed calls,
including during backward recomputation. The codec and readout parameters
receive no auxiliary gradients. Main RD still updates the complete C.

The total loss is `L_C + ramp * (cosine + D411(auxiliary_image, image))`.
There is no factor 2 for the subset mean, residual NLL, erasure of router
codes, target-difference normalization, or gradient-calibrated alpha.
The outer SF weight reaches **1** over 2,000 updates starting after C10K.

C gains an explicit zero-initialized six-level token-quality conditioner.
The native full decoder uses uniform quality; the training mixture uses a
spatial quality map. Teacher conditioning follows EMA (decay .999). This
conditioner is an architectural change and must be acknowledged in ablations.
Only the six native quality points are supported by this experimental decoder.
The predictor is auxiliary only; deployment uses all 28 C layers.

Before native updates, perform 64 **head-only** preparation updates using
`cosine + auxiliary D411 + relative feature MSE`. C, conditioner and teacher
stay fixed. Preparation does not advance global C steps, sample counters,
source optimizer or RNG progression. Its optimizer is separate and discarded;
the subsequent main SF objective does **not** contain feature MSE.
The prepared state is saved as `sf_start.pt` before updating C.

## Resources and continuation

- Two V100S 32GB GPUs, local batch **4**, accumulation **1**, effective batch **8**.
- Each GPU holds a complete student and its frozen EMA teacher prefix.
- Main backward runs before the auxiliary forward/backward, releasing its
  saved activations. The optimizer steps once after both losses.
- This method uses explicit mean gradient reduction once after accumulation,
  without a concurrent DDP reducer. Unused parameters participate as zeros.
- FP16 AMP + GradScaler with existing FP32 attention/codec numerical fixes;
  V100 does not provide native BF16. Source scaler is initialized to 64.
- If the real active-teacher configuration OOMs, change local batch to 2 and
  accumulation to 2, preserving effective batch 8 via explicit migration.
- Source C raw/EMA weights and named Adam/auxiliary moments are preserved.
  LR milestones remain 100K/160K/190K; total target is 200K. Existing source
  rank RNG states are retained. The global 80,000-sample counter is remapped
  from 20,000 to 10,000 microbatches. Image path-layout changes were confirmed
  by the user; the local 89,026-image list fingerprint is recorded.
- Kodak24 EMA evaluations use actual packet bpp, every 2K. Log every 100;
  periodic complete saves every 10K plus selected best and graceful pause.
  `--max-updates` pauses at an absolute global step without changing the run
  contract. The shared supervisor supports graceful worker checkpointing and
  verified, bounded FP32 restart after a nonfinite forward.

## Commands

Use the provided environment's Python, local pretrained assets and offline
HF/torch caches. First migrate once into a fresh output directory:

```bash
python scripts/migrate_dual_rate_task.py --checkpoint /path/to/pure-C/0010000.pt \
  --config configs/bt_self_flow/c_dual_rate_task.json --allow-path-relocation
python scripts/check_dual_rate_task_gpu.py --config configs/bt_self_flow/c_dual_rate_task.json \
  --checkpoint /path/to/pure-C/0010000.pt --output /path/to/preflight.json
python scripts/launch_dual_rate_task.py --root /path/to/run \
  --config configs/bt_self_flow/c_dual_rate_task.json \
  --init-checkpoint /path/to/pure-C/0010000.pt --gpus 0,1
```

Commit reviewed code before launching; the supervisor freezes its commit,
config and initialization hash. Source SHA256:
`21d23b3beef59a9b30e08a11776b14720041f994f742c6de961c0ee2375c8ab6`.

## What constitutes evidence

CPU tests cover split/full readout equality, checkpoint recomputation,
image-only gradients reaching prefix/predictor while codec/tail/VAE stay
untouched, separate-quality analyses, and actual two-rank gradient averaging.
The real 512px preflight measures zero-adapter equivalence, main+auxiliary
backward memory, feature amplitude, and teacher correct/wrong-image/spatial
shuffle readouts at multiple qualities. FP16 decoder recomputation can differ
by rounding; both maximum image error and mean square error are reported.
Teacher features need to affect the readout, but an oracle substitution can
be incompatible with a different latent base; its immediate improvement is
reported, not promised. A tiny cosine loss or good auxiliary reconstruction
cannot establish compression gains.

Judge the native **full C** Kodak24 curve against external pure C at the same
steps and total bpp. This trial is exploratory: added conditioning and extra
image supervision are confounds. A future equal-supervision/no-cosine control
is necessary to isolate alignment's contribution. No pure-C training run is
launched on these GPUs.
