# Batch-8 C versus C + quantization-residual Self-Flow

This comparison continues the user's **pure-C 10K checkpoint with effective
batch 8**. Preparation does not stop the existing batch-16 run, start a new
trainer, or reinterpret a joint checkpoint as a new backend initialization.
The source checkpoint has not yet arrived; its embedded configuration remains
the authority if any prepared baseline setting differs.

| Setting | Pure C | C + SF |
|---|---|---|
| Model | Original `BTCodec`, no residual head | `CQuantizationResidualCodec` |
| Source | Same pure-C 10K raw weights, EMA and optimizer state | Same |
| GPUs / local batch / accumulation | 2 / 4 / 1 = effective batch 8 | Same |
| Input | 512 × 512 | Same |
| Quality | Independent uniform draw from six qualities per optimizer step | Same draw sequence |
| Precision on V100 | Explicit FP16 + GradScaler, shared FP32 fixes | Same |
| Image loss | 4 MSE + 1 LPIPS + 1 DISTS | Same |
| Latent objective | 0.1 base NMSE, repaired NMSE weight 0 | Same |
| Rate coefficients for q = 0, .2, .4, .6, .8, 1 | 16, 8, 4, 2, 1, .5 | Same |
| Main learning rates | Codec / DiT / VAE: 1e-4 | Same; SF head also 1e-4 |
| Auxiliary learning rate | 1e-3 | Same |
| Main optimizer | AdamW, betas (.9, .999), weight decay 0, gradient clip 4 | Same |
| LR milestones | 100K ×.5, 160K ×.2, 190K ×.1, cumulative | Same |
| Target step | 200K including the existing 10K prefix | Same |
| Save / log / validation | Periodic 10K / 100 / 2K | Same |

New best validation scores can also save snapshots between periodic 10K saves.
Validation uses Kodak24, deployment EMA (.999), and total entropy-coded bpp.
Full packet decode occurs every 10K; the other validation intervals check two
packet samples, using the existing evaluator in both branches.

SF keeps the reviewed mechanism: student layer 8 predicts the teacher layer-20
quantization residual; a correction is injected into the student's layer 20.
The student still optimizes the reconstruction/rate objective. The teacher
compares native-precision and locally degraded versions of the same transmitted
latent, has no gradients and tracks student weights with EMA .999. The student
sees the degraded version and predicts this feature difference. Student-local
erasure masks provide the within-student asymmetry.
Both ranks keep a teacher replica, following the existing working DDP design.

At the saved 10K boundary, initialize the residual head and teacher, recalibrate
the SF coefficient on this source, and ramp SF over updates 10001–12000.
Participation ramps to half the samples; selected samples erase a quarter of
their latent cells. The SF gradient target is .05 with four calibration samples
per quality. Do not reuse the alpha from the previous batch-16 experiment.
`mixed_sampling: false` disables the legacy cross-quality binary mixture;
local SF erasure remains enabled.

The independent quality sampler explicitly overrides residual-SF's older
six-step balanced default. Both new branches use the same step-seeded draws,
which are shared by both ranks. Omitting this override in old configs retains
their original behavior. The explicit FP16 setting also prevents accidental
precision changes if the configs later move to an Ampere GPU.

## Preparation and source migration

```bash
python scripts/prepare_c_sf_ablation.py \
  --output /data2/ls/experiments/c_sf_ablation_b8_20261026

# Once the checkpoint is uploaded, inspect its metadata without starting CUDA:
python scripts/prepare_c_sf_ablation.py \
  --output /data2/ls/experiments/c_sf_ablation_b8_20261026 \
  --checkpoint /absolute/path/to/uploaded_10k.pt
```

The script writes both fully resolved configs and `plan.json`. Its
`launch_ready` stays false: metadata inspection alone is not a state migration.
No trainer process or training output directory is created by this command.

Ordinary `--resume` rejects a single-rank BF16 checkpoint against the new
two-rank FP16 training contract. Once the file arrives, verify source tensors,
assets, dataset identity and configuration, then explicitly migrate:

1. Preserve raw weights, deployment EMA, named Adam/auxiliary moments and LR
   scheduler at optimizer step 10000. Only the SF branch adds new head state.
2. Preserve the global data/crop counter. For 1 × 8 × accumulation 1 to
   2 × 4 × accumulation 1, the existing counter sampler partitions the same
   global sequence of eight samples; `micro_seen` need not change.
3. Document a shared two-rank RNG policy for both branches and create a fresh
   FP16 GradScaler. BF16 source scalers are disabled/empty; do not load an empty
   scaler into an enabled FP16 one. Cross-device training cannot be bitwise
   identical to the old single-GPU BF16 trajectory.
4. Verify C outputs and loss agree before SF, calibrate on the uploaded source,
   and smoke-test the actual resumed update before a long run.

This controls the continuation comparison from 10K onward. The historical
single-GPU BF16 curve remains useful context; the new pure-C FP16 continuation
is the control for attributing gains to SF.
