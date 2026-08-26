# Variable-rate LIC and Multi-rate Plain DiT training

All commands run from the repository root. Replace model/data paths in the
YAML files before launching. The same launcher supports one or two GPUs.
Every selected GPU uses batch size 8. Accumulation defaults to 1:

```bash
# One GPU: per-GPU batch 8, global batch 8.
bash scripts/run_variable_rate_train.sh lic-smoke 0

# Two GPUs: per-GPU batch 8, global batch 16.
bash scripts/run_variable_rate_train.sh lic-smoke 0,1
```

For an explicit override, prefix the command with values such as
`PER_GPU_BATCH_SIZE=4 ACCUMULATION_STEPS=2`. The available stages are
`lic-smoke`, `lic-pilot`, `lic-main`, `plain-smoke`, and `plain`.

The experiment DAG is:

```text
scratch two-phase LIC -> accepted V2 LIC -> multi-rate Plain P0
                                           -> Self-Flow feasibility experiments
```

Self-Flow is intentionally deferred until P0 passes its six-rate and frozen
bitstream gates.

## 0. Scratch initialization contract

Stage 1 does not load a DiT-IC baseline checkpoint. It initializes the complete
variable-rate LIC randomly, loads only the pretrained SANA VAE, and freezes the
VAE encoder and decoder. It uses the complete DCVC-FM control path cited by
FlowCodec: learned per-quality `q_e/q_d` feature gains plus hyperprior-predicted
content-adaptive `w_e/w_d` quantization maps. All six quality gain tables start
at one and separate during training under their different lambda values.

Each LIC config must keep `model.codec_initialization: scratch` and must not
define `model.source_checkpoint` or VAE LoRA.

## 1. Automatic two-phase LIC training

One `lic-pilot` or `lic-main` launch runs both phases without a manual restart:

1. Phase A optimizes `latent_mse + lambda_lat(q) * R` and does not run the
   frozen VAE decoder.
2. At the configured boundary the trainer resets the optimizer, scheduler,
   and EMA, writes a Phase-B-ready `phase_a_complete.pt`, and automatically
   enters Phase B.
3. Phase B drops latent MSE and optimizes
   `(pixel_l1 + lpips + dists) + lambda_img(q) * R` through the frozen VAE
   decoder. Only the LIC is updated in both phases; there is no GAN loss.

The LPIPS and DISTS training losses both receive images in `[-1, 1]`, matching
the baseline loss path. The loss weights and the two phases' lambda values are
explicit under `lic_train.phase_a` and `lic_train.phase_b`.

The pilot is 2.5k Phase A steps followed by 2.5k Phase B steps. The main run is
50k + 50k. A checkpoint records the current phase and both boundaries, so
resuming continues in the correct phase. Resume also requires the same scratch
initialization contract, pretrained VAE source, formula version, lambda tables,
and loss weights.

Before the main run, run the throwaway 20+20-step path once to catch loading,
validation, phase-switch, and checkpoint errors:

```bash
bash scripts/run_variable_rate_train.sh lic-smoke 0,1
```

## 2. Six-rate lambda table

Both phases use the initial rate-weight table
`[16.0, 8.0, 4.0, 2.0, 1.0, 0.5]`, mapped directly from q0 through q5.
The two phases keep separate copies of the table because their distortion
scales differ. This follows the baseline convention `D + lambda * R`: smaller
lambda means weaker rate pressure, higher BPP, and higher quality. Therefore q0
is the lowest-quality point and q5 is the highest-quality point.

The continued Phase-B run measured monotonic likelihood BPP from about
`0.0139` at q0 to `0.0504` at q5, while all distortion and perceptual metrics
improved monotonically. This is the accepted operating range. Serialized rANS
BPP remains the final measurement because likelihood BPP excludes container
overhead.

## 3. V2 main

```bash
bash scripts/run_variable_rate_train.sh lic-main 0 --calibration_confirmed
# Or: bash scripts/run_variable_rate_train.sh lic-main 0,1 --calibration_confirmed
```

Run the direct LIC six-rate evaluation again. Gate A requires every q to
round-trip and mean real BPP to satisfy `q0 < q1 < ... < q5`. Likelihood BPP
is diagnostic only.

## 4. P0 Multi-rate Plain DiT

Set `model.params.codec_path` in `configs/multirate_plain_dit.yaml` to the
accepted V2 checkpoint. The config loads and freezes only the V2 LIC, loads the
pretrained SANA VAE and DiT backbones, keeps the VAE frozen without LoRA, and
trains newly initialized DiT LoRA plus the fixed-condition text projection. One
q is sampled on rank 0 per optimizer step and shared by all selected GPU ranks.
Strict loading rejects `best_phase_a_ckpt.pt` and
`phase_a_complete.pt`; use a checkpoint that has completed at least one Phase B
step, normally `best_ckpt.pt`.

First set `model.params.codec_path` in both Plain configs to the accepted V2
checkpoint and run the 20-step handoff smoke:

```bash
bash scripts/run_variable_rate_train.sh plain-smoke 0,1
```

Only after this succeeds, launch the full P0 run below.

```bash
bash scripts/run_variable_rate_train.sh plain 0
# Or: bash scripts/run_variable_rate_train.sh plain 0,1
```

Resume only from a P0 checkpoint:

```bash
bash scripts/run_variable_rate_train.sh plain 0 \
  --resume_checkpoint /path/to/P0/checkpoint.pt
```

## 5. Frozen-bitstream Gate B

Use the direct LIC evaluation's V2 binary tree and run
`scripts/run_variable_rate_eval.sh` on P0 using the same images. Then compare
the corresponding binary trees:

```bash
PYTHONPATH=. python scripts/compare_bitstreams.py \
  /path/to/V2_eval/q0/bin \
  /path/to/P0_eval/q0/bin
```

Repeat for q0..q5. Every file must be byte-identical before attributing a
quality difference to Plain DiT training.

CUDA, one-/two-rank DDP, real CompressAI/rANS, memory, throughput, and full
RD/RP validation must be performed on the training server; the local Windows
environment is only suitable for static and standard-library tests.
