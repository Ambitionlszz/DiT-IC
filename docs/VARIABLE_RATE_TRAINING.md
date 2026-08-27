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
scratch two-phase LIC -> accepted V4 LIC -> multi-rate Plain P0
                                           -> Self-Flow feasibility experiments
```

Self-Flow is intentionally deferred until P0 passes its six-rate and frozen
bitstream gates.

## 0. Scratch initialization contract

Stage 1 does not load a DiT-IC baseline checkpoint. It initializes the complete
variable-rate LIC randomly, loads only the pretrained SANA VAE, and freezes the
VAE encoder and decoder. It uses the complete DCVC-FM control path cited by
FlowCodec: learned per-quality `q_e/q_d` feature gains plus hyperprior-predicted
content-adaptive `w_e/w_d` quantization maps. The low/high `q_e/q_d` endpoint
vectors start at one; all six qualities are produced by log-domain exponential
interpolation and separate under their different RD weights.

Each LIC config must keep `model.codec_initialization: scratch` and must not
define `model.source_checkpoint` or VAE LoRA.

## 1. Automatic two-phase LIC training

One `lic-pilot` or `lic-main` launch runs both phases without a manual restart:

1. Phase A optimizes `latent_mse + lambda_rate(q) * R` and does not run the
   frozen VAE decoder.
2. At the configured boundary the trainer resets the optimizer, scheduler,
   and EMA, writes a Phase-B-ready `phase_a_complete.pt`, and automatically
   enters Phase B.
3. Phase B drops latent MSE and optimizes
   `pixel_mse + lpips + 0.2 * dists + lambda_rate(q) * R` through the frozen
   VAE decoder. Only the LIC is updated in both phases; there is no GAN loss.

The LPIPS and DISTS training losses both receive images in `[-1, 1]`, matching
the baseline loss path. The loss weights and the two phases' lambda values are
explicit under `lic_train.phase_a` and `lic_train.phase_b`.

The pilot is 2.5k Phase A steps followed by 2.5k Phase B steps. The main run is
50k + 100k. Phase B is deliberately longer because the previous Phase-B curve
was still descending at 100k and had only a short high-learning-rate window.
A checkpoint records the current phase and both boundaries, so
resuming continues in the correct phase. Resume also requires the same scratch
initialization contract, pretrained VAE source, formula version, lambda tables,
and loss weights.

Before the main run, run the throwaway 20+20-step path once to catch loading,
validation, phase-switch, and checkpoint errors:

```bash
bash scripts/run_variable_rate_train.sh lic-smoke 0,1
```

## 2. Six-rate lambda table

Both phases use the requested lambda set `[2, 4, 8, 16, 32, 64]`, bound in
descending order `[64.0, 32.0, 16.0, 8.0, 4.0, 2.0]` from q0 through q5.
This preserves the semantic order q0=lowest BPP and q5=highest BPP while using
the standard baseline/FlowCodec form `D + lambda_rate * R`. Larger lambda
therefore means lower rate, and smaller lambda means higher quality.

The target operating range remains approximately `0.01` BPP at q0 through
`0.05` BPP at q5. Earlier numbers came from the superseded independent-table
objective and are not accepted calibration evidence for schema 8. Serialized
rANS BPP remains the final measurement because likelihood BPP excludes
container overhead.

## 3. V4 main

```bash
bash scripts/run_variable_rate_train.sh lic-main 0 --calibration_confirmed
# Or: bash scripts/run_variable_rate_train.sh lic-main 0,1 --calibration_confirmed
```

The chained launcher performs this direct LIC evaluation itself before it
starts DiT. Gate A uses serialized rANS packets and requires every q to
round-trip, `q0 < q1 < ... < q5`, a BPP span of at least `0.02`, `q0 <= 0.03`,
and `0.045 <= q5 <= 0.08`. It evaluates the latest completed candidate and
`best_ckpt.pt` (when present), then selects the accepted one by mean PSNR, with
LPIPS, DISTS, and BPP as deterministic tie-breakers. Likelihood BPP is
diagnostic only.

## 4. P0 Multi-rate Plain DiT

To run the complete V4-to-DiT handoff without monitoring the phase boundary,
use the chained launcher. It validates the V4 checkpoint before starting Plain
DiT, resumes an incomplete but valid V4 run, and refuses to launch DiT for an
invalid or incomplete checkpoint:

```bash
# One GPU; each selected GPU uses batch size 8.
PER_GPU_BATCH_SIZE=8 bash scripts/run_v4_then_dit.sh 0

# Two GPUs; each GPU still uses batch size 8.
PER_GPU_BATCH_SIZE=8 bash scripts/run_v4_then_dit.sh 0,1
```

The launcher evaluates the accepted V4 checkpoint rather than assuming a
particular filename. It scans V4 numbered checkpoints when `last_ckpt.pt` does
not yet exist, so a run interrupted before the final save resumes from the
latest valid V4 state. It also finds the latest Plain DiT numbered checkpoint whose recorded
`source_checkpoint_sha256` matches this V4 file, so rerunning the same command
resumes Plain DiT after an interruption instead of restarting it.

Pass the accepted V4 checkpoint with `--codec_path`. The config loads and
freezes only the V4 LIC, loads the pretrained SANA VAE and DiT backbones, keeps
the VAE frozen without LoRA, and
trains newly initialized DiT LoRA plus the fixed-condition text projection. One
q is sampled on rank 0 per optimizer step and shared by all selected GPU ranks.
Strict loading rejects `best_phase_a_ckpt.pt` and
`phase_a_complete.pt`; use a checkpoint that has completed at least one Phase B
step, normally `best_ckpt.pt`.

First run the 20-step handoff smoke:

```bash
bash scripts/run_variable_rate_train.sh plain-smoke 0,1 \
  --codec_path /path/to/V4_variable_rate_checkpoint.pt
```

Only after this succeeds, launch the full P0 run below.

```bash
bash scripts/run_variable_rate_train.sh plain 0 \
  --codec_path /path/to/V4_variable_rate_checkpoint.pt
# Or: bash scripts/run_variable_rate_train.sh plain 0,1 \
#   --codec_path /path/to/V4_variable_rate_checkpoint.pt
```

Resume only from a P0 checkpoint:

```bash
bash scripts/run_variable_rate_train.sh plain 0 \
  --codec_path /path/to/V4_variable_rate_checkpoint.pt \
  --resume_checkpoint /path/to/P0/checkpoint.pt
```

## 5. Frozen-bitstream Gate B

After P0 completes, the same chained launcher runs all six P0 real-rANS
evaluations and Gate B automatically checks every q's packets against the
accepted LIC evaluation. Every corresponding file must be byte-identical, and
the P0 rate ladder must remain strictly increasing before attributing a quality
difference to Plain DiT training. Both gate reports and their evaluation trees
are written under `results/v4_then_dit_gate_<timestamp>/`. Override the image
set or report directory only when needed with `GATE_IMAGE_DIR=...` or
`GATE_OUTPUT_ROOT=...`.

CUDA, one-/two-rank DDP, real CompressAI/rANS, memory, throughput, and full
RD/RP validation must be performed on the training server; the local Windows
environment is only suitable for static and standard-library tests.
