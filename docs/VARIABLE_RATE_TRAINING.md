# Variable-rate LIC and Multi-rate Plain DiT training

All commands run from the repository root. Replace checkpoint/model/data paths
in the YAML files before launching. The same launcher supports one or two GPUs.
Every selected GPU uses batch size 8 and accumulation defaults to 1:

```bash
# One GPU: per-GPU batch 8, global batch 8.
bash scripts/run_variable_rate_train.sh lic-pilot 0

# Two GPUs: per-GPU batch 8, global batch 16.
bash scripts/run_variable_rate_train.sh lic-pilot 0,1
```

For an explicit override, prefix the command with values such as
`PER_GPU_BATCH_SIZE=4 ACCUMULATION_STEPS=2`. The available stages are
`lic-pilot`, `lic-main`, and `plain`.

## 0. V0 identity gate

Set `model.source_checkpoint` in `configs/variable_rate_lic_pilot.yaml` to the
M0 Pure/Plain checkpoint. Keep `source_use_ema` consistent with the M0 weights
that define the baseline.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/verify_variable_rate_v0.py \
  --config configs/variable_rate_lic_pilot.yaml \
  --image /path/to/Kodak/01.png \
  --gpu 0
```

Do not start training unless forward tensors, likelihoods, rANS payloads, and
decoded latents are identical for q0..q5.

## 1. V1 pilot

The pilot uses the provisional lambda endpoints `0.625` and `10.0`, derived
from the old `D + 0.4 R` scale only as a starting point. They are not calibrated
constants.

```bash
# Choose either one GPU...
bash scripts/run_variable_rate_train.sh lic-pilot 0

# ...or two GPUs.
bash scripts/run_variable_rate_train.sh lic-pilot 0,1
```

Evaluate all six real streams:

```bash
bash scripts/run_variable_rate_eval.sh \
  checkpoints/variable_rate_lic_v1_pilot/checkpoints/best_ckpt.pt \
  /data2/luosheng/data/Datasets/Kodak \
  0 \
  configs/multirate_plain_dit.yaml
```

Inspect real BPP ordering and per-q quality. Adjust `lambda_min/lambda_max` at
most once in `configs/variable_rate_lic_main.yaml`, then explicitly set
`calibration_confirmed: true`. The main run is guarded against launching before
that confirmation and must still use the original M0 checkpoint, not a
cherry-picked pilot checkpoint.

## 2. V1 main

```bash
bash scripts/run_variable_rate_train.sh lic-main 0
# Or: bash scripts/run_variable_rate_train.sh lic-main 0,1
```

Run the six-rate evaluation again. Gate A requires every q to round-trip and
mean real BPP to satisfy `q0 < q1 < ... < q5`. Likelihood BPP is diagnostic
only.

## 3. P0 Multi-rate Plain DiT

Set `model.params.codec_path` in `configs/multirate_plain_dit.yaml` to the
accepted V1 checkpoint. The config loads the M0 VAE decoder LoRA weights but
freezes them, freezes the complete LIC and text projection, and trains only DiT
LoRA. One q is sampled on rank 0 per optimizer step and shared by all selected
GPU ranks.

```bash
bash scripts/run_variable_rate_train.sh plain 0
# Or: bash scripts/run_variable_rate_train.sh plain 0,1
```

Resume only from a P0 checkpoint:

```bash
bash scripts/run_variable_rate_train.sh plain 0 \
  --resume_checkpoint /path/to/P0/checkpoint.pt
```

## 4. Frozen-bitstream Gate B

Run `scripts/run_variable_rate_eval.sh` once with V1 and once with P0 on the
same images. Then compare the corresponding binary trees:

```bash
PYTHONPATH=. python scripts/compare_bitstreams.py \
  /path/to/V1_eval/q0/bin \
  /path/to/P0_eval/q0/bin
```

Repeat for q0..q5. Every file must be byte-identical before attributing a
quality difference to Plain DiT training.

CUDA, one-/two-rank DDP, real CompressAI/rANS, memory, throughput, and full
RD/RP validation must be performed on the training server; the local Windows
environment is only suitable for static and standard-library tests.
