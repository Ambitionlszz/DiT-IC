#!/usr/bin/env bash
set -euo pipefail

E2E_WARMUP_CONFIG=configs/multirate_e2e_latent_warmup_historical_entropy.yaml \
E2E_NOGAN_CONFIG=configs/multirate_e2e_warmup_nogan_historical_entropy_200k.yaml \
RESULT_ROOT=results/e2e_historical_entropy_warmup_nogan \
STOP_AFTER_NOGAN=true \
PER_GPU_BATCH_SIZE=8 \
ACCUMULATION_STEPS=1 \
  bash scripts/run_e2e_warmup_nogan_then_gan.sh "$@"
