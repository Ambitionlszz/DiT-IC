#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?"usage: run_variable_rate_lic_direct_eval.sh CHECKPOINT IMAGE_DIR CONFIG [GPU]"}
image_dir=${2:?"usage: run_variable_rate_lic_direct_eval.sh CHECKPOINT IMAGE_DIR CONFIG [GPU]"}
config=${3:?"usage: run_variable_rate_lic_direct_eval.sh CHECKPOINT IMAGE_DIR CONFIG [GPU]"}
gpu=${4:-0}
timestamp=$(date +%Y%m%d_%H%M%S)
output_root="results/variable_rate_lic_direct_${timestamp}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. python -u \
  scripts/eval_variable_rate_lic_direct.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --image_dir "${image_dir}" \
  --output_dir "${output_root}" \
  --gpu 0 \
  --weights ema

echo "Six-rate direct LIC real-rANS results: ${output_root}"
