#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?"usage: run_variable_rate_eval.sh CHECKPOINT IMAGE_DIR [GPU] [CONFIG] [OUTPUT_DIR]"}
image_dir=${2:?"usage: run_variable_rate_eval.sh CHECKPOINT IMAGE_DIR [GPU] [CONFIG] [OUTPUT_DIR]"}
gpu=${3:-0}
config=${4:-configs/multirate_plain_dit.yaml}
timestamp=$(date +%Y%m%d_%H%M%S)
output_root=${5:-"results/variable_rate_${timestamp}"}
python_bin=${PYTHON_BIN:-/root/miniconda3/envs/ditic/bin/python}
if [[ ! -x "${python_bin}" ]]; then
  echo "PYTHON_BIN is not executable: ${python_bin}" >&2
  exit 2
fi

for quality in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. "${python_bin}" -u compress.py \
    --config_path "${config}" \
    --codec_path "${checkpoint}" \
    --img_path "${image_dir}" \
    --rec_path "${output_root}/q${quality}/rec" \
    --bin_path "${output_root}/q${quality}/bin" \
    --results_dir "${output_root}/q${quality}/metrics" \
    --quality_index "${quality}" \
    --gpu 0 \
    --use_ema \
    --save_img \
    --verbose_perf
done

echo "Six-rate real-rANS results: ${output_root}"
