#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:?"usage: infer_pure_sana.sh CHECKPOINT IMAGE_DIR"}
image_dir=${2:?"usage: infer_pure_sana.sh CHECKPOINT IMAGE_DIR"}

PYTHONPATH=. python compress.py \
  --config_path configs/pure_sana_infer.yaml \
  --codec_path "$checkpoint" \
  --img_path "$image_dir" \
  --gpu 0 \
  --use_ema \
  --save_img
