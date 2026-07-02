#!/bin/bash
cd /data2/luosheng/code/DiT-IC
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python compress.py \
  --config_path configs/infer_noaux.yaml \
  --codec_path "checkpoints/pure_sana_0.5lrd_bs8_100k_noauxe_noauxd_novarguide_noselfdist_textcond/checkpoints/best_ckpt.pt" \
  --img_path /data2/luosheng/data/Datasets/Kodak \
  --gpu 0 \
  --use_ema \
  --save_img
