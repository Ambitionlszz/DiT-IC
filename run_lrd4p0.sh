#!/bin/bash
cd /data2/luosheng/code/DiT-IC
mkdir -p checkpoints/withaux_100k_lrd4p0/checkpoints
cp checkpoints/withaux_100k/checkpoints/best_ckpt.pt checkpoints/withaux_100k_lrd4p0/checkpoints/0099000.pt
ls -lh checkpoints/withaux_100k_lrd4p0/checkpoints/0099000.pt
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. conda run -n ditic torchrun --standalone --nproc_per_node=1 --nnodes=1 train_nogan_ddp.py --config configs/train_resume_withaux_lrd4p0.yaml
