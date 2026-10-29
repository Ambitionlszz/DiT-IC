#!/usr/bin/env bash
set -euo pipefail
cd /data2/ls/code/DiT-IC-bt-self-flow
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_HOME=/data2/ls/checkpoints/torch-hub
export HF_HOME=/data2/ls/checkpoints/hf-cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
exec /tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python \
  scripts/launch_bt_c_ablation.py \
  --root /data2/ls/checkpoints/bt_self_flow_lowrate_20261019 \
  --gpus 0,1 >> /data2/ls/checkpoints/bt_self_flow_lowrate_20261019/supervisor.log 2>&1
