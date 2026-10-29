#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME=/data2/ls/checkpoints/torch-hub
export HF_HOME=/data2/ls/checkpoints/hf-cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /data2/ls/code/DiT-IC-bt-self-flow
exec /tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python -u \
  scripts/launch_bt_pair.py --root /data2/ls/checkpoints/bt_self_flow_v1 --gpus 0,1
