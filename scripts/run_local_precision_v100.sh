#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bash /data2/ls/code/deployment-bootstrap/mount_linux_code.sh
export HF_HUB_OFFLINE=1
export TORCH_HOME=/data2/ls/checkpoints/torch-hub
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_dir"
exec /data2/ls/code/linux/venvs/ditic/bin/python -u scripts/launch_local_precision.py \
  --root /data2/ls/checkpoints/c_local_precision_sf_20261028 \
  --config configs/bt_self_flow/c_local_precision.json \
  --init-checkpoint /data2/ls/checkpoints/c_local_precision_sf_20261028/joint/0010000.pt \
  --gpus 0,1 --max-updates 20000
