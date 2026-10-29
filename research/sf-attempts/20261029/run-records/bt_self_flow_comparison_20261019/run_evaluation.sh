#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME=/data2/ls/checkpoints/torch-hub
export HF_HOME=/data2/ls/checkpoints/hf-cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
bt_eval_python=/tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python
bt_eval_root=/data2/ls/checkpoints/bt_self_flow_comparison_20261019
case "${1:?Specify C, D or baseline}" in
  C|D)
    export CUDA_VISIBLE_DEVICES=${2:?Specify physical GPU}
    cd /data2/ls/code/DiT-IC-bt-self-flow
    export PYTHONPATH="$PWD"
    exec "$bt_eval_python" eval_bt_self_flow.py \
      --checkpoint "$bt_eval_root/checkpoints/${1}_34000.pt" \
      --input /data2/ls/data/Kodak --output "$bt_eval_root/$1" \
      --weights ema --qualities 0 .2 .4 .6 .8 1 --limit 24 --device cuda --save-images
    ;;
  baseline)
    export CUDA_VISIBLE_DEVICES=${2:?Specify physical GPU}
    cd /data2/ls/code/DiT-IC-baseline512
    export PYTHONPATH="$PWD"
    exec "$bt_eval_python" "$bt_eval_root/evaluate_baseline_source.py" \
      --checkpoint "$bt_eval_root/checkpoints/baseline_30000.pt" \
      --config "$bt_eval_root/baseline_train.yaml" \
      --images /data2/ls/data/Kodak --output "$bt_eval_root/baseline" \
      --expected-step 30000 --limit 24 --qualities 0 1 2 3 4 5 --variants global
    ;;
  *) exit 2 ;;
esac
