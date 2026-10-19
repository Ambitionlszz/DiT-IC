#!/usr/bin/env bash
set -euo pipefail

stage=${1:?'usage: run_bt_self_flow.sh codec|prepare|C|D|D_mix|E GPU_LIST [ARGS...]'}
gpu_list=${2:?'GPU_LIST is required'}
shift 2
for bt_arg in "$@"; do
  if [[ "${bt_arg}" == --resume || "${bt_arg}" == --resume=* ]]; then
    echo 'Use train_bt_self_flow.py with the matching phase config to resume a stage; see docs/BT_SELF_FLOW.md' >&2
    exit 2
  fi
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname -- "${script_dir}")
cd -- "${repo_dir}"
bt_python=${BT_PYTHON:-python3}
bt_root=${BT_RUN_ROOT:-${repo_dir}/checkpoints/bt_self_flow_v1}
bt_cache=${BT_CACHE_ROOT:-${HOME}/.cache/ditic}
export TORCH_HOME=${TORCH_HOME:-${bt_cache}/torch}
export HF_HOME=${HF_HOME:-${bt_cache}/huggingface}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${gpu_list}
if [[ ! "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo 'GPU_LIST must be comma-separated nonnegative indices' >&2
  exit 2
fi
IFS=',' read -r -a bt_gpus <<< "${gpu_list}"
if [[ "${stage}" == codec ]]; then
  if (( ${#bt_gpus[@]} != 2 )); then
    echo 'the shared codec recipe requires two GPUs for effective batch 8' >&2
    exit 2
  fi
  "${bt_python}" -m torch.distributed.run --standalone --nproc_per_node="${#bt_gpus[@]}" \
    train_bt_self_flow.py --config configs/bt_self_flow/common.yaml --run-root "${bt_root}" "$@"
  exit
fi
if (( ${#bt_gpus[@]} != 1 )); then
  echo 'prepare and C/D continuations require one GPU per process' >&2
  exit 2
fi
case "${stage}" in
  prepare)
    "${bt_python}" train_bt_self_flow.py --config configs/bt_self_flow/c_joint.yaml \
      --prepare-backend "${bt_root}/common/complete.pt" --output "${bt_root}/backend_init.pt" "$@"
    ;;
  C|D)
    lower=${stage,,}
    "${bt_python}" train_bt_self_flow.py --config "configs/bt_self_flow/${lower}_joint.yaml" \
      --run-root "${bt_root}" --init-checkpoint "${bt_root}/backend_init.pt" "$@"
    ;;
  D_mix|E)
    config=d_mix
    [[ "${stage}" == E ]] && config=e_self_flow
    "${bt_python}" train_bt_self_flow.py --config "configs/bt_self_flow/${config}.yaml" \
      --run-root "${bt_root}" --init-checkpoint "${bt_root}/D/joint/complete.pt" "$@"
    ;;
  *) echo "Unknown BT stage: ${stage}" >&2; exit 2 ;;
esac
