#!/usr/bin/env bash
set -euo pipefail

stage=${1:?'usage: run_variable_rate_train.sh STAGE GPU_LIST [TRAIN_ARGS...]'}
gpu_list=${2:?'usage: run_variable_rate_train.sh STAGE GPU_LIST [TRAIN_ARGS...]'}

case "${stage}" in
  lic-pilot)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_pilot.yaml
    ;;
  lic-main)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_main.yaml
    ;;
  plain)
    trainer=train_nogan_ddp.py
    config=configs/multirate_plain_dit.yaml
    ;;
  *)
    echo "STAGE must be one of: lic-pilot, lic-main, plain" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
num_gpus=${#gpu_ids[@]}
if (( num_gpus < 1 || num_gpus > 2 )); then
  echo "GPU_LIST must select one or two GPUs, for example 0 or 0,1" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu_id}" >&2
    exit 2
  fi
done

global_batch_size=${GLOBAL_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}
if (( global_batch_size < 1 || accumulation_steps < 1 )); then
  echo "GLOBAL_BATCH_SIZE and ACCUMULATION_STEPS must be positive" >&2
  exit 2
fi
if (( global_batch_size % num_gpus != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by the selected GPU count" >&2
  exit 2
fi
per_gpu_batch_size=$((global_batch_size / num_gpus))
effective_batch_size=$((global_batch_size * accumulation_steps))

echo "stage=${stage} GPUs=${gpu_list} per_gpu_batch=${per_gpu_batch_size} "\
"accumulation=${accumulation_steps} effective_batch=${effective_batch_size}"

shift 2
python_path=.
if [[ -n "${PYTHONPATH:-}" ]]; then
  python_path=".:${PYTHONPATH}"
fi

CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONPATH="${python_path}" \
  torchrun --standalone --nproc_per_node="${num_gpus}" \
  "${trainer}" \
  --config "${config}" \
  --global_batch_size "${global_batch_size}" \
  --accumulation_steps "${accumulation_steps}" \
  "$@"
