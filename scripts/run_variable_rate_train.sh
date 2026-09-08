#!/usr/bin/env bash
set -euo pipefail

stage=${1:?'usage: run_variable_rate_train.sh STAGE GPU_LIST [TRAIN_ARGS...]'}
gpu_list=${2:?'usage: run_variable_rate_train.sh STAGE GPU_LIST [TRAIN_ARGS...]'}

case "${stage}" in
  lic-smoke)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_smoke.yaml
    default_per_gpu_batch_size=8
    ;;
  lic-pilot)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_pilot.yaml
    default_per_gpu_batch_size=8
    ;;
  lic-main)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_main.yaml
    default_per_gpu_batch_size=8
    ;;
  lic-adv-phase2)
    trainer=train_variable_rate_lic_ddp.py
    config=configs/variable_rate_lic_adv_phase_b.yaml
    default_per_gpu_batch_size=8
    ;;
  plain-smoke)
    trainer=train_nogan_ddp.py
    config=configs/multirate_plain_dit_smoke.yaml
    default_per_gpu_batch_size=8
    ;;
  plain)
    trainer=train_nogan_ddp.py
    config=configs/multirate_plain_dit.yaml
    default_per_gpu_batch_size=8
    ;;
  plain-adv)
    trainer=train_nogan_ddp.py
    config=configs/multirate_plain_dit_adv.yaml
    default_per_gpu_batch_size=8
    ;;
  self-flow-full)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_full.yaml
    default_per_gpu_batch_size=4
    ;;
  self-flow-posttrain-control)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_posttrain_control.yaml
    default_per_gpu_batch_size=4
    ;;
  self-flow-q-only)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_q_only.yaml
    default_per_gpu_batch_size=4
    ;;
  self-flow-mix-only)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_mix_only.yaml
    default_per_gpu_batch_size=4
    ;;
  self-flow-distill-only)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_distill_only.yaml
    default_per_gpu_batch_size=4
    ;;
  self-flow-same-rate-teacher)
    trainer=train_nogan_ddp.py
    config=configs/dual_rate_self_flow_same_rate_teacher.yaml
    default_per_gpu_batch_size=4
    ;;
  *)
    echo "STAGE must be one of: lic-smoke, lic-pilot, lic-main, lic-adv-phase2, plain-smoke, plain, plain-adv, self-flow-full, self-flow-posttrain-control, self-flow-q-only, self-flow-mix-only, self-flow-distill-only, self-flow-same-rate-teacher" >&2
    exit 2
    ;;
esac

default_accumulation_steps=1
if [[ "${stage}" == self-flow-* ]]; then
  default_accumulation_steps=2
fi

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

per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-${default_per_gpu_batch_size}}
accumulation_steps=${ACCUMULATION_STEPS:-${default_accumulation_steps}}
if (( per_gpu_batch_size < 1 || accumulation_steps < 1 )); then
  echo "PER_GPU_BATCH_SIZE and ACCUMULATION_STEPS must be positive" >&2
  exit 2
fi
global_batch_size=$((per_gpu_batch_size * num_gpus))
effective_batch_size=$((global_batch_size * accumulation_steps))

echo "stage=${stage} GPUs=${gpu_list} per_gpu_batch=${per_gpu_batch_size} "\
"accumulation=${accumulation_steps} effective_batch=${effective_batch_size}"

shift 2
python_path=.
if [[ -n "${PYTHONPATH:-}" ]]; then
  python_path=".:${PYTHONPATH}"
fi

torchrun_bin=${TORCHRUN_BIN:-/root/miniconda3/envs/ditic/bin/torchrun}
if [[ ! -x "${torchrun_bin}" ]]; then
  echo "TORCHRUN_BIN is not executable: ${torchrun_bin}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONPATH="${python_path}" \
  "${torchrun_bin}" --standalone --nproc_per_node="${num_gpus}" \
  "${trainer}" \
  --config "${config}" \
  --per_gpu_batch_size "${per_gpu_batch_size}" \
  --accumulation_steps "${accumulation_steps}" \
  "$@"
