#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:-0,1,2}
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
if (( ${#gpu_ids[@]} != 3 )); then
  echo "GPU_LIST must contain exactly three GPUs, for example 0,1,2" >&2
  exit 2
fi
if [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" \
   || "${gpu_ids[0]}" == "${gpu_ids[2]}" \
   || "${gpu_ids[1]}" == "${gpu_ids[2]}" ]]; then
  echo "GPU_LIST entries must be distinct" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu_id}" >&2
    exit 2
  fi
done

config=configs/bgfr_e2e.yaml
python_dir=/root/miniconda3/envs/ditic/bin
log_dir=checkpoints/bgfr_ablation_launch_logs
mkdir -p "${log_dir}"

modes=(none decoder_self bitstream)
names=(bgfr_b0_e2e_baseline bgfr_b1_decoder_self_e2e bgfr_b2_bitstream_e2e)
pids=()

for index in 0 1 2; do
  gpu=${gpu_ids[${index}]}
  mode=${modes[${index}]}
  name=${names[${index}]}
  log_path=${log_dir}/${name}.log
  echo "Launching ${name}: GPU=${gpu}, relay=${mode}, log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PATH="${python_dir}:${PATH}" \
  PYTHONPATH=".:${PYTHONPATH:-}" \
    torchrun --standalone --nproc_per_node=1 train_nogan_ddp.py \
      --config "${config}" \
      --decoder_relay_mode "${mode}" \
      --exp_name "${name}" \
      >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2; do
  if ! wait "${pids[${index}]}"; then
    echo "${names[${index}]} failed; inspect ${log_dir}/${names[${index}]}.log" >&2
    status=1
  fi
done
exit "${status}"
