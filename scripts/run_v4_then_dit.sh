#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:?'usage: run_v4_then_dit.sh GPU_LIST'}
if (( $# != 1 )); then
  echo "usage: run_v4_then_dit.sh GPU_LIST" >&2
  exit 2
fi

lic_config=${LIC_CONFIG:-configs/variable_rate_lic_main.yaml}
lic_checkpoint=${LIC_CHECKPOINT:-checkpoints/variable_rate_lic_v4_main/checkpoints/last_ckpt.pt}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}

if [[ ! -f "${lic_config}" ]]; then
  echo "Missing LIC config: ${lic_config}" >&2
  exit 1
fi

validate_checkpoint() {
  local checkpoint_path=$1
  local config_path=$2
  python - "${checkpoint_path}" "${config_path}" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

checkpoint_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

expected_schema = 8
expected_stage = "variable_quality_codec"
expected_mechanism = "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
phase_a_steps = int(config["lic_train"]["phase_a"]["steps"])
phase_b_steps = int(config["lic_train"]["phase_b"]["steps"])
total_steps = phase_a_steps + phase_b_steps

if checkpoint.get("stage") != expected_stage:
    raise SystemExit(
        f"checkpoint stage must be {expected_stage!r}; "
        f"got {checkpoint.get('stage')!r}"
    )
if int(checkpoint.get("schema_version", -1)) != expected_schema:
    raise SystemExit(
        f"checkpoint schema must be {expected_schema}; "
        f"got {checkpoint.get('schema_version')!r}"
    )
if checkpoint.get("variable_rate_mechanism") != expected_mechanism:
    raise SystemExit("checkpoint variable-rate mechanism is not the V4 contract")
if int(checkpoint.get("phase_a_steps", -1)) != phase_a_steps:
    raise SystemExit("checkpoint Phase-A boundary does not match the V4 config")
if int(checkpoint.get("phase_b_steps", -1)) != phase_b_steps:
    raise SystemExit("checkpoint Phase-B boundary does not match the V4 config")
if "model" not in checkpoint or "state_dict_codec" not in checkpoint["model"]:
    raise SystemExit("checkpoint does not contain the live LIC codec state")
if "ema" not in checkpoint or "state_dict_codec" not in checkpoint["ema"]:
    raise SystemExit("checkpoint does not contain the EMA LIC codec state")

global_step = int(checkpoint.get("global_step", -1))
if global_step >= total_steps:
    if global_step != total_steps:
        raise SystemExit(
            f"checkpoint global_step {global_step} exceeds configured total "
            f"{total_steps}"
        )
    if checkpoint.get("training_phase") != "pixel_perceptual_rd":
        raise SystemExit("completed checkpoint is not marked as Phase B")
    print("complete")
else:
    if global_step < 0:
        raise SystemExit("checkpoint global_step is invalid")
    print(f"resume:{global_step}")
PY
}

checkpoint_state="missing"
if [[ -f "${lic_checkpoint}" ]]; then
  if checkpoint_state_output=$(validate_checkpoint "${lic_checkpoint}" "${lic_config}"); then
    checkpoint_state=${checkpoint_state_output}
  else
    echo "Refusing to start DiT: V4 checkpoint validation failed." >&2
    exit 1
  fi
fi

case "${checkpoint_state}" in
  complete)
    echo "V4 LIC is complete; skipping LIC training."
    ;;
  missing)
    echo "No V4 LIC checkpoint found; starting V4 LIC from scratch."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-main "${gpu_list}" \
        --calibration_confirmed
    ;;
  resume:*)
    echo "V4 LIC is incomplete (${checkpoint_state#resume:} steps); resuming LIC."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-main "${gpu_list}" \
        --calibration_confirmed \
        --resume_checkpoint "${lic_checkpoint}"
    ;;
  *)
    echo "Unexpected V4 checkpoint state: ${checkpoint_state}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${lic_checkpoint}" ]]; then
  echo "V4 LIC exited without ${lic_checkpoint}; refusing to start DiT." >&2
  exit 1
fi
if [[ "$(validate_checkpoint "${lic_checkpoint}" "${lic_config}")" != "complete" ]]; then
  echo "V4 LIC checkpoint is not complete after training; refusing to start DiT." >&2
  exit 1
fi

echo "V4 LIC complete; starting multi-rate Plain DiT with ${lic_checkpoint}."
PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
ACCUMULATION_STEPS="${accumulation_steps}" \
  bash scripts/run_variable_rate_train.sh plain "${gpu_list}" \
    --codec_path "${lic_checkpoint}"
