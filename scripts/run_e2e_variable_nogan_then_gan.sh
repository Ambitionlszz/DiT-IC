#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:?'usage: run_e2e_variable_nogan_then_gan.sh GPU_LIST [EVAL_IMAGE_DIR]'}
eval_image_dir=${2:-${EVAL_IMAGE_DIR:-}}

nogan_config=${E2E_NOGAN_CONFIG:-configs/multirate_e2e_variable_nogan.yaml}
gan_config=${E2E_GAN_CONFIG:-configs/multirate_e2e_variable_gan.yaml}
result_root=${RESULT_ROOT:-results/e2e_variable_nogan_then_gan}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for required_file in "${nogan_config}" "${gan_config}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

config_value() {
  PYTHONPATH=. python - "$1" "$2" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
value = config
for key in sys.argv[2].split("."):
    value = value[int(key)] if isinstance(value, list) else value[key]
print(value)
PY
}

nogan_checkpoint_dir="$(config_value "${nogan_config}" train.output_dir)/$(config_value "${nogan_config}" train.exp_name)/checkpoints"
gan_checkpoint_dir="$(config_value "${gan_config}" train.output_dir)/$(config_value "${gan_config}" train.exp_name)/checkpoints"
if [[ -z "${eval_image_dir}" ]]; then
  eval_image_dir="$(config_value "${nogan_config}" data.valid_path.0)"
fi
if [[ ! -d "${eval_image_dir}" ]]; then
  echo "Evaluation image directory does not exist: ${eval_image_dir}" >&2
  exit 1
fi

inspect_phase() {
  PYTHONPATH=. python - "$1" "$2" "$3" "$4" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
phase = sys.argv[3]
source_path = sys.argv[4]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected_steps = int(config["train"]["max_steps"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_contract = {
    "version": "e2e_variable_rate_nogan_gan_v1",
    "phase": phase,
    "rate_weights": [
        float(value) for value in config["quality_control"]["rate_weights"]
    ],
    "freeze_codec": False,
    "loss_form": "pixel_perceptual_plus_quality_rate",
    "variable_rate_mechanism": "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2",
}
expected_source_hash = None if source_path == "-" else sha256(Path(source_path))
candidates = []
if checkpoint_dir.is_dir():
    paths = list(checkpoint_dir.glob("[0-9]*.pt"))
    paths.append(checkpoint_dir / "last_ckpt.pt")
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "multirate_reconstruction":
            raise SystemExit(f"Unexpected checkpoint stage in {checkpoint_dir}: {path}")
        if checkpoint.get("e2e_variable_rate_contract") != expected_contract:
            raise SystemExit(f"Checkpoint contract does not match {phase}: {path}")
        if checkpoint.get("source_checkpoint_sha256") != expected_source_hash:
            raise SystemExit(f"Checkpoint came from a different source: {path}")
        train_steps = int(checkpoint.get("train_steps", -1))
        if not 0 <= train_steps <= expected_steps:
            raise SystemExit(f"Checkpoint has an invalid step: {path}")
        candidates.append((train_steps, path))

if not candidates:
    print("missing")
else:
    train_steps, path = max(
        candidates,
        key=lambda item: (item[0], item[1].name == "last_ckpt.pt"),
    )
    state = "complete" if train_steps == expected_steps else "resume"
    print(f"{state}:{path}")
PY
}

require_final_checkpoint() {
  PYTHONPATH=. python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

checkpoint_path = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
phase = sys.argv[3]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if checkpoint_path.name != "last_ckpt.pt":
    raise SystemExit("The handoff must use last_ckpt.pt, not a validation-best checkpoint")
if int(checkpoint.get("train_steps", -1)) != int(config["train"]["max_steps"]):
    raise SystemExit("The handoff checkpoint is not at the configured final step")
if checkpoint.get("e2e_variable_rate_contract", {}).get("phase") != phase:
    raise SystemExit("The handoff checkpoint has the wrong E2E phase")
print(checkpoint_path)
PY
}

run_eval() {
  local checkpoint=$1
  local config=$2
  local output_dir=$3
  local eval_gpu=${gpu_list%%,*}
  bash scripts/run_variable_rate_eval.sh \
    "${checkpoint}" "${eval_image_dir}" "${eval_gpu}" "${config}" "${output_dir}"
}

nogan_state="$(inspect_phase "${nogan_checkpoint_dir}" "${nogan_config}" nogan -)"
case "${nogan_state}" in
  complete:*)
    echo "End-to-end NoGAN is already complete: ${nogan_state#complete:}"
    ;;
  resume:*)
    echo "Resuming end-to-end NoGAN from ${nogan_state#resume:}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh e2e-variable-nogan "${gpu_list}" \
        --resume_checkpoint "${nogan_state#resume:}"
    ;;
  missing)
    echo "Starting 200k end-to-end NoGAN from scratch"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh e2e-variable-nogan "${gpu_list}"
    ;;
  *)
    echo "Unexpected NoGAN checkpoint state: ${nogan_state}" >&2
    exit 1
    ;;
esac

nogan_state="$(inspect_phase "${nogan_checkpoint_dir}" "${nogan_config}" nogan -)"
if [[ "${nogan_state}" != complete:* ]]; then
  echo "NoGAN did not produce a complete checkpoint; rerun to resume." >&2
  exit 1
fi
nogan_final="${nogan_checkpoint_dir}/last_ckpt.pt"
require_final_checkpoint "${nogan_final}" "${nogan_config}" nogan
run_eval "${nogan_final}" "${nogan_config}" "${result_root}/nogan_200k_final"

gan_state="$(inspect_phase "${gan_checkpoint_dir}" "${gan_config}" gan "${nogan_final}")"
case "${gan_state}" in
  complete:*)
    echo "End-to-end GAN is already complete: ${gan_state#complete:}"
    ;;
  resume:*)
    echo "Resuming 50k end-to-end GAN from ${gan_state#resume:}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh e2e-variable-gan "${gpu_list}" \
        --codec_path "${nogan_final}" \
        --resume_checkpoint "${gan_state#resume:}"
    ;;
  missing)
    echo "Starting 50k end-to-end GAN from ${nogan_final}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh e2e-variable-gan "${gpu_list}" \
        --codec_path "${nogan_final}"
    ;;
  *)
    echo "Unexpected GAN checkpoint state: ${gan_state}" >&2
    exit 1
    ;;
esac

gan_state="$(inspect_phase "${gan_checkpoint_dir}" "${gan_config}" gan "${nogan_final}")"
if [[ "${gan_state}" != complete:* ]]; then
  echo "GAN did not produce a complete checkpoint; rerun to resume." >&2
  exit 1
fi
gan_final="${gan_checkpoint_dir}/last_ckpt.pt"
require_final_checkpoint "${gan_final}" "${gan_config}" gan
run_eval "${gan_final}" "${gan_config}" "${result_root}/gan_250k_final"

echo "End-to-end 200k NoGAN, 50k GAN, and both six-rate evaluations are complete."
