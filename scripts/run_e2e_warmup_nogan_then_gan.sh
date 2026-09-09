#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:?'usage: run_e2e_warmup_nogan_then_gan.sh GPU_LIST [EVAL_IMAGE_DIR]'}
eval_image_dir=${2:-${EVAL_IMAGE_DIR:-}}

warmup_config=${E2E_WARMUP_CONFIG:-configs/multirate_e2e_latent_warmup.yaml}
nogan_config=${E2E_NOGAN_CONFIG:-configs/multirate_e2e_warmup_nogan.yaml}
gan_config=${E2E_GAN_CONFIG:-configs/multirate_e2e_warmup_gan.yaml}
result_root=${RESULT_ROOT:-results/e2e_warmup_nogan_then_gan}
stop_after_nogan=${STOP_AFTER_NOGAN:-false}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${stop_after_nogan}" != true && "${stop_after_nogan}" != false ]]; then
  echo "STOP_AFTER_NOGAN must be true or false." >&2
  exit 1
fi

for required_file in "${warmup_config}" "${nogan_config}" "${gan_config}"; do
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

for config in "${warmup_config}" "${nogan_config}" "${gan_config}"; do
  if [[ "$(config_value "${config}" train.training_mode)" != phase1_warmup_e2e ]]; then
    echo "Config must use train.training_mode=phase1_warmup_e2e: ${config}" >&2
    exit 1
  fi
done

warmup_checkpoint_dir="$(config_value "${warmup_config}" train.output_dir)/$(config_value "${warmup_config}" train.exp_name)/checkpoints"
nogan_checkpoint_dir="$(config_value "${nogan_config}" train.output_dir)/$(config_value "${nogan_config}" train.exp_name)/checkpoints"
gan_checkpoint_dir="$(config_value "${gan_config}" train.output_dir)/$(config_value "${gan_config}" train.exp_name)/checkpoints"
if [[ -z "${eval_image_dir}" ]]; then
  eval_image_dir="$(config_value "${nogan_config}" data.valid_path.0)"
fi
if [[ ! -d "${eval_image_dir}" ]]; then
  echo "Evaluation image directory does not exist: ${eval_image_dir}" >&2
  exit 1
fi

inspect_warmup() {
  PYTHONPATH=. python - "$1" "$2" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_steps = int(config["train"]["max_steps"])
expected_entropy_mode = config.get("optimizer", {}).get("entropy_update_mode", "official_post_main")
candidates = []
if checkpoint_dir.is_dir():
    paths = list(checkpoint_dir.glob("[0-9]*.pt"))
    paths.extend(
        [checkpoint_dir / "phase_a_complete.pt", checkpoint_dir / "last_ckpt.pt"]
    )
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "variable_quality_codec":
            raise SystemExit(f"Unexpected warmup checkpoint stage: {path}")
        if not checkpoint.get("stop_after_phase_a", False):
            raise SystemExit(f"Checkpoint is not a latent-warmup-only run: {path}")
        checkpoint_entropy_mode = checkpoint.get("objective_contract", {}).get(
            "entropy_update_mode", "official_post_main"
        )
        if checkpoint_entropy_mode != expected_entropy_mode:
            raise SystemExit(f"Warmup checkpoint entropy mode mismatch: {path}")
        step = int(checkpoint.get("global_step", -1))
        if not 0 <= step <= expected_steps:
            raise SystemExit(f"Warmup checkpoint has an invalid step: {path}")
        candidates.append((step, path))

if not candidates:
    print("missing")
else:
    step, path = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1].name == "last_ckpt.pt",
            item[1].name == "phase_a_complete.pt",
        ),
    )
    state = (
        "complete"
        if step == expected_steps
        and path.name in ("phase_a_complete.pt", "last_ckpt.pt")
        else "resume"
    )
    print(f"{state}:{path}")
PY
}

inspect_e2e_phase() {
  PYTHONPATH=. python - "$1" "$2" "$3" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
source_path = Path(sys.argv[3])
expected_steps = int(config["train"]["max_steps"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_contract = {
    "version": "phase1_warmup_e2e_nogan_gan_v1",
    "phase": str(config["train"]["e2e_phase"]),
    "rate_weights": [
        float(value) for value in config["quality_control"]["rate_weights"]
    ],
    "freeze_codec": bool(config["model"]["params"]["freeze_codec"]),
    "loss_form": "pixel_perceptual_plus_quality_rate",
    "variable_rate_mechanism": "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2",
    "training_mode": "phase1_warmup_e2e",
    "freeze_quality_endpoints": True,
}
entropy_update_mode = config.get("optimizer", {}).get("entropy_update_mode", "official_post_main")
if entropy_update_mode != "official_post_main":
    expected_contract["entropy_update_mode"] = entropy_update_mode
expected_source_hash = sha256(source_path)
candidates = []
if checkpoint_dir.is_dir():
    paths = list(checkpoint_dir.glob("[0-9]*.pt"))
    paths.append(checkpoint_dir / "last_ckpt.pt")
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "multirate_reconstruction":
            raise SystemExit(f"Unexpected E2E checkpoint stage: {path}")
        if checkpoint.get("e2e_variable_rate_contract") != expected_contract:
            raise SystemExit(f"E2E checkpoint contract mismatch: {path}")
        if checkpoint.get("source_checkpoint_sha256") != expected_source_hash:
            raise SystemExit(f"E2E checkpoint source mismatch: {path}")
        step = int(checkpoint.get("train_steps", -1))
        if not 0 <= step <= expected_steps:
            raise SystemExit(f"E2E checkpoint has an invalid step: {path}")
        candidates.append((step, path))

if not candidates:
    print("missing")
else:
    step, path = max(
        candidates,
        key=lambda item: (item[0], item[1].name == "last_ckpt.pt"),
    )
    state = (
        "complete"
        if step == expected_steps and path.name == "last_ckpt.pt"
        else "resume"
    )
    print(f"{state}:{path}")
PY
}

run_training() {
  local stage=$1
  local config=$2
  shift 2
  TRAIN_CONFIG="${config}" \
  PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
  ACCUMULATION_STEPS="${accumulation_steps}" \
    bash scripts/run_variable_rate_train.sh "${stage}" "${gpu_list}" "$@"
}

run_e2e_eval() {
  local checkpoint=$1
  local config=$2
  local output_dir=$3
  local eval_gpu=${gpu_list%%,*}
  bash scripts/run_variable_rate_eval.sh \
    "${checkpoint}" "${eval_image_dir}" "${eval_gpu}" "${config}" "${output_dir}"
}

warmup_state="$(inspect_warmup "${warmup_checkpoint_dir}" "${warmup_config}")"
case "${warmup_state}" in
  complete:*) echo "50k latent warmup is already complete." ;;
  resume:*)
    echo "Resuming 50k latent warmup from ${warmup_state#resume:}"
    run_training e2e-latent-warmup "${warmup_config}" \
      --resume_checkpoint "${warmup_state#resume:}"
    ;;
  missing)
    echo "Starting 50k codec-only latent warmup"
    run_training e2e-latent-warmup "${warmup_config}"
    ;;
  *) echo "Unexpected warmup checkpoint state: ${warmup_state}" >&2; exit 1 ;;
esac

warmup_state="$(inspect_warmup "${warmup_checkpoint_dir}" "${warmup_config}")"
if [[ "${warmup_state}" != complete:* ]]; then
  echo "Latent warmup did not produce a complete checkpoint." >&2
  exit 1
fi
warmup_final="${warmup_checkpoint_dir}/phase_a_complete.pt"
if [[ ! -f "${warmup_final}" ]]; then
  echo "Missing canonical warmup handoff: ${warmup_final}" >&2
  exit 1
fi

warmup_eval_dir="${result_root}/latent_warmup_50k_real_rans"
eval_gpu=${gpu_list%%,*}
if ! CUDA_VISIBLE_DEVICES="${eval_gpu}" PYTHONPATH=. python -u \
    scripts/eval_variable_rate_lic_direct.py \
    --config "${warmup_config}" \
    --checkpoint "${warmup_final}" \
    --image_dir "${eval_image_dir}" \
    --output_dir "${warmup_eval_dir}" \
    --gpu 0 \
    --weights ema; then
  echo "WARNING: latent-warmup evaluation failed; continuing to NoGAN." >&2
fi

warmup_steps="$(config_value "${warmup_config}" train.max_steps)"
nogan_steps="$(config_value "${nogan_config}" train.max_steps)"
nogan_logical_steps=$((warmup_steps + nogan_steps))
nogan_state="$(inspect_e2e_phase "${nogan_checkpoint_dir}" "${nogan_config}" "${warmup_final}")"
case "${nogan_state}" in
  complete:*) echo "${nogan_steps}-step E2E NoGAN is already complete." ;;
  resume:*)
    echo "Resuming ${nogan_steps}-step E2E NoGAN from ${nogan_state#resume:}"
    run_training e2e-warmup-nogan "${nogan_config}" \
      --codec_path "${warmup_final}" \
      --resume_checkpoint "${nogan_state#resume:}"
    ;;
  missing)
    echo "Starting ${nogan_steps}-step E2E NoGAN from ${warmup_final}"
    run_training e2e-warmup-nogan "${nogan_config}" \
      --codec_path "${warmup_final}"
    ;;
  *) echo "Unexpected NoGAN checkpoint state: ${nogan_state}" >&2; exit 1 ;;
esac

nogan_final="${nogan_checkpoint_dir}/last_ckpt.pt"
nogan_state="$(inspect_e2e_phase "${nogan_checkpoint_dir}" "${nogan_config}" "${warmup_final}")"
if [[ "${nogan_state}" != complete:* || ! -f "${nogan_final}" ]]; then
  echo "NoGAN did not produce its configured final checkpoint." >&2
  exit 1
fi
if ! run_e2e_eval \
    "${nogan_final}" \
    "${nogan_config}" \
    "${result_root}/nogan_logical_${nogan_logical_steps}"; then
  if [[ "${stop_after_nogan}" == true ]]; then
    echo "WARNING: NoGAN evaluation failed after training completed." >&2
  else
    echo "WARNING: NoGAN evaluation failed; continuing to GAN." >&2
  fi
fi

if [[ "${stop_after_nogan}" == true ]]; then
  echo "50k latent warmup and ${nogan_steps}-step E2E NoGAN training are complete; evaluations were attempted."
  exit 0
fi

gan_state="$(inspect_e2e_phase "${gan_checkpoint_dir}" "${gan_config}" "${nogan_final}")"
case "${gan_state}" in
  complete:*) echo "50k E2E GAN is already complete." ;;
  resume:*)
    echo "Resuming 50k E2E GAN from ${gan_state#resume:}"
    run_training e2e-warmup-gan "${gan_config}" \
      --codec_path "${nogan_final}" \
      --resume_checkpoint "${gan_state#resume:}"
    ;;
  missing)
    echo "Starting 50k E2E GAN from ${nogan_final}"
    run_training e2e-warmup-gan "${gan_config}" \
      --codec_path "${nogan_final}"
    ;;
  *) echo "Unexpected GAN checkpoint state: ${gan_state}" >&2; exit 1 ;;
esac

gan_final="${gan_checkpoint_dir}/last_ckpt.pt"
gan_state="$(inspect_e2e_phase "${gan_checkpoint_dir}" "${gan_config}" "${nogan_final}")"
if [[ "${gan_state}" != complete:* || ! -f "${gan_final}" ]]; then
  echo "GAN did not produce its configured final checkpoint." >&2
  exit 1
fi
if ! run_e2e_eval \
    "${gan_final}" \
    "${gan_config}" \
    "${result_root}/gan_logical_250k"; then
  echo "WARNING: GAN evaluation failed after training completed." >&2
fi

echo "50k latent warmup, 150k E2E NoGAN, and 50k E2E GAN training are complete; evaluations were attempted without blocking later training."
