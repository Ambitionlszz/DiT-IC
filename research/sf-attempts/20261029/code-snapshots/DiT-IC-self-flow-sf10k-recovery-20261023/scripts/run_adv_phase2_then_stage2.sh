#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:-0,1}
phase_a_checkpoint=${2:-${PHASE1_CHECKPOINT:-checkpoints/variable_rate_lic_v5_main/checkpoints/phase_a_complete.pt}}
eval_image_dir=${3:-${EVAL_IMAGE_DIR:-}}

stage1_config=${STAGE1_CONFIG:-configs/variable_rate_lic_adv_phase_b.yaml}
stage2_config=${STAGE2_CONFIG:-configs/multirate_plain_dit_adv.yaml}
result_root=${RESULT_ROOT:-results/adv_phase2_then_stage2}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for required_file in "${phase_a_checkpoint}" "${stage1_config}" "${stage2_config}"; do
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

stage1_checkpoint_dir="$(config_value "${stage1_config}" train.output_dir)/$(config_value "${stage1_config}" train.exp_name)/checkpoints"
stage2_checkpoint_dir="$(config_value "${stage2_config}" train.output_dir)/$(config_value "${stage2_config}" train.exp_name)/checkpoints"
if [[ -z "${eval_image_dir}" ]]; then
  eval_image_dir="$(config_value "${stage1_config}" data.valid_path.0)"
fi
if [[ ! -d "${eval_image_dir}" ]]; then
  echo "Evaluation image directory does not exist: ${eval_image_dir}" >&2
  exit 1
fi

validate_phase_a_checkpoint() {
  PYTHONPATH=. python - "${phase_a_checkpoint}" "${stage1_config}" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

checkpoint_path = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
phase_a_steps = int(config["lic_train"]["phase_a"]["steps"])
phase_b_steps = int(config["lic_train"]["phase_b"]["steps"])
if checkpoint.get("stage") != "variable_quality_codec":
    raise SystemExit("Phase 1 checkpoint has the wrong stage")
if int(checkpoint.get("schema_version", -1)) != 8:
    raise SystemExit("Phase 1 handoff must be a schema-version 8 checkpoint")
if int(checkpoint.get("global_step", -1)) != phase_a_steps:
    raise SystemExit("Phase 1 checkpoint is not exactly at the Phase A boundary")
if checkpoint.get("training_phase") != "pixel_perceptual_rd":
    raise SystemExit("Phase 1 checkpoint is not ready to enter Phase B")
if int(checkpoint.get("phase_a_steps", -1)) != phase_a_steps:
    raise SystemExit("Phase 1 checkpoint has a different Phase A schedule")
if int(checkpoint.get("phase_b_steps", -1)) != phase_b_steps:
    raise SystemExit("Phase 1 checkpoint has a different Phase B schedule")
print(checkpoint_path)
PY
}

inspect_stage1() {
  PYTHONPATH=. python - "${stage1_checkpoint_dir}" "${phase_a_checkpoint}" "${stage1_config}" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
phase_a_path = Path(sys.argv[2])
config = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
phase_a_steps = int(config["lic_train"]["phase_a"]["steps"])
total_steps = phase_a_steps + int(config["lic_train"]["phase_b"]["steps"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_hash = sha256(phase_a_path)
candidates = []
if checkpoint_dir.is_dir():
    paths = list(checkpoint_dir.glob("[0-9]*.pt"))
    paths.extend(checkpoint_dir / name for name in ("last_ckpt.pt", "best_ckpt.pt"))
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "variable_quality_codec":
            continue
        if int(checkpoint.get("schema_version", -1)) != 9:
            raise SystemExit(f"Unexpected non-adversarial checkpoint in {checkpoint_dir}: {path}")
        if checkpoint.get("phase_a_handoff_sha256") != source_hash:
            raise SystemExit(f"Stage 1 checkpoint came from a different Phase 1 file: {path}")
        global_step = int(checkpoint.get("global_step", -1))
        if not phase_a_steps <= global_step <= total_steps:
            raise SystemExit(f"Stage 1 checkpoint has an invalid global step: {path}")
        candidates.append((global_step, path))

if not candidates:
    print("missing")
else:
    global_step, path = max(
        candidates,
        key=lambda item: (item[0], item[1].name == "last_ckpt.pt"),
    )
    state = "complete" if global_step == total_steps else "resume"
    print(f"{state}:{path}")
PY
}

select_stage1_checkpoint() {
  PYTHONPATH=. python - "$1" "${stage1_checkpoint_dir}" "${phase_a_checkpoint}" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

final_path = Path(sys.argv[1])
checkpoint_dir = Path(sys.argv[2])
phase_a_path = Path(sys.argv[3])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_hash = sha256(phase_a_path)
best_path = checkpoint_dir / "best_ckpt.pt"
if best_path.exists():
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if (
        int(best.get("schema_version", -1)) == 9
        and best.get("phase_a_handoff_sha256") == source_hash
    ):
        print(best_path)
        raise SystemExit(0)
print(final_path)
PY
}

inspect_stage2() {
  PYTHONPATH=. python - "${stage2_checkpoint_dir}" "$1" "${stage2_config}" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
stage1_path = Path(sys.argv[2])
config = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_steps = int(config["train"]["max_steps"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_hash = sha256(stage1_path)
candidates = []
if checkpoint_dir.is_dir():
    paths = list(checkpoint_dir.glob("[0-9]*.pt"))
    paths.append(checkpoint_dir / "best_ckpt.pt")
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("stage") != "multirate_reconstruction":
            continue
        if checkpoint.get("source_checkpoint_sha256") != source_hash:
            raise SystemExit(f"Stage 2 checkpoint came from a different Stage 1 file: {path}")
        train_steps = int(checkpoint.get("train_steps", -1))
        if not 0 <= train_steps <= expected_steps:
            raise SystemExit(f"Stage 2 checkpoint has an invalid step: {path}")
        candidates.append((train_steps, path))

if not candidates:
    print("missing")
else:
    train_steps, path = max(candidates, key=lambda item: item[0])
    state = "complete" if train_steps == expected_steps else "resume"
    print(f"{state}:{path}")
PY
}

run_stage1_eval() {
  local checkpoint=$1
  local output_dir=$2
  local eval_gpu=${gpu_list%%,*}
  CUDA_VISIBLE_DEVICES="${eval_gpu}" PYTHONPATH=. python -u \
    scripts/eval_variable_rate_lic_direct.py \
      --config "${stage1_config}" \
      --checkpoint "${checkpoint}" \
      --image_dir "${eval_image_dir}" \
      --output_dir "${output_dir}" \
      --gpu 0 \
      --weights ema
}

validate_phase_a_checkpoint
stage1_state="$(inspect_stage1)"
case "${stage1_state}" in
  complete:*)
    stage1_final=${stage1_state#complete:}
    echo "Adversarial Stage 1 Phase B is already complete: ${stage1_final}"
    ;;
  resume:*)
    stage1_resume=${stage1_state#resume:}
    echo "Resuming adversarial Stage 1 Phase B from ${stage1_resume}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-adv-phase2 "${gpu_list}" \
        --calibration_confirmed \
        --resume_checkpoint "${stage1_resume}"
    ;;
  missing)
    echo "Starting adversarial Stage 1 Phase B from ${phase_a_checkpoint}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-adv-phase2 "${gpu_list}" \
        --calibration_confirmed \
        --resume_checkpoint "${phase_a_checkpoint}"
    ;;
  *)
    echo "Unexpected Stage 1 checkpoint state: ${stage1_state}" >&2
    exit 1
    ;;
esac

stage1_state="$(inspect_stage1)"
if [[ "${stage1_state}" != complete:* ]]; then
  echo "Stage 1 Phase B did not produce a complete checkpoint; rerun to resume." >&2
  exit 1
fi
stage1_final=${stage1_state#complete:}
stage1_selected="$(select_stage1_checkpoint "${stage1_final}")"
run_stage1_eval "${stage1_final}" "${result_root}/stage1_phase_b_final"
if [[ "${stage1_selected}" != "${stage1_final}" ]]; then
  run_stage1_eval "${stage1_selected}" "${result_root}/stage1_phase_b_best"
fi

stage2_state="$(inspect_stage2 "${stage1_selected}")"
case "${stage2_state}" in
  complete:*)
    stage2_final=${stage2_state#complete:}
    echo "Adversarial Stage 2 is already complete: ${stage2_final}"
    ;;
  resume:*)
    stage2_resume=${stage2_state#resume:}
    echo "Resuming adversarial Stage 2 from ${stage2_resume}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh plain-adv "${gpu_list}" \
        --codec_path "${stage1_selected}" \
        --resume_checkpoint "${stage2_resume}"
    ;;
  missing)
    echo "Starting adversarial Stage 2 from ${stage1_selected}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh plain-adv "${gpu_list}" \
        --codec_path "${stage1_selected}"
    ;;
  *)
    echo "Unexpected Stage 2 checkpoint state: ${stage2_state}" >&2
    exit 1
    ;;
esac

stage2_state="$(inspect_stage2 "${stage1_selected}")"
if [[ "${stage2_state}" != complete:* ]]; then
  echo "Stage 2 did not produce a complete checkpoint; rerun to resume." >&2
  exit 1
fi
stage2_final=${stage2_state#complete:}
eval_gpu=${gpu_list%%,*}
bash scripts/run_variable_rate_eval.sh \
  "${stage2_final}" "${eval_image_dir}" "${eval_gpu}" \
  "${stage2_config}" "${result_root}/stage2_final"

stage2_best=${stage2_checkpoint_dir}/best_ckpt.pt
if [[ -f "${stage2_best}" && "${stage2_best}" != "${stage2_final}" ]]; then
  bash scripts/run_variable_rate_eval.sh \
    "${stage2_best}" "${eval_image_dir}" "${eval_gpu}" \
    "${stage2_config}" "${result_root}/stage2_best"
fi

echo "Adversarial Stage 1 Phase B, Stage 2, and all evaluations are complete."
