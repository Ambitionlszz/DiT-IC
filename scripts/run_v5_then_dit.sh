#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:?'usage: run_v5_then_dit.sh GPU_LIST'}
if (( $# != 1 )); then
  echo "usage: run_v5_then_dit.sh GPU_LIST" >&2
  exit 2
fi

lic_config=${LIC_CONFIG:-configs/variable_rate_lic_main.yaml}
plain_config=configs/multirate_plain_dit.yaml
lic_checkpoint_dir=checkpoints/variable_rate_lic_v5_main/checkpoints
lic_checkpoint=${lic_checkpoint_dir}/last_ckpt.pt
plain_checkpoint_dir=${PLAIN_CHECKPOINT_DIR:-checkpoints/multirate_plain_dit_p0/checkpoints}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}
gate_gpu=${GATE_GPU:-${gpu_list%%,*}}
gate_output_root=${GATE_OUTPUT_ROOT:-"results/v5_then_dit_gate_$(date +%Y%m%d_%H%M%S)"}
gate_image_dir=${GATE_IMAGE_DIR:-}

if [[ ! -f "${lic_config}" ]]; then
  echo "Missing LIC config: ${lic_config}" >&2
  exit 1
fi
if [[ ! -f "${plain_config}" ]]; then
  echo "Missing Plain DiT config: ${plain_config}" >&2
  exit 1
fi

if [[ -z "${gate_image_dir}" ]]; then
  gate_image_dir="$(python - "${lic_config}" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
paths = config.get("data", {}).get("valid_path", [])
if not paths:
    raise SystemExit("LIC config has no data.valid_path; set GATE_IMAGE_DIR")
print(paths[0])
PY
)"
fi

if [[ ! -d "${gate_image_dir}" ]]; then
  echo "Gate image directory does not exist: ${gate_image_dir}" >&2
  exit 1
fi

inspect_lic_checkpoints() {
  python - "${lic_checkpoint_dir}" "${lic_config}" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
phase_a_steps = int(config["lic_train"]["phase_a"]["steps"])
phase_b_steps = int(config["lic_train"]["phase_b"]["steps"])
total_steps = phase_a_steps + phase_b_steps

if checkpoint_dir.is_dir():
    candidates = []
    last = checkpoint_dir / "last_ckpt.pt"
    if last.exists():
        candidates.append(last)
    candidates.extend(sorted(checkpoint_dir.glob("[0-9]*.pt")))
    candidates.extend(
        path
        for path in (checkpoint_dir / "phase_a_complete.pt", checkpoint_dir / "best_ckpt.pt")
        if path.exists()
    )
else:
    candidates = []

valid = []
for path in candidates:
    if not path.exists():
        continue
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"cannot read V5 checkpoint {path}: {exc}")
    if checkpoint.get("stage") != "variable_quality_codec":
        continue
    if int(checkpoint.get("schema_version", -1)) != 8:
        continue
    if checkpoint.get("variable_rate_mechanism") != (
        "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
    ):
        continue
    if int(checkpoint.get("phase_a_steps", -1)) != phase_a_steps:
        continue
    if int(checkpoint.get("phase_b_steps", -1)) != phase_b_steps:
        continue
    if "model" not in checkpoint or "state_dict_codec" not in checkpoint["model"]:
        continue
    if "ema" not in checkpoint or "state_dict_codec" not in checkpoint["ema"]:
        continue
    global_step = int(checkpoint.get("global_step", -1))
    if 0 <= global_step <= total_steps:
        if global_step == total_steps and checkpoint.get("training_phase") != (
            "pixel_perceptual_rd"
        ):
            raise SystemExit(
                f"completed V5 checkpoint is not marked as Phase B: {path}"
            )
        valid.append((global_step, path))

if not valid:
    print("missing")
    raise SystemExit(0)

global_step, path = max(
    valid,
    key=lambda item: (item[0], item[1].name == "last_ckpt.pt"),
)
if global_step == total_steps:
    print(f"complete:{path}")
else:
    print(f"resume:{path}")
PY
}

inspect_plain_checkpoints() {
  python - "${plain_checkpoint_dir}" "${lic_checkpoint}" "${plain_config}" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
import yaml

checkpoint_dir = Path(sys.argv[1])
lic_path = Path(sys.argv[2])
config_path = Path(sys.argv[3])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected_steps = int(config["train"]["max_steps"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not checkpoint_dir.is_dir():
    print("missing")
    raise SystemExit(0)

source_hash = sha256(lic_path)
candidates = []
for path in checkpoint_dir.glob("*.pt"):
    stem = path.stem
    if not stem.isdigit():
        continue
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"cannot read Plain DiT checkpoint {path}: {exc}")
    if checkpoint.get("source_checkpoint_sha256") != source_hash:
        continue
    if checkpoint.get("stage") != "multirate_reconstruction":
        raise SystemExit(f"Plain DiT checkpoint has an unexpected stage: {path}")
    if int(checkpoint.get("schema_version", -1)) != 3:
        raise SystemExit(f"Plain DiT checkpoint has an unexpected schema: {path}")
    if checkpoint.get("variable_rate_mechanism") != (
        "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"
    ):
        raise SystemExit(f"Plain DiT checkpoint has an unexpected LIC mechanism: {path}")
    train_steps = int(checkpoint.get("train_steps", -1))
    if train_steps < 0 or train_steps > expected_steps:
        raise SystemExit(f"Plain DiT checkpoint has invalid train_steps: {path}")
    candidates.append((train_steps, path))

if not candidates:
    print("missing")
    raise SystemExit(0)

train_steps, path = max(candidates, key=lambda item: item[0])
if train_steps == expected_steps:
    print(f"complete:{path}")
else:
    print(f"resume:{path}")
PY
}

run_lic_direct_eval() {
  local checkpoint=$1
  local output_dir=$2
  if [[ -f "${output_dir}/summary.json" ]]; then
    echo "Reusing LIC Gate-A evaluation: ${output_dir}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gate_gpu}" PYTHONPATH=. python -u \
    scripts/eval_variable_rate_lic_direct.py \
      --config "${lic_config}" \
      --checkpoint "${checkpoint}" \
      --image_dir "${gate_image_dir}" \
      --output_dir "${output_dir}" \
      --gpu 0 \
      --weights ema
}

run_gate_a() {
  local last_checkpoint="${lic_checkpoint}"
  local last_eval_dir="${gate_output_root}/lic_last"
  local best_checkpoint="${lic_checkpoint_dir}/best_ckpt.pt"
  local best_eval_dir="${gate_output_root}/lic_best"
  local gate_a_report="${gate_output_root}/gate_a.json"

  if [[ ! -f "${last_checkpoint}" ]]; then
    echo "Completed V5 LIC checkpoint is missing ${last_checkpoint}" >&2
    exit 1
  fi
  run_lic_direct_eval "${last_checkpoint}" "${last_eval_dir}"

  local best_args=()
  if [[ -f "${best_checkpoint}" ]]; then
    run_lic_direct_eval "${best_checkpoint}" "${best_eval_dir}"
    best_args=(
      --best-checkpoint "${best_checkpoint}"
      --best-summary "${best_eval_dir}/summary.json"
      --best-eval-dir "${best_eval_dir}"
    )
  fi

  PYTHONPATH=. python scripts/variable_rate_gate.py select-lic \
    --last-checkpoint "${last_checkpoint}" \
    --last-summary "${last_eval_dir}/summary.json" \
    --last-eval-dir "${last_eval_dir}" \
    "${best_args[@]}" \
    --output "${gate_a_report}" >/dev/null

  lic_checkpoint="$(python - "${gate_a_report}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint"])
PY
)"
  lic_eval_dir="$(python - "${gate_a_report}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["evaluation_dir"])
PY
)"
  echo "Gate A accepted ${lic_checkpoint}; report: ${gate_a_report}"
}

run_gate_b() {
  local plain_checkpoint=$1
  local p0_eval_root="${gate_output_root}/plain_p0"
  local gate_b_report="${gate_output_root}/gate_b.json"
  if [[ -f "${gate_b_report}" ]]; then
    echo "Gate B already accepted: ${gate_b_report}"
    return
  fi
  bash scripts/run_variable_rate_eval.sh \
    "${plain_checkpoint}" "${gate_image_dir}" "${gate_gpu}" \
    "${plain_config}" "${p0_eval_root}"
  PYTHONPATH=. python scripts/variable_rate_gate.py verify-p0 \
    --lic-summary "${lic_eval_dir}/summary.json" \
    --lic-eval-dir "${lic_eval_dir}" \
    --p0-root "${p0_eval_root}" \
    --output "${gate_b_report}"
}

lic_state="$(inspect_lic_checkpoints)"
case "${lic_state}" in
  complete:*)
    lic_checkpoint=${lic_state#complete:}
    checkpoint_state=complete
    ;;
  resume:*)
    lic_checkpoint=${lic_state#resume:}
    checkpoint_state="resume:${lic_checkpoint}"
    ;;
  missing)
    checkpoint_state=missing
    ;;
  *)
    echo "Refusing to start DiT: V5 checkpoint validation failed." >&2
    exit 1
    ;;
esac

case "${checkpoint_state}" in
  complete)
    echo "V5 LIC is complete; skipping LIC training."
    ;;
  missing)
    echo "No V5 LIC checkpoint found; starting V5 LIC from scratch."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-main "${gpu_list}" \
        --calibration_confirmed
    ;;
  resume:*)
    echo "V5 LIC is incomplete (${checkpoint_state#resume:} steps); resuming LIC."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh lic-main "${gpu_list}" \
        --calibration_confirmed \
        --resume_checkpoint "${lic_checkpoint}"
    ;;
  *)
    echo "Unexpected V5 checkpoint state: ${checkpoint_state}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${lic_checkpoint}" ]]; then
  echo "V5 LIC exited without ${lic_checkpoint}; refusing to start DiT." >&2
  exit 1
fi
lic_state_after="$(inspect_lic_checkpoints)"
if [[ "${lic_state_after}" != complete:* ]]; then
  echo "V5 LIC checkpoint is not complete after training; refusing to start DiT." >&2
  exit 1
fi
lic_checkpoint=${lic_state_after#complete:}
run_gate_a

plain_state="$(inspect_plain_checkpoints)"
case "${plain_state}" in
  complete:*)
    echo "Plain DiT is complete; the three-stage chain is already finished."
    plain_checkpoint=${plain_state#complete:}
    ;;
  resume:*)
    plain_resume_checkpoint=${plain_state#resume:}
    echo "V5 LIC complete; resuming Plain DiT from ${plain_resume_checkpoint}."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh plain "${gpu_list}" \
        --codec_path "${lic_checkpoint}" \
        --resume_checkpoint "${plain_resume_checkpoint}"
    ;;
  missing)
    echo "V5 LIC complete; starting Plain DiT from step 0 with ${lic_checkpoint}."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh plain "${gpu_list}" \
        --codec_path "${lic_checkpoint}"
    ;;
  *)
    echo "Unexpected Plain DiT checkpoint state: ${plain_state}" >&2
    exit 1
    ;;
esac

plain_state_after="$(inspect_plain_checkpoints)"
if [[ "${plain_state_after}" != complete:* ]]; then
  echo "Plain DiT exited without a complete checkpoint; rerun the same command to resume." >&2
  exit 1
fi
plain_checkpoint=${plain_state_after#complete:}
run_gate_b "${plain_checkpoint}"
echo "V5 LIC, Phase B, Plain DiT, and both real-rANS gates are complete."
