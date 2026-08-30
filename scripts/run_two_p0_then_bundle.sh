#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: run_two_p0_then_bundle.sh GPU" >&2
  exit 2
fi

gpu=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
default_v5_repo=$(cd "${script_dir}/.." && pwd)

v5_repo=${V5_REPO:-${default_v5_repo}}
e2e_repo=${E2E_REPO:-/data2/luosheng/code/DiT-IC-e2e-pixel}
image_dir=${GATE_IMAGE_DIR:-/data2/luosheng/data/Datasets/Kodak}
bundle_parent=${BUNDLE_OUTPUT_DIR:-/data2/luosheng}
v5_results=${V5_GATE_OUTPUT_ROOT:-${v5_repo}/results/v5_then_dit_nonblocking_20260829}
e2e_results=${E2E_GATE_OUTPUT_ROOT:-${e2e_repo}/results/e2e_pixel_then_dit_nonblocking_20260829}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}

v5_p0_dir=${v5_repo}/checkpoints/multirate_plain_dit_p0/checkpoints
e2e_p0_dir=${e2e_repo}/checkpoints/multirate_plain_dit_e2e_pixel_p0/checkpoints

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_dir "${v5_repo}"
require_dir "${e2e_repo}"
require_dir "${image_dir}"
require_file "${v5_repo}/scripts/run_v5_then_dit.sh"
require_file "${e2e_repo}/scripts/run_e2e_pixel_then_dit.sh"
require_file "${v5_repo}/configs/multirate_plain_dit.yaml"
require_file "${e2e_repo}/configs/multirate_plain_dit_e2e_pixel.yaml"

inspect_p0() {
  local label=$1
  local checkpoint_dir=$2
  local config_path=$3
  python - "${label}" "${checkpoint_dir}" "${config_path}" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

label = sys.argv[1]
checkpoint_dir = Path(sys.argv[2])
config_path = Path(sys.argv[3])
expected_steps = int(yaml.safe_load(config_path.read_text(encoding="utf-8"))["train"]["max_steps"])

candidates = []
if checkpoint_dir.is_dir():
    for path in checkpoint_dir.glob("*.pt"):
        if not path.stem.isdigit():
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SystemExit(f"{label}: cannot read {path}: {exc}")
        if checkpoint.get("stage") != "multirate_reconstruction":
            continue
        train_steps = int(checkpoint.get("train_steps", -1))
        if 0 <= train_steps <= expected_steps:
            candidates.append((train_steps, path))

if not candidates:
    print(f"{label}: MISSING (expected {expected_steps} steps; no valid numeric checkpoint)")
else:
    train_steps, path = max(candidates, key=lambda item: item[0])
    state = "COMPLETE" if train_steps == expected_steps else "INCOMPLETE"
    print(f"{label}: {state} {train_steps}/{expected_steps} latest={path}")
PY
}

echo "Saved-checkpoint diagnosis:"
inspect_p0 "V5 P0" "${v5_p0_dir}" "${v5_repo}/configs/multirate_plain_dit.yaml"
inspect_p0 "E2E P0" "${e2e_p0_dir}" "${e2e_repo}/configs/multirate_plain_dit_e2e_pixel.yaml"

if [[ "${DIAGNOSE_ONLY:-0}" == "1" ]]; then
  echo "Diagnosis only; no training or evaluation was started."
  exit 0
fi

if pgrep -f '[t]rain_nogan_ddp.py|[t]rain_variable_rate_lic_ddp.py' >/dev/null; then
  echo "A DiT/LIC training process is already running." >&2
  echo "Wait for it to finish or stop it deliberately before launching this wrapper." >&2
  exit 1
fi

run_pipeline() {
  local label=$1
  local repo=$2
  local pipeline=$3
  local results=$4
  local gate_report=${results}/gate_b.json

  if [[ -f "${gate_report}" ]]; then
    echo "${label}: existing Gate B report found; skipping training and evaluation."
    return
  fi

  echo "${label}: running resume -> six-rate real-rANS evaluation -> Gate B."
  (
    cd "${repo}"
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
    GATE_GPU="${gpu}" \
    GATE_IMAGE_DIR="${image_dir}" \
    GATE_OUTPUT_ROOT="${results}" \
      bash "${pipeline}" "${gpu}"
  )
}

validate_gate_b() {
  local label=$1
  local report=$2
  python - "${label}" "${report}" <<'PY'
import json
import sys
from pathlib import Path

label = sys.argv[1]
path = Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"{label}: missing Gate B report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
expected = {f"q{index}" for index in range(6)}
qualities = report.get("qualities", {})
if report.get("accepted") is not True:
    raise SystemExit(f"{label}: Gate B is not accepted: {path}")
if set(qualities) != expected:
    raise SystemExit(f"{label}: Gate B does not contain exactly q0-q5: {path}")
if not all(qualities[q].get("byte_identical") is True for q in expected):
    raise SystemExit(f"{label}: Gate B contains a non-identical bitstream result: {path}")
print(f"{label}: Gate B accepted with byte-identical q0-q5 bitstreams.")
PY
}

run_pipeline \
  "V5 P0" "${v5_repo}" "scripts/run_v5_then_dit.sh" "${v5_results}"
validate_gate_b "V5 P0" "${v5_results}/gate_b.json"

run_pipeline \
  "E2E P0" "${e2e_repo}" "scripts/run_e2e_pixel_then_dit.sh" "${e2e_results}"
validate_gate_b "E2E P0" "${e2e_results}/gate_b.json"

bundle_name="ditic_two_p0_report_$(date +%Y%m%d_%H%M%S)"
bundle_dir=${bundle_parent}/${bundle_name}
mkdir -p "${bundle_dir}/v5/configs" "${bundle_dir}/e2e/configs"

collect_experiment() {
  local label=$1
  local repo=$2
  local results=$3
  local lic_exp=$4
  local p0_exp=$5
  local lic_config=$6
  local p0_config=$7
  local destination=${bundle_dir}/${label}

  cp "${results}/lic_handoff.json" "${destination}/"
  cp "${results}/gate_b.json" "${destination}/"

  local lic_eval
  local quality
  for lic_eval in lic_best lic_last; do
    if [[ ! -f "${results}/${lic_eval}/summary.json" ]]; then
      continue
    fi
    mkdir -p "${destination}/${lic_eval}"
    cp "${results}/${lic_eval}/summary.json" "${destination}/${lic_eval}/"
    for quality in 0 1 2 3 4 5; do
      mkdir -p "${destination}/${lic_eval}/q${quality}"
      cp "${results}/${lic_eval}/q${quality}/metrics.json" \
        "${destination}/${lic_eval}/q${quality}/"
      cp -a "${results}/${lic_eval}/q${quality}/bin" \
        "${destination}/${lic_eval}/q${quality}/"
    done
  done

  for quality in 0 1 2 3 4 5; do
    mkdir -p "${destination}/plain_p0/q${quality}"
    cp -a "${results}/plain_p0/q${quality}/metrics" \
      "${destination}/plain_p0/q${quality}/"
    cp -a "${results}/plain_p0/q${quality}/bin" \
      "${destination}/plain_p0/q${quality}/"
  done

  cp "${repo}/checkpoints/${lic_exp}/log.txt" "${destination}/lic_train.log"
  cp "${repo}/checkpoints/${p0_exp}/log.txt" "${destination}/p0_train.log"
  cp "${repo}/${lic_config}" "${destination}/configs/"
  cp "${repo}/${p0_config}" "${destination}/configs/"
  git -C "${repo}" status --short --branch > "${destination}/git_state.txt"
  git -C "${repo}" log -5 --oneline >> "${destination}/git_state.txt"
  sha256sum "${repo}/checkpoints/${p0_exp}/checkpoints/0100000.pt" \
    > "${destination}/p0_checkpoint_sha256.txt"
}

collect_experiment \
  v5 "${v5_repo}" "${v5_results}" \
  variable_rate_lic_v5_main multirate_plain_dit_p0 \
  configs/variable_rate_lic_main.yaml configs/multirate_plain_dit.yaml

collect_experiment \
  e2e "${e2e_repo}" "${e2e_results}" \
  variable_rate_lic_e2e_pixel_main multirate_plain_dit_e2e_pixel_p0 \
  configs/variable_rate_lic_e2e_pixel.yaml configs/multirate_plain_dit_e2e_pixel.yaml

tar -C "${bundle_parent}" -czf "${bundle_dir}.tar.gz" "${bundle_name}"

echo "Both P0 experiments, six-rate evaluations, and Gate B checks are complete."
echo "Send this archive: ${bundle_dir}.tar.gz"
ls -lh "${bundle_dir}.tar.gz"
