#!/usr/bin/env bash
set -euo pipefail

gpu_list=${1:?'usage: run_v4_then_dit.sh GPU_LIST'}
if (( $# != 1 )); then
  echo "usage: run_v4_then_dit.sh GPU_LIST" >&2
  exit 2
fi

lic_config=${LIC_CONFIG:-configs/variable_rate_lic_main.yaml}
plain_config=configs/multirate_plain_dit.yaml
lic_checkpoint_dir=checkpoints/variable_rate_lic_v4_main/checkpoints
lic_checkpoint=${lic_checkpoint_dir}/last_ckpt.pt
plain_checkpoint_dir=${PLAIN_CHECKPOINT_DIR:-checkpoints/multirate_plain_dit_p0/checkpoints}
per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}
accumulation_steps=${ACCUMULATION_STEPS:-1}

if [[ ! -f "${lic_config}" ]]; then
  echo "Missing LIC config: ${lic_config}" >&2
  exit 1
fi
if [[ ! -f "${plain_config}" ]]; then
  echo "Missing Plain DiT config: ${plain_config}" >&2
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
        raise SystemExit(f"cannot read V4 checkpoint {path}: {exc}")
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
                f"completed V4 checkpoint is not marked as Phase B: {path}"
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
    print("complete")
else:
    print(f"resume:{path}")
PY
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
    echo "Refusing to start DiT: V4 checkpoint validation failed." >&2
    exit 1
    ;;
esac

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
lic_state_after="$(inspect_lic_checkpoints)"
if [[ "${lic_state_after}" != complete:* ]]; then
  echo "V4 LIC checkpoint is not complete after training; refusing to start DiT." >&2
  exit 1
fi
lic_checkpoint=${lic_state_after#complete:}

plain_state="$(inspect_plain_checkpoints)"
case "${plain_state}" in
  complete)
    echo "Plain DiT is complete; the three-stage chain is already finished."
    ;;
  resume:*)
    plain_resume_checkpoint=${plain_state#resume:}
    echo "V4 LIC complete; resuming Plain DiT from ${plain_resume_checkpoint}."
    PER_GPU_BATCH_SIZE="${per_gpu_batch_size}" \
    ACCUMULATION_STEPS="${accumulation_steps}" \
      bash scripts/run_variable_rate_train.sh plain "${gpu_list}" \
        --codec_path "${lic_checkpoint}" \
        --resume_checkpoint "${plain_resume_checkpoint}"
    ;;
  missing)
    echo "V4 LIC complete; starting Plain DiT from step 0 with ${lic_checkpoint}."
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

if [[ "$(inspect_plain_checkpoints)" != "complete" ]]; then
  echo "Plain DiT exited without a complete checkpoint; rerun the same command to resume." >&2
  exit 1
fi
echo "V4 LIC, Phase B, and Plain DiT training are complete."
