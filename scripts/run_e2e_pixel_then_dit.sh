#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: run_e2e_pixel_then_dit.sh GPU_LIST" >&2
  exit 2
fi

export LIC_CONFIG=${LIC_CONFIG:-configs/variable_rate_lic_e2e_pixel.yaml}
export LIC_STAGE=${LIC_STAGE:-lic-e2e-pixel}
export LIC_RUN_LABEL=${LIC_RUN_LABEL:-E2E Pixel LIC}
export LIC_CHECKPOINT_DIR=${LIC_CHECKPOINT_DIR:-checkpoints/variable_rate_lic_e2e_pixel_main/checkpoints}
export PLAIN_CONFIG=${PLAIN_CONFIG:-configs/multirate_plain_dit_e2e_pixel.yaml}
export PLAIN_STAGE=${PLAIN_STAGE:-plain-e2e-pixel}
export PLAIN_RUN_LABEL=${PLAIN_RUN_LABEL:-E2E Pixel Plain DiT}
export PLAIN_CHECKPOINT_DIR=${PLAIN_CHECKPOINT_DIR:-checkpoints/multirate_plain_dit_e2e_pixel_p0/checkpoints}

if [[ -z "${GATE_OUTPUT_ROOT:-}" ]]; then
  export GATE_OUTPUT_ROOT="results/e2e_pixel_then_dit_gate_$(date +%Y%m%d_%H%M%S)"
fi

exec bash scripts/run_v5_then_dit.sh "$@"
