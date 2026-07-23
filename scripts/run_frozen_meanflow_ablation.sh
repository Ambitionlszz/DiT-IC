#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <pure|meanflow> <config.yaml> <pure_checkpoint.pt> [nproc_per_node]" >&2
  exit 2
fi

MODE="$1"
CONFIG="$2"
INIT_CHECKPOINT="$3"
NPROC_PER_NODE="${4:-1}"

if [[ "$MODE" != "pure" && "$MODE" != "meanflow" ]]; then
  echo "Mode must be 'pure' or 'meanflow', got: $MODE" >&2
  exit 2
fi

exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  train_nogan_ddp.py \
  --config "$CONFIG" \
  --frozen_meanflow_ablation "$MODE" \
  --init_checkpoint "$INIT_CHECKPOINT"
