#!/usr/bin/env bash
set -euo pipefail
bt_metrics_root=/data2/ls/checkpoints/bt_self_flow_metrics_20261019
bash "$bt_metrics_root/run_metrics.sh" C > "$bt_metrics_root/C_36000.log" 2>&1
bash "$bt_metrics_root/run_metrics.sh" D > "$bt_metrics_root/D_38000.log" 2>&1
python3 "$bt_metrics_root/build_tables.py" > "$bt_metrics_root/tables.log" 2>&1
