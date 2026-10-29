#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export TORCH_HOME=/data2/ls/checkpoints/torch-hub
export HF_HOME=/data2/ls/checkpoints/hf-cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
cd /data2/ls/code/DiT-IC-bt-self-flow
bt_metrics_root=/data2/ls/checkpoints/bt_self_flow_metrics_20261019
bt_metrics_python=/tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python
case "${1:?Choose C, D or C_probe}" in
 C)
   exec "$bt_metrics_python" -u "$bt_metrics_root/evaluate_with_ms_ssim.py" \
     --checkpoint /data2/ls/checkpoints/bt_self_flow_v1/C/joint/0036000.pt \
     --output "$bt_metrics_root/C_36000" --limit 24
   ;;
 D)
   export CUDA_VISIBLE_DEVICES=
   export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
   exec 9>"$bt_metrics_root/D_38000.lock"
   flock 9
   if test -f "$bt_metrics_root/D_38000/summary.json"; then
     python3 - "$bt_metrics_root/D_38000/summary.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads(p.read_text())
assert m['step']==38000 and m['weights']=='ema' and m['images']==24
assert 'ms_ssim_protocol' in m and all('ms_ssim' in v for v in m['summary'].values())
assert len((p.parent/'images.jsonl').read_text().splitlines())==144
print('D metrics already completed and verified')
PY
     exit 0
   fi
   exec "$bt_metrics_python" -u "$bt_metrics_root/evaluate_with_ms_ssim.py" \
     --checkpoint /data2/ls/checkpoints/bt_self_flow_v1/D/joint/0038000.pt \
     --output "$bt_metrics_root/D_38000" --limit 24 --device cpu
   ;;
 C_probe)
   exec "$bt_metrics_python" -u "$bt_metrics_root/evaluate_with_ms_ssim.py" \
     --checkpoint /data2/ls/checkpoints/bt_self_flow_v1/C/joint/0036000.pt \
     --output "$bt_metrics_root/C_probe" --limit 1
   ;;
 *) exit 2 ;;
esac
