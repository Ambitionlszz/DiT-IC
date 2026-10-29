#!/usr/bin/env bash
set -euo pipefail
gpu_list=${1:?'usage: run_c_precision_pair.sh GPU0,GPU1 [launch_bt_pair.py options]'}
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname -- "${script_dir}")
cd -- "${repo_dir}"
bt_python=${BT_PYTHON:-python}
bt_run_root=${BT_RUN_ROOT:-checkpoints/c_precision_v2}
exec "${bt_python}" scripts/launch_bt_pair.py --root "${bt_run_root}" --gpus "${gpu_list}" \
  --python "${bt_python}" --config-dir "${repo_dir}/configs/bt_self_flow/precision_v2" "$@"
