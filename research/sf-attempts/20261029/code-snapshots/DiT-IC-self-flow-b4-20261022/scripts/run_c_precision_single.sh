#!/usr/bin/env bash
set -euo pipefail
backend=${1:?'usage: run_c_precision_single.sh C|D GPU0,GPU1'}
gpu_list=${2:?'usage: run_c_precision_single.sh C|D GPU0,GPU1'}
shift 2
case "${backend}" in C|D) ;; *) exit 2 ;; esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/run_c_precision_pair.sh" "${gpu_list}" --backend "${backend}" "$@"
