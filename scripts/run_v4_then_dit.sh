#!/usr/bin/env bash
set -euo pipefail

echo "run_v4_then_dit.sh is retained as an alias; launching the current V5 chain." >&2
exec bash "$(dirname "$0")/run_v5_then_dit.sh" "$@"
