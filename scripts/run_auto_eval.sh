#!/usr/bin/env bash
set -euo pipefail

# Dedicated wrapper for automatic metrics evaluation.
# Default behavior: evaluate all files for the given method.
#
# Usage:
#   ./scripts/run_auto_eval.sh <method> [extra auto_eval args...]
#
# Examples:
#   ./scripts/run_auto_eval.sh teler
#   ./scripts/run_auto_eval.sh reasoning --batch_size 32
#   ./scripts/run_auto_eval.sh teler --input_excel google_gemma-2-2b-it.xlsx

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <method:teler|reasoning> [extra args...]"
  exit 1
fi

METHOD="$1"
shift 1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --method "$METHOD"
)

# If caller did not specify target scope, default to --all.
if [[ " $* " != *" --all "* && " $* " != *" --input_excel "* ]]; then
  COMMON_ARGS+=(--all)
fi

exec "$PYTHON_BIN" "metrics/auto_eval.py" "${COMMON_ARGS[@]}" "$@"
