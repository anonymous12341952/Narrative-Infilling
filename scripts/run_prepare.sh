#!/usr/bin/env bash
set -euo pipefail

# Dedicated wrapper for prepare_excel with fixed dataset path and defaults.
# Usage:
#   ./scripts/run_prepare.sh <method> <model> [extra infill_workflow args...]
#
# Example:
#   ./scripts/run_prepare.sh teler google/gemma-2-2b-it
#   ./scripts/run_prepare.sh reasoning google/gemma-2-2b-it --limit 10

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <method:teler|reasoning> <model> [extra args...]"
  exit 1
fi

METHOD="$1"
MODEL="$2"
shift 2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_CSV="$PROJECT_ROOT/dataset/infilling_dataset.csv"
DEFAULT_TEMP="${DEFAULT_TEMP:-0.3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c "import infill_workflow" >/dev/null 2>&1; then
  echo "[info] infill_workflow package not found in current Python env."
  echo "[info] installing editable package (no dependency resolution)..."
  "$PYTHON_BIN" -m pip install -e . --no-deps
fi

exec "$PYTHON_BIN" -m infill_workflow prepare_excel \
  --method "$METHOD" \
  --dataset-csv "$DATASET_CSV" \
  --model "$MODEL" \
  --temperature "$DEFAULT_TEMP" \
  "$@"

