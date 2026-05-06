#!/usr/bin/env bash
set -euo pipefail

# Dedicated wrapper for collect_responses with defaults.
# Usage:
#   ./scripts/run_collect.sh <method> <model> [extra infill_workflow args...]
#
# Example:
#   ./scripts/run_collect.sh teler google/gemma-2-2b-it
#   ./scripts/run_collect.sh teler google/gemma-2-2b-it --limit 20 --batch_size 64

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <method:teler|reasoning> <model> [extra args...]"
  exit 1
fi

METHOD="$1"
MODEL="$2"
shift 2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TEMP="${DEFAULT_TEMP:-0.3}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-256}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Keep HF/vLLM progress output minimal; outer tqdm stays visible.
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$PROJECT_ROOT"

if ! "$PYTHON_BIN" -c "import infill_workflow" >/dev/null 2>&1; then
  echo "[info] infill_workflow package not found in current Python env."
  echo "[info] installing editable package (no dependency resolution)..."
  "$PYTHON_BIN" -m pip install -e . --no-deps
fi

exec "$PYTHON_BIN" -m infill_workflow collect_responses \
  --method "$METHOD" \
  --model "$MODEL" \
  --temperature "$DEFAULT_TEMP" \
  --batch_size "$DEFAULT_BATCH_SIZE" \
  "$@"

