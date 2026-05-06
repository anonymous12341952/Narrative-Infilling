#!/usr/bin/env bash
set -euo pipefail

# Dedicated wrapper for GPT-based evaluation.
# Default behavior: evaluate all files for the given method, in-place on data/eval_files.
# Requires API credentials in environment variables.
# gpt_eval.py auto-loads:
#   - metrics/run_evaluation.env
#   - metrics/.env
#   - ~/.env
#
# Usage:
#   ./scripts/run_gpt_eval.sh <method> [extra gpt_eval args...]
#
# Examples:
#   ./scripts/run_gpt_eval.sh teler
#   ./scripts/run_gpt_eval.sh teler --rpm 20 --batch-size 3
#   ./scripts/run_gpt_eval.sh reasoning --llm google_gemma-2-2b-it --model gpt-4o

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <method:teler|reasoning> [extra args...]"
  exit 1
fi

METHOD="$1"
shift 1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/cfg/prompts/eval_prompt.yaml}"

cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --method "$METHOD"
  --inplace
  --prompt-file "$PROMPT_FILE"
)

# If caller did not specify target scope, default to --all.
if [[ " $* " != *" --all "* && " $* " != *" --llm "* ]]; then
  COMMON_ARGS+=(--all)
fi

exec "$PYTHON_BIN" "metrics/gpt_eval.py" "${COMMON_ARGS[@]}" "$@"
