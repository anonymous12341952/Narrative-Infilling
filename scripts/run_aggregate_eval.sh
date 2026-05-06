#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Aggregate Evaluation Runner (edit values in this section)
# ============================================================
# Set METHOD to: teler | reasoning | both
METHOD="teler"

# Optional: set to a model stem (e.g., google_gemma-2-2b-it), or keep empty.
LLM_NAME="google_gemma-2-2b-it"

# Toggle flags: true | false
BY_DATASET="true"
COMMON_MODELS_ONLY="false"
TXT_NO_DETAIL="false"

# Optional custom input dir (leave empty to use defaults from script).
# Default behavior:
#   teler    -> data/eval_files/teler
#   reasoning-> data/eval_files/reasoning
#   both     -> uses both defaults above
INPUT_DIR=""

# Output paths (relative to repo root).
OUTPUT_JSON="data/eval_files/aggregate/evaluation_by_template.json"
OUTPUT_TXT="data/eval_files/aggregate/evaluation_by_template.txt"

# ============================================================
# End config
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_PATH="$PROJECT_ROOT/result_scripts/result_by_template.py"

cd "$PROJECT_ROOT"

ARGS=(
  --method "$METHOD"
  --output "$OUTPUT_JSON"
  --output-txt "$OUTPUT_TXT"
)

if [[ -n "$INPUT_DIR" ]]; then
  ARGS+=(--input-dir "$INPUT_DIR")
fi

if [[ -n "$LLM_NAME" ]]; then
  ARGS+=(--llm-name "$LLM_NAME")
fi

if [[ "$BY_DATASET" == "true" ]]; then
  ARGS+=(--by-dataset)
fi

if [[ "$COMMON_MODELS_ONLY" == "true" ]]; then
  ARGS+=(--common-models-only)
fi

if [[ "$TXT_NO_DETAIL" == "true" ]]; then
  ARGS+=(--txt-no-detail)
fi

echo "[info] running aggregate eval with:"
echo "       METHOD=$METHOD"
echo "       LLM_NAME=${LLM_NAME:-<all>}"
echo "       BY_DATASET=$BY_DATASET"
echo "       COMMON_MODELS_ONLY=$COMMON_MODELS_ONLY"
echo "       TXT_NO_DETAIL=$TXT_NO_DETAIL"
echo "       INPUT_DIR=${INPUT_DIR:-<default>}"
echo "       OUTPUT_JSON=$OUTPUT_JSON"
echo "       OUTPUT_TXT=$OUTPUT_TXT"

exec "$PYTHON_BIN" "$SCRIPT_PATH" "${ARGS[@]}"
