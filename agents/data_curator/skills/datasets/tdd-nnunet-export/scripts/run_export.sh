#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../../../.." && pwd)"

TASK="${1:-teeth_binary}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/TDD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/artifacts/datasets/nnUNet}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-$REPO_ROOT/artifacts/datasets/nnUNet/nnUNet_preprocessed}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/artifacts/models/nnUNet/nnUNet_results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CMD=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/export_tdd_to_nnunet.py"
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --preprocessed-root "$PREPROCESSED_ROOT"
  --results-root "$RESULTS_ROOT"
  --task "$TASK"
)

if [ "${DATASET_ID:-}" != "" ]; then
  CMD+=(--dataset-id "$DATASET_ID")
fi
if [ "${DATASET_NAME:-}" != "" ]; then
  CMD+=(--dataset-name "$DATASET_NAME")
fi
if [ "${THRESHOLD:-}" != "" ]; then
  CMD+=(--threshold "$THRESHOLD")
fi
if [ "${TEST_RATIO:-}" != "" ]; then
  CMD+=(--test-ratio "$TEST_RATIO")
fi
if [ "${SEED:-}" != "" ]; then
  CMD+=(--seed "$SEED")
fi
if [ "${LIMIT:-}" != "" ]; then
  CMD+=(--limit "$LIMIT")
fi

printf 'Exporting TDD task %s to %s\n' "$TASK" "$OUTPUT_ROOT"
"${CMD[@]}"
printf 'Delivery reports written under %s\n' "$(dirname "$OUTPUT_ROOT")"
