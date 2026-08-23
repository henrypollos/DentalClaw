#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-$CONDA_HOME/envs/nnunetv2/bin/python}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_mvp_fullflow.py" --python "$PYTHON_BIN" "$@"
