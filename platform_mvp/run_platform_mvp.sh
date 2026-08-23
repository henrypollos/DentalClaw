#!/usr/bin/env sh
set -eu

RUNNER_PYTHON="${RUNNER_PYTHON:-$CONDA_HOME/envs/nnunetv2/bin/python}"
DENTALCLAW_PYTHON="${DENTALCLAW_PYTHON:-$CONDA_HOME/envs/nnunetv2/bin/python}"

exec "$RUNNER_PYTHON" platform_mvp/run_platform_mvp.py \
  --dentalclaw-python "$DENTALCLAW_PYTHON" \
  "$@"
