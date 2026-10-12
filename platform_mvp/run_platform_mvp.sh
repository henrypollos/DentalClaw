#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(CDPATH='' cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${CONDA_HOME:-}" ]; then
  DEFAULT_PYTHON="$CONDA_HOME/envs/nnunetv2/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
else
  DEFAULT_PYTHON="$(command -v python3 || command -v python || true)"
fi
RUNNER_PYTHON="${RUNNER_PYTHON:-$DEFAULT_PYTHON}"
DENTALCLAW_PYTHON="${DENTALCLAW_PYTHON:-$RUNNER_PYTHON}"
for interpreter in "$RUNNER_PYTHON" "$DENTALCLAW_PYTHON"; do
  if [ -z "$interpreter" ] || ! command -v "$interpreter" >/dev/null 2>&1; then
    echo "Python interpreter unavailable: $interpreter. Set RUNNER_PYTHON and DENTALCLAW_PYTHON explicitly." >&2
    exit 2
  fi
done
cd "$REPO_DIR"

# Specialized parsers do not accept the legacy --dentalclaw-python option.
export DENTALCLAW_PYTHON
exec "$RUNNER_PYTHON" platform_mvp/run_platform_mvp.py "$@"
