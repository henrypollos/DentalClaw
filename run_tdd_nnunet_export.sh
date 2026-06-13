#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$SCRIPT_DIR/agents/data_curator/skills/datasets/tdd-nnunet-export/scripts/run_export.sh" "$@"
