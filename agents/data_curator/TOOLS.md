# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## Workspace Paths

For the data curation workspace (`agents/data_curator`):

- **Registry**: `registry/`
- **Intake**: `intake/`
- **Canonical**: `canonical/`
- **Memory**: `memory/`
- **Reports**: `reports/`
- **Exports**: `exports/`
- **Local skills**: `skills/`

## Project Context

Project-level shared skills live under the repository root `skills/`.
Only truly cross-agent utilities should stay there.

This workspace is intended for:
- dataset intake
- dataset normalization
- validation outputs
- handoff artifacts for training

## QC Report Convention

Dataset QC reports should be published to:

- `$DENTALCLAW_HOME/artifacts/results/reports/datasets_qc`

Use `json + markdown` pairs so `main` can inspect both machine-readable status and a concise human summary.

For nnUNet export runs:

- `nnUNet_delivery_*.status.json` is the authoritative completion signal.
- Do not infer completion from directory existence alone.
- Do not hand-create `dataset.json`; if it is missing, the export is still incomplete or failed.
- If the request includes a holdout split, `imagesTs/` and `labelsTs/` must both be populated before reporting export-ready status.
- For `skills/datasets/tdd-nnunet-export/scripts/export_tdd_to_nnunet.py`, the maintained direct-CLI flags are:
  - `--dataset-root $DENTALCLAW_HOME/data/TDD`
  - `--output-root $DENTALCLAW_HOME/artifacts/datasets/nnUNet`
  - `--task teeth_binary|maxillomandibular_binary|teeth_32class`
  - `--test-ratio 0.1` for a 10% holdout
- Do not use non-existent aliases such as `--class-mode` or `--holdout`.

## Current Core Local Skills

- `skills/core/source_registry/`
- `skills/core/local_dataset_intake/`
- `skills/core/dataset_prober/`
- `skills/core/cbct_qc/`
- `skills/core/case_builder/`
- `skills/core/annotation_linker/`
- `skills/core/canonical_packager/`
- `skills/core/dataset_validator/`
- `skills/core/split_manager/`
- `skills/core/task_exporter/`

## Current Dataset-Specific Skill

- `skills/datasets/tdd-curation/`
- `skills/datasets/tdd-nnunet-export/`

## Current Example Dataset

- Local TDD dataset root: `$DENTALCLAW_HOME/data/TDD`

## What Goes Here

Things like:

- preferred snapshot locations
- canonical dataset layout conventions
- export naming conventions
- environment-specific notes for large datasets
- local storage caveats and path policies

## Why Separate?

Skills are shared or reusable workflows.
This file is for environment-specific and workspace-specific notes.

---

Add whatever helps you do your job.
