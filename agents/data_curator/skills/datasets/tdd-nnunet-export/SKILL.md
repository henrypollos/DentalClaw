---
name: tdd-nnunet-export
description: Export the local TDD dataset into nnUNet v2-compatible 2D PNG segmentation datasets for the teeth binary, maxillomandibular binary, or 32-class tooth task, and write trainer handoff artifacts with one command.
---

# TDD nnUNet Export

Use this skill when you need to turn the local TDD dataset under `data/TDD` into a published nnUNet-style dataset that can be handed to a training agent.

## What It Does

- loads matched TDD radiographs and masks
- converts radiographs to lossless single-channel grayscale PNG
- thresholds JPEG masks into integer PNG labels with values `0/1`
- can also rasterize tooth polygons into integer labels `1..32`
- writes an nnUNet-compatible dataset folder:
  - `dataset.json`
  - `imagesTr/`
  - `labelsTr/`
  - optional `imagesTs/`
  - optional `labelsTs/`
- writes QA and trainer handoff files

## Supported Tasks

- `teeth_binary`
- `maxillomandibular_binary`
- `teeth_32class`

For an English request such as "train a tooth segmentation model on TDD" without an explicit class count, treat `teeth_binary` as the default single-class export.

## Scripts

- `scripts/export_tdd_to_nnunet.py`
  - main exporter
- `scripts/build_ready_training_subset.py`
  - materialize a second nnUNet dataset that keeps only QC-selected cases, such as `ready`
- `scripts/run_export.sh`
  - one-command wrapper for the local TDD dataset

## Default Output

The shell wrapper writes into a stable nnUNet workspace root:

- `artifacts/datasets/nnUNet/`

Inside it, the raw dataset folder is fixed:

- `nnUNet_raw/`

The trainer handoff defaults point to model-side artifact roots:

- `artifacts/datasets/nnUNet/nnUNet_preprocessed/`
- `artifacts/models/nnUNet/nnUNet_results/`

The exported dataset is created under:

- `artifacts/datasets/nnUNet/nnUNet_raw/DatasetXXX_NAME/`

The exporter also writes delivery reports under:

- `artifacts/datasets/nnUNet/nnUNet_delivery_DatasetXXX_NAME.json`
- `artifacts/datasets/nnUNet/nnUNet_delivery_DatasetXXX_NAME.md`

It also writes dataset QC reports under:

- `artifacts/results/reports/datasets_qc/DatasetXXX_NAME.qc.json`
- `artifacts/results/reports/datasets_qc/DatasetXXX_NAME.qc.md`

## Typical Usage

```bash
bash scripts/run_export.sh teeth_binary

bash scripts/run_export.sh maxillomandibular_binary

bash scripts/run_export.sh teeth_32class
```

Optional environment variables for the wrapper:

- `DATASET_ROOT`
- `OUTPUT_ROOT`
- `DATASET_ID`
- `THRESHOLD`
- `LIMIT`
- `TEST_RATIO`
- `SEED`

For direct CLI calls, prefer:

- `--output-root $DENTALCLAW_HOME/artifacts/datasets/nnUNet`
- `--dataset-root $DENTALCLAW_HOME/data/TDD`
- `--task teeth_binary` for the default single-class tooth-vs-background export
- `--test-ratio 0.1` for a 10% holdout

Canonical binary export example:

```bash
python scripts/export_tdd_to_nnunet.py \
  --dataset-root $DENTALCLAW_HOME/data/TDD \
  --output-root $DENTALCLAW_HOME/artifacts/datasets/nnUNet \
  --task teeth_binary \
  --test-ratio 0.1
```

The exporter also accepts:

- `--output-root $DENTALCLAW_HOME/artifacts/datasets/nnUNet/nnUNet_raw`
- `--qc-limit N`
- `--skip-qc`

for compatibility, and will normalize it to the same final dataset location.

When `TEST_RATIO` is set, the holdout split is written to:

- `imagesTs/`
- `labelsTs/`

Do not use older or invented flag names such as:

- `--class-mode`
- `--holdout`

Those are not valid for `export_tdd_to_nnunet.py`. The maintained CLI uses `--task` and `--test-ratio`.

## Notes

- This skill is intentionally specialized for TDD.
- It exports segmentation tasks only. Detection is out of scope.
- Labels are normalized to `0/1` because nnUNet expects integer segmentation maps with background `0`.
- TDD radiographs are exported as single-channel grayscale PNG because nnU-Net treats one `.png` file as three input channels only for RGB natural-image tasks.
- For `teeth_32class`, polygons are rasterized into `0..32` label PNGs using numeric tooth labels from `teeth_polygon.json`; non-numeric labels such as `A..T` are skipped and recorded in the delivery report.
- Every export now emits an export-level dataset QC report against the produced nnUNet files so `main` can read status, blocked cases, and split leakage warnings from the registry.
- Export QC is on by default. Use `--skip-qc` only when the caller explicitly disables QC for that run.
- When QC is enabled and training should use only safe cases, run `build_ready_training_subset.py` on the exported dataset plus the QC report, then hand the derived ready-only dataset to `experimentation`.
- By default, exporter-driven QC runs on the full exported dataset. Use `--qc-limit N` only when you intentionally want sampled QC for a smoke test.
- If `DATASET_ID` is not provided, the exporter auto-assigns the next available dataset id starting from `501`.
