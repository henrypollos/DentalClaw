---
name: dataset_validator
description: Run dataset QC for canonical or TDD datasets, covering completeness, correspondence, annotation integrity, plausibility, and split leakage.
---

# Dataset Validator

Use this skill after canonical packaging or before TDD export to catch integrity issues before splitting, export, or training handoff.

## Script

- `scripts/validate_dataset.py`
  - supports `--canonical-root` or `--tdd-root`
  - writes structured `json + markdown` QC reports
  - default report root: `/data/data2/yiyang/DentalClaw/artifacts/results/reports/datasets_qc`

## What It Checks

- completeness
- image/annotation correspondence
- annotation integrity and label legality
- numbering/schema consistency
- plausibility checks such as empty masks, tiny fragments, and out-of-bounds geometry
- split integrity including duplicate identifiers and potential leakage
