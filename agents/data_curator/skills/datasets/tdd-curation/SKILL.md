---
name: tdd-curation
description: Probe and export the TDD panoramic dental dataset into training-ready detection, binary segmentation, and 32-class tooth segmentation subsets.
---

# TDD Curation

Use this skill when working with the local TDD dataset under `data/TDD`.

## What It Does

- probes dataset structure and label availability
- validates case matching across radiographs, masks, and bbox annotations
- exports training-ready subsets for:
  - teeth detection from bbox JSON
  - binary teeth segmentation from `teeth_mask`
  - binary maxillomandibular segmentation from `maxillomandibular`
  - 32-class permanent-tooth segmentation from `teeth_polygon.json`

## Scripts

- `scripts/probe_tdd.py`
  - inspect TDD and write JSON / Markdown reports
- `scripts/export_tdd.py`
  - create canonical manifests and task-specific exports

## Assumptions

- source images live in `data/TDD/Radiographs`
- bbox labels come from `data/TDD/Segmentation/teeth_bbox.json`
- polygon labels come from `data/TDD/Segmentation/teeth_polygon.json`
- mask files are JPEG and are thresholded into binary PNG masks during export
- 32-class segmentation exports numeric tooth labels `1..32` and skips non-numeric labels such as `A..T`

## Typical Usage

```bash
# From agents/data_curator
python skills/datasets/tdd-curation/scripts/probe_tdd.py   --dataset-root ../../data/TDD   --output-json reports/tdd_profile.json   --output-md reports/tdd_profile.md

python skills/datasets/tdd-curation/scripts/export_tdd.py   --dataset-root ../../data/TDD   --output-root exports/tdd_full_export
```

## Outputs

The export script writes:

- `manifest.json`
- `cases.jsonl`
- `segmentation_teeth_binary/`
- `segmentation_maxillomandibular_binary/`
- `segmentation_teeth_32class/`
- `detection_teeth_coco/`

Prefer `symlink` mode unless a downstream consumer requires copied files.
