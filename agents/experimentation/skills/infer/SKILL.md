---
name: infer
description: Compatibility inference skill for dental segmentation. Runs mask prediction and overlays from checkpoints produced by the tooth training workflow and can emit evaluation summaries when labels are available.
---

# Infer

Use this skill when existing project code expects a `skills/infer` entrypoint.

## Scripts

- `infer_2d.py`
  - Compatibility wrapper for 2D mask prediction
- `../tooth_autoinfer_nnunet/scripts/run_inference.py`
  - Recommended inference entrypoint with optional Dice and HD95 evaluation output
