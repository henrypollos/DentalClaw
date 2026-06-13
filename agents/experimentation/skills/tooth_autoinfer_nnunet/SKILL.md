---
name: tooth_autoinfer_nnunet
description: Run tooth segmentation inference from a trained model plus dataset and task definitions, detect 2D versus 3D inputs, and export masks, overlays, and evaluation summaries including Dice, IoU, and HD95 when ground truth is available.
---

# Tooth AutoInfer nnUNet

Use this skill when you already have a trained tooth segmentation checkpoint and need reproducible inference outputs.

## Scripts

- `scripts/run_inference.py`
  - Main entrypoint for mask prediction, overlay export, and optional evaluation
  - Required direct-CLI flags:
    - `--model-path <path>`
    - `--dataset-spec <path>`
    - `--task-spec <path>`
    - `--output-dir <path>`
  - Optional flags:
    - `--input-dir <path>`
    - `--gt-dir <path>`
  - Canonical invocation pattern:
    - `/home/yiyang/miniconda3/envs/nnunetv2/bin/python /data/data2/yiyang/DentalClaw/agents/experimentation/skills/tooth_autoinfer_nnunet/scripts/run_inference.py --model-path <checkpoint> --dataset-spec <dataset_spec.json> --task-spec <task_spec.json> --output-dir <inference_dir> [--input-dir <images>] [--gt-dir <labels>]`

## Notes

- For 2D panoramic images, the script loads the checkpoint format produced by the auto-train skill.
- For 3D datasets, it delegates to `nnUNetv2_predict` when the CLI is available.
- If `--gt-dir` is provided, the script also writes an `inference_summary.json` with case-level metrics, mean Dice, mean IoU, mean HD95, and pixel accuracy.
