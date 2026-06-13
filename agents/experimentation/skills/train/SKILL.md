---
name: train
description: Compatibility training skill for dental segmentation. Dispatches dataset and task specs into the tooth auto-train workflow, supports preprocessing plus multi-round auto-tuning, and exports the best model, inference outputs, and experiment report.
---

# Train

Use this skill when existing project code expects a `skills/train` entrypoint.

## Scripts

- `train_2d.py`
  - Compatibility wrapper for single 2D training runs with preprocessing
  - Direct-CLI flags:
    - `--data_root <path>`
    - `--output_dir <path>`
    - optional: `--backbone <name> --lr <float> --epochs <int> --img_size <int>`
- `../tooth_autotrain_nnunet/scripts/run_training.py`
  - Recommended end-to-end multi-round training entrypoint with report and memory output
  - Use the maintained spec-driven interface:
    - `--dataset-spec <path>`
    - `--task-spec <path>`
    - `--budget-spec <path>`
    - `--workspace <path>`
    - optional: `--detach`, `--foreground`
  - Do not substitute ad hoc flags like `--dataset-id`, `--dataset-root`, or `--output-root` when calling this maintained entrypoint.
