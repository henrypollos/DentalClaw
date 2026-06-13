---
name: tooth_autotrain_nnunet
description: Auto-train a tooth segmentation model from dataset and task definitions, detect 2D or 3D input, preprocess the data, run multi-round tuning, remember prior experiments for the same task, and output the current best model, inference masks, and experiment report with Dice and HD95.
---

# Tooth AutoTrain nnUNet

Use this skill when you need end-to-end tooth segmentation training from `DatasetSpec`, `TaskSpec`, and `BudgetSpec`.

## Scripts

- `scripts/run_training.py`
  - Main entrypoint for dataset analysis, preprocessing, multi-round training, best-model selection, best-model inference, experiment memory persistence, and report generation
  - Canonical absolute path: `/data/data2/yiyang/DentalClaw/agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py`
  - Required direct-CLI flags:
    - `--dataset-spec <path>`
    - `--task-spec <path>`
    - `--budget-spec <path>`
    - `--workspace <path>`
  - Optional flags:
    - `--detach`
    - `--foreground`
  - Pass `--detach` when the caller is an agent/session that may end before training finishes. This writes `launcher_status.json`, `controller_stdout.log`, and `controller_stderr.log` under the workspace and keeps the training controller alive independently of the launching chat session.
  - Canonical invocation pattern:
    - `/home/yiyang/miniconda3/envs/nnunetv2/bin/python /data/data2/yiyang/DentalClaw/agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py --dataset-spec <dataset_spec.json> --task-spec <task_spec.json> --budget-spec <budget_spec.json> --workspace <workspace_dir> --detach`
  - Do not invent ad hoc flags such as `--dataset-id`, `--dataset-root`, `--output-root`, `--plan`, `--trials`, or `--seed` when calling this maintained entrypoint directly. Those settings belong in the spec files.

## Assets

- `assets/dataset_spec.template.json`
- `assets/dataset_spec_nnunet_raw.template.json`
- `assets/task_spec_teeth_binary_nnunetv2.template.json`
- `assets/task_spec_teeth32.template.json`
- `assets/task_spec_teeth32_nnunetv2.template.json`
- `assets/budget_spec.template.json`

## Notes

- The script first detects whether the dataset is 2D or 3D.
- For 2D panoramic PNG/JPG datasets, it applies built-in preprocessing and uses the built-in `nnUNet-style` fallback trainer so the skill can run even without `nnUNetv2`.
- For `nnUNet_raw/DatasetXXX_NAME` datasets, or when `dataset_spec.extra.target_backend=nnunetv2_cli`, it skips built-in preprocessing and calls `nnUNetv2_plan_and_preprocess`, `nnUNetv2_train`, and `nnUNetv2_predict`.
- Those lower-level nnUNet commands are internal implementation details of this maintained workflow. Main-orchestration callers should invoke this DentalClaw script, not those lower-level commands directly.
- For TDD nnUNet training, the maintained workflow first runs a baseline with the default `nnUNetTrainer` and then, only for later hyperparameter-search follow-up trials, generates trainer subclasses that inherit from the current best trainer so each later trial has a traceable trainer file.
- For TDD nnUNet hyperparameter search, use this maintained workflow instead of generating ad hoc trainer/search scripts. The maintained workflow records one trial at a time, writes the current search reasoning, materializes the next trainer subclass when needed, and then schedules the next trial from the observed results.
- If training is launched from a main-agent or subagent session, do not make that short-lived agent own the only live training process. Launch `scripts/run_training.py --detach ...` and let any helper agents only monitor `run_status.json`, `search_events.jsonl`, `launcher_status.json`, and the workspace logs.
- Do not substitute `JoD/nnUNet/nnunetv2/run/run_training.py` for this maintained entrypoint when orchestrating DentalClaw workflows.
- On multi-GPU hosts, the skill checks GPU occupancy before each nnUNet CLI trial and prefers the least-busy GPU by setting `CUDA_VISIBLE_DEVICES`.
- If upstream QC was enabled, prefer the QC-filtered ready-only nnUNet dataset root as `dataset_spec.root` so training and evaluation exclude `manual_review` and `blocked` cases.
- When `BudgetSpec.max_epochs_per_trial` or the experiment config specifies `epochs`, the skill maps that value onto an nnUNet trainer class by changing the trainer's `self.num_epochs`. Built-in epoch trainers are used when available, and DentalClaw falls back to a custom dynamic trainer for other positive integers.
- For adaptive nnUNet search on TDD binary data, later trials may write trainer source files under `JoD/nnUNet/nnunetv2/training/nnUNetTrainer/` and record the generated trainer path in the workspace artifacts. Outside of those follow-up search trials, do not create new trainer source files.
- The default nnUNet artifact convention is:
  - raw datasets under `artifacts/datasets/nnUNet/nnUNet_raw`
  - preprocessing cache under `artifacts/datasets/nnUNet/nnUNet_preprocessed`
  - model outputs under `artifacts/models/nnUNet/nnUNet_results`
- Outputs include the exported best checkpoint, test-set predictions, test-set `inference_summary.json/.md`, overlays, a markdown/json experiment report, a `main_handoff.json/.md` summary for the main agent, a `search_strategy.json/.md` reasoning record, a `search_events.jsonl` trial log, the nnUNet `progress.png` training curve when available, and a persistent memory bank that is reused on later runs of the same dataset/task signature.
- Reports and inference summaries include Dice, IoU, HD95, and pixel accuracy when labels are available.
