---
name: multi-agent-orchestration
description: Decompose multi-stage DentalClaw requests across Research, Data Curation, Experimentation, and Reporting; preserve exact artifact paths between stages; and return a single main-session summary.
---

# Multi-Agent Orchestration

Use this skill when a user request spans more than one specialist agent.

Examples:

- organize a dataset and then train a model
- train a model and then generate a report
- review literature and then propose an experiment
- curate data, run inference, and summarize results

## Core Rule

Main keeps ownership of the whole request.

Do not stop after one specialist stage if the user asked for an end-to-end outcome.

Do not end the first execution turn with acknowledgment text alone.

For a compound workflow request, the same turn must also do at least one concrete action:

- refresh the supervision registry
- start the Stage 1 specialist handoff
- or run the first required command locally if direct handoff is unavailable

If no tool call, handoff, or command was issued yet, the task has not started.

## Agent Map

- `research`
  - latest papers
  - literature review
  - journal monitoring
- `data_curator`
  - data intake
  - data normalization
  - task-specific export
  - nnUNet raw dataset preparation
- `experimentation`
  - training
  - nnUNet plan/preprocess/train/predict
  - ablations
  - controlled validation
- `clinical_result`
  - reporting
  - clinician-friendly result packaging

## Execution Pattern

For a compound request:

1. Restate the objective as explicit stages.
2. Refresh the supervision registry first with:
   - `/home/yiyang/miniconda3/envs/nnunetv2/bin/python /data/data2/yiyang/DentalClaw/agents/main/skills/supervision-registry/scripts/refresh_registry.py`
3. Assign each stage to the appropriate specialist role.
4. For each stage, define:
   - exact input paths
   - exact output paths
   - success criteria
   - explicit training budget when the user specifies epochs, such as mapping `训练100epoch` to `BudgetSpec.max_epochs_per_trial=100`
5. Execute or hand off the stage.
6. Verify artifacts exist before moving to the next stage.
7. Carry forward the produced artifact paths.
8. Return a single main-session summary covering the whole chain.

After the brief user-facing acknowledgment, continue immediately into Step 2 or Step 3 in the same run.

If ACP/session-based specialist handoff is unavailable, do not search for alternative similarly named scripts. Fall back directly to the maintained DentalClaw absolute-path entrypoint for that stage.

For workflows that include QC and training:

1. finish Data Curation
2. read the emitted dataset QC markdown report
3. if QC is enabled, materialize a ready-only nnUNet dataset from the QC report before training
4. report the QC conclusion and the derived training dataset in the main chat before launching training
5. launch Experimentation only after the dataset is export-ready and QC processing has finished
6. when training finishes, read the emitted `main_handoff.md` and `inference_summary.md`
7. when the experimentation stage generated a search log, also read `search_strategy.md`
8. report the training curve path, trainer, epochs, tried combinations, validation summary, and independent test metrics in the main chat
9. match the user's language in every main-session update and final summary

Stage completion is strict:

- Do not treat the mere presence of `DatasetXXX_*` directories as completion.
- Do not treat partial `imagesTr/` growth as completion.
- For nnUNet export, Stage 1 is complete only when:
  - `nnUNet_delivery_*.status.json` says `status: completed`
  - `dataset.json` exists
  - `imagesTr/` and `labelsTr/` are populated
  - if a holdout was requested, `imagesTs/` and `labelsTs/` are also populated
  - if QC is enabled, the QC report exists
- If the delivery status is still `running`, keep monitoring; do not summarize Stage 1 as finished.

## Handoff Contract

Every specialist handoff should include:

- objective
- input paths
- output paths
- required task format
- constraints or assumptions

Bad:

- "Please work on this"

Good:

- "Export `/data/data2/yiyang/DentalClaw/data/TDD` with output root `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet`; the final dataset should appear under `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_raw/DatasetXXX_NAME`; report the dataset id and output folder."

## Default Artifact Roots

- nnUNet raw datasets:
  - `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_raw`
- nnUNet preprocessing cache:
  - `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_preprocessed`
- nnUNet training results:
  - `/data/data2/yiyang/DentalClaw/artifacts/models/nnUNet/nnUNet_results`

## TDD 32-Class Recipe

For the request:

`将TDD的数据集组织成为32类分割的nnunetv2格式数据，并自动训练出一个模型。`

Use this sequence:

1. `data_curator`
   - use `skills/datasets/tdd-nnunet-export/`
   - call only the maintained exporter:
     - `/data/data2/yiyang/DentalClaw/agents/data_curator/skills/datasets/tdd-nnunet-export/scripts/export_tdd_to_nnunet.py`
   - before launching, check whether the same export request is already running; if yes, reuse and monitor that run instead of launching a second copy
   - if a shell wrapper uses `tee`, make sure `artifacts/results/logs/` exists first
   - do not treat a `tee` or log-path failure as exporter failure if the exporter process is already running
   - when calling the exporter directly, pass:
     - `--output-root /data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet`
   - the exporter has built-in duplicate protection:
     - matching running-process detection
     - request-signature file locking
   - if the exporter returns JSON with `status: already_running`, switch to status tracking and do not relaunch it
   - if the exporter returns JSON with `status: already_exists`, treat the returned dataset root as export-ready and continue to QC recap / Experimentation without rerunning export
   - never manually create, patch, or overwrite `dataset.json`; only the maintained exporter is allowed to emit final nnUNet dataset metadata
   - never declare export success based only on directory listing checks; use the delivery status file as the source of truth
   - QC is enabled by default; only pass `--skip-qc` if the user explicitly says to skip, disable, or avoid QC
   - default exporter QC runs on the full exported dataset; only pass `--qc-limit N` when the user explicitly wants a sampled smoke-test QC
   - if QC is enabled, after the exporter finishes and the QC report exists, build the ready-only training subset with:
     - `/data/data2/yiyang/DentalClaw/agents/data_curator/skills/datasets/tdd-nnunet-export/scripts/build_ready_training_subset.py`
   - never call older ad hoc scripts such as:
     - `/data/data2/yiyang/DentalClaw/agents/data_curator/tdd_to_nnuentv2_converter.py`
     - `/data/data2/yiyang/DentalClaw/agents/data_curator/finalize_conversion.py`
     - `/data/data2/yiyang/DentalClaw/agents/data_curator/fix_split.py`
   - do not invent a manual Universal-to-FDI mapping step in chat; rely on the exporter implementation and emitted delivery/QC reports
   - task: `teeth_32class`
   - if the user requests a holdout such as `10%`, pass `--test-ratio 0.1`
   - for direct exporter CLI calls, use the maintained flag names exactly:
     - `--dataset-root /data/data2/yiyang/DentalClaw/data/TDD`
     - `--output-root /data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet`
     - `--task teeth_32class`
     - `--test-ratio 0.1` when a holdout is requested
   - do not improvise old aliases such as `--class-mode` or `--holdout`
   - expected output:
     - `artifacts/datasets/nnUNet/nnUNet_raw/DatasetXXX_TDDTeeth32Class2D`
     - `artifacts/results/reports/datasets_qc/DatasetXXX_TDDTeeth32Class2D.qc.md`
   - verify:
     - `nnUNet_delivery_*.status.json` is `completed`
     - `dataset.json`
     - `imagesTr/`
     - `labelsTr/`
     - if holdout was requested, `imagesTs/` and `labelsTs/`
     - QC report exists and is readable
2. `experimentation`
   - use `skills/tooth_autotrain_nnunet/`
   - point `dataset_spec.root` at the QC-filtered ready-only `DatasetXXX_*` when QC is enabled
   - if QC was explicitly skipped, point `dataset_spec.root` at the original export dataset
   - set nnUNet model roots to:
     - `artifacts/datasets/nnUNet/nnUNet_preprocessed`
     - `artifacts/models/nnUNet/nnUNet_results`
   - before launch, materialize:
     - `dataset_spec.json`
     - `task_spec.json`
     - `budget_spec.json`
     - a workspace directory under `/data/data2/yiyang/DentalClaw/artifacts`
   - call the maintained entrypoint only with:
     - `--dataset-spec <dataset_spec.json>`
     - `--task-spec <task_spec.json>`
     - `--budget-spec <budget_spec.json>`
     - `--workspace <workspace_dir>`
     - `--detach` when launching from a session that may end
   - do not improvise direct flags like:
     - `--dataset-id`
     - `--dataset-root`
     - `--output-root`
     - `--plan`
     - `--trials`
     - `--seed`
     when calling `scripts/run_training.py`; these belong in the spec files, not the CLI
   - when the user requests a training length such as `100epoch`, set `BudgetSpec.max_epochs_per_trial=100`
   - verify:
     - preprocess/train logs exist
     - model artifacts exist under `nnUNet_results`
     - `best_inference/inference_summary.md` exists
     - `main_handoff.md` exists
3. `main`
   - summarize:
     - dataset id
     - dataset folder
     - QC report path and QC conclusion
     - ready-only training dataset path when QC is enabled
     - model result folder
     - requested epochs and resolved nnUNet trainer
     - training curve path (`progress.png`) when available
     - test inference summary path
     - test mean Dice / HD95 / pixel accuracy when available
     - commands used or generated
     - caveats
   - after each major stage, refresh the registry again with the absolute-path command above before reporting status

## TDD Binary Tooth Recipe

For the English request:

`Based on nnUNet, use the TDD dataset to train a tooth segmentation model.`

Use this sequence unless the user explicitly asks for a different class structure:

1. `main`
   - interpret `tooth segmentation model` as **single-class tooth-vs-background segmentation**
   - keep the user-facing chat in English
   - enable QC by default
   - if the user explicitly says `skip QC`, `without QC`, `do not run QC`, or equivalent, disable QC for this run
   - set the default search budget to:
     - `BudgetSpec.max_trials=5`
   - keep the nnUNet trainer fixed to:
     - `nnUNetTrainer`
2. `data_curator`
   - use `skills/datasets/tdd-nnunet-export/`
   - export:
     - `--dataset-root /data/data2/yiyang/DentalClaw/data/TDD`
     - `--task teeth_binary`
     - `--output-root /data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet`
     - default to `--test-ratio 0.1` unless the user explicitly requests a different holdout
     - pass `--skip-qc` only when QC was explicitly disabled by the user
   - do not translate these to ad hoc names such as `--class-mode binary` or `--holdout 0.1`; those are not valid exporter flags
   - expected output:
     - `artifacts/datasets/nnUNet/nnUNet_raw/DatasetXXX_TDDTeethBinary2D`
     - `artifacts/results/reports/datasets_qc/DatasetXXX_TDDTeethBinary2D.qc.md`
   - verify:
      - `nnUNet_delivery_*.status.json` is `completed`
      - `dataset.json`
      - `imagesTr/`
      - `labelsTr/`
      - `imagesTs/`
      - `labelsTs/`
      - if QC is enabled, the QC report exists and is readable
   - if QC is enabled:
     - read the QC report
     - do not start training until QC report generation is complete
     - build the ready-only training subset with:
       - `/data/data2/yiyang/DentalClaw/agents/data_curator/skills/datasets/tdd-nnunet-export/scripts/build_ready_training_subset.py --dataset-root <exported_dataset_root> --qc-report <qc_report_json> --output-root /data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet`
     - use only the `ready` cases for training and evaluation
     - if the derived ready-only subset has zero ready training cases, stop and report the blocker instead of launching Experimentation
3. `experimentation`
   - use `skills/tooth_autotrain_nnunet/`
   - invoke the maintained entrypoint at the absolute path:
     - `/data/data2/yiyang/DentalClaw/agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py`
   - do not search the repository for `run_training.py`
   - do not substitute:
     - `/data/data2/yiyang/JoD/nnUNet/nnunetv2/run/run_training.py`
     - `nnUNetv2_train`
     - `nnUNetv2_plan_and_preprocess`
     for the maintained DentalClaw orchestration entrypoint
   - when launching a real training run from main/subagent orchestration, pass `--detach` so the training controller keeps running even if the launching agent session ends early
   - after launch, treat `launcher_status.json`, `run_status.json`, `search_events.jsonl`, `controller_stdout.log`, and `controller_stderr.log` in the workspace as the monitoring source of truth
   - continue supervision after launch: do not end with the launch response alone. Keep checking the workspace until the controller records `completed` or `failed`, and use `history.json` plus `search_events.jsonl` to confirm that later hyperparameter-search trials were actually scheduled
   - if the controller session is aborted or disappears, stop polling that dead session id and reconcile using the workspace status files plus the process table
   - helper subagents may monitor those files, but do not make a short-lived monitoring subagent the owner of the live training process
   - do not create or run ad hoc search scripts or one-shot trial shell scripts; the legacy paths under `artifacts/experiments/teeth_binary_2d/`, `agents/main/scripts/training_trial_*.sh`, and `artifacts/results/training_runs/run_baseline_*.sh` have been retired
   - the only allowed custom file creation during experimentation is a follow-up hyperparameter-search `nnUNetTrainer` subclass under `/data/data2/yiyang/JoD/nnUNet/nnunetv2/training/nnUNetTrainer/`; do not create any other ad hoc launcher, wrapper, export helper, or manual patch script
   - point `dataset_spec.root` at the ready-only binary `DatasetXXX_*` when QC is enabled
   - if QC was skipped, point `dataset_spec.root` at the original exported binary dataset
   - use:
     - `assets/dataset_spec_nnunet_raw.template.json`
     - `assets/task_spec_teeth_binary_nnunetv2.template.json`
     - `assets/budget_spec.template.json`
   - use up to `5` recorded trials by default
   - for the TDD binary nnUNet workflow, start with the default `nnUNetTrainer` on `fold=all`, then let the maintained workflow create inherited trainer subclasses for later hyperparameter-search trials instead of five-fold CV
   - when GPU auto-selection is enabled, pick the least-busy GPU before each trial and report which GPU was used
   - verify:
     - preprocess/train logs exist
     - model artifacts exist under `nnUNet_results`
     - `best_inference/inference_summary.md` exists
     - `search_strategy.md` exists
     - `main_handoff.md` exists
4. `main`
  - monitor:
    - `workspace/**/run_status.json`
    - `registry/task_runs.json`
  - if a cleanup or kill command fails with `SIGKILL` twice in a row, stop issuing the same command and inspect artifact state plus process state instead
  - summarize in English:
      - dataset id and binary task name
      - whether QC was enabled or skipped
      - QC conclusion
      - ready-only training dataset path and ready-case counts when QC is enabled
      - trainer name
     - max trial budget
     - tried training combinations
     - search rationale from `search_strategy.md`
     - recorded training time / workflow time when available
     - GPU snapshot / GPU-related runtime notes when available
     - validation mean Dice / IoU
     - test mean Dice / IoU
     - training curve path when available
     - final best model path
     - key logs and report files
   - do not claim the search is complete until `main_handoff.md` exists

## Reporting Back

The final main-session reply should say:

- what stages ran
- which specialist role owned each stage
- what artifacts were produced
- what remains pending

If direct specialist handoff is unavailable in the runtime, follow the specialist skill locally but keep the same staged structure in the reply.
