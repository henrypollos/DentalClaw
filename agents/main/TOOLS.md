# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

## DentalClaw Agent Map

Configured specialist agents:

- `research` → `$DENTALCLAW_HOME/agents/research`
- `data_curator` / `Data Curation` → `$DENTALCLAW_HOME/agents/data_curator`
- `experimentation` / `Experimentation` → `$DENTALCLAW_HOME/agents/experimentation`
- `clinical_result` / `Reporting` → `$DENTALCLAW_HOME/agents/clinical_result`

## Routing Defaults

Use these defaults unless the user explicitly says otherwise:

- literature, latest papers, journal survey → `research`
- dataset intake, dataset formatting, nnUNet export, split generation → `data_curator`
- model training, nnUNet training, ablation, tuning, inference benchmark → `experimentation`
- clinical or human-readable result summary → `clinical_result`

## Shared Artifact Conventions

Important project handoff roots:

- nnUNet raw datasets: `$DENTALCLAW_HOME/artifacts/datasets/nnUNet/nnUNet_raw`
- nnUNet preprocessing cache: `$DENTALCLAW_HOME/artifacts/datasets/nnUNet/nnUNet_preprocessed`
- nnUNet training outputs: `$DENTALCLAW_HOME/artifacts/models/nnUNet/nnUNet_results`

When coordinating a cross-agent workflow, prefer these roots over ad hoc output folders.

## Supervision Registry

Project supervision snapshots live under:

- `$DENTALCLAW_HOME/registry/datasets.json`
- `$DENTALCLAW_HOME/registry/dataset_qc_reports.json`
- `$DENTALCLAW_HOME/registry/models.json`
- `$DENTALCLAW_HOME/registry/task_runs.json`
- `$DENTALCLAW_HOME/registry/lineage.json`
- `$DENTALCLAW_HOME/registry/overview.md`

Dataset QC reports themselves live under:

- `$DENTALCLAW_HOME/artifacts/results/reports/datasets_qc`

Training handoff summaries for main live under:

- `$DENTALCLAW_HOME/artifacts/results/reports/training_runs`

Refresh them with:

- `$CONDA_HOME/envs/nnunetv2/bin/python $DENTALCLAW_HOME/agents/main/skills/supervision-registry/scripts/refresh_registry.py`

## TDD Workflow Convention

For the local TDD dataset:

- source dataset: `$DENTALCLAW_HOME/data/TDD`
- 32-class nnUNet export is a Data Curation task
- use only the maintained exporter at `$DENTALCLAW_HOME/agents/data_curator/skills/datasets/tdd-nnunet-export/scripts/export_tdd_to_nnunet.py`
- when calling that exporter directly, the canonical flags are:
  - `--dataset-root $DENTALCLAW_HOME/data/TDD`
  - `--output-root $DENTALCLAW_HOME/artifacts/datasets/nnUNet`
  - `--task teeth_binary` or `--task teeth_32class`
  - `--test-ratio 0.1` for a 10% holdout
- do not invent old aliases such as `--class-mode` or `--holdout`; they are not accepted by the maintained exporter
- before launching the exporter, check whether the same request is already running; if it is, reuse that run instead of launching another one
- if a wrapper uses `tee`, create `artifacts/results/logs/` first; a log-path failure is not a reason to restart the exporter
- if the exporter returns JSON with `status: already_running`, treat that as success-plus-reuse, not as a failure
- if the exporter returns JSON with `status: already_exists`, treat the returned dataset as ready and continue to the next stage
- do not use older ad hoc scripts such as `tdd_to_nnuentv2_converter.py`, `finalize_conversion.py`, or `fix_split.py`
- do not invent a manual Universal-to-FDI remapping workflow in chat; rely on the maintained exporter and its reports
- training from `nnUNet_raw/DatasetXXX_NAME` is an Experimentation task
- for TDD training, use the maintained experimentation entrypoint and spec files rather than creating ad hoc launchers such as `artifacts/results/training_runs/run_baseline_*.sh` or `agents/main/scripts/training_trial_*.sh`
- `$DENTALCLAW_HOME/artifacts/results/training_runs/` is a guarded legacy directory. Do not write launchers there; use `agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py` instead.
- the canonical maintained training entrypoint is the absolute path:
  - `$DENTALCLAW_HOME/agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py`
- do not search the filesystem for `run_training.py` and do not use:
  - `$NNUNET_HOME/nnUNet/nnunetv2/run/run_training.py`
  - `nnUNetv2_train`
  - `nnUNetv2_plan_and_preprocess`
  as substitutes for the maintained DentalClaw orchestration entrypoint
- the only permitted self-authored file in the training path is a follow-up hyperparameter-search trainer subclass placed under `$NNUNET_HOME/nnUNet/nnunetv2/training/nnUNetTrainer/`; all non-search steps must use the maintained skills directly
- after launching `run_training.py --detach`, supervise the workspace instead of improvising follow-up commands. The minimum monitoring set is `launcher_status.json`, `run_status.json`, `search_events.jsonl`, `history.json`, and the latest training log.
- if a background process session disappears, check artifact status files first; do not keep polling a dead session id
- if the same cleanup command fails with `SIGKILL` twice, stop retrying and inspect status instead
- do not patch nnUNet preprocessed directory layouts by hand during orchestration; if the maintained workflow cannot consume the exported dataset, report the maintained-workflow issue instead of inventing a local workaround
- the main agent should summarize dataset id, dataset folder, model result folder, and any validation caveats
