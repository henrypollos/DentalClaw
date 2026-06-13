---
name: supervision-registry
description: Refresh a project-wide supervision registry for datasets, models, and task runs so the main agent can monitor progress and know what artifacts are available.
---

# Supervision Registry

Use this skill when the main agent needs a durable, queryable view of:

- which datasets are available
- which models are available
- which long-running tasks are in progress, completed, or failed

## Outputs

This skill refreshes the project registry under:

- `registry/datasets.json`
- `registry/models.json`
- `registry/task_runs.json`
- `registry/lineage.json`
- `registry/overview.md`

## What It Scans

- nnUNet raw datasets under `artifacts/datasets/nnUNet/nnUNet_raw/`
- published or in-progress model outputs under `artifacts/models/`
- workspace run status files such as `run_status.json`
- dataset export status files such as `nnUNet_delivery_*.status.json`

## Typical Usage

From any working directory, use the absolute script path:

```bash
/home/yiyang/miniconda3/envs/nnunetv2/bin/python \
  /data/data2/yiyang/DentalClaw/agents/main/skills/supervision-registry/scripts/refresh_registry.py
```

## Main-Agent Supervision Pattern

When supervising a long workflow:

1. refresh the registry
2. read `registry/task_runs.json` for running or failed work
3. read `registry/datasets.json` for available datasets
4. read `registry/models.json` for available models
5. summarize the current state in the main session

## Notes

- The registry is a filesystem-derived snapshot, not a database.
- If a tool did not write a status file, this skill still attempts to infer completed artifacts from existing outputs.
- Prefer refreshing the registry before answering supervision questions.
