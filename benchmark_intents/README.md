# DentalClaw Test Intents

This directory contains reference test intents for comparing a platform actual execution trace against the expected DentalClaw workflow path.

The main asset is:

- `intents.jsonl`: 30 JSONL records, one intent per line.

Derived files are kept in sync for older and newer evaluators:

- `dentalclaw_30_intents.jsonl`: same 30-record design set as `intents.jsonl`.
- `intents.compat.jsonl`: compatibility form where `reference_workflow` keeps the full raw path strings.
- `intents.eval.jsonl`: evaluator form where `reference_workflow` is normalized to canonical path nodes and `reference_workflow_raw` preserves the full raw path strings.

Run `python normalize_reference_workflows.py` from the project root after editing `intents.jsonl` to regenerate the derived files.

## Scope

The set covers three dataset groups:

- `TDD`: `$DENTALCLAW_HOME/data/TDD`
- `ToothFairy3`: primarily `$NNUNET_HOME/ToothFairy3_LPS`, plus the existing ToothFairy3 images-only subset under the data curator reports.
- `Private2D`: `$DENTALCLAW_HOME/data/2d`

The user request listed segmentation, detection, and classification while asking for four task classes. This benchmark uses four task families:

- `segmentation_2d`
- `segmentation_3d`
- `detection`
- `classification`

This keeps 2D panoramic/PNG segmentation separate from 3D CBCT segmentation, because the correct agent, QC, and training paths differ materially.

## Coverage

- 30 intents total.
- 10 intents per dataset group.
- 11 standard executable or QC/report intents.
- 6 ambiguous intents that must request clarification before training or inference.
- 6 trap intents that must detect known data defects and warn/stop.
- 7 boundary intents that must stop with a structured missing-label or unsupported-backend result.

## Record Schema

Each JSONL record includes these fields:

- `id` and `intent_id`: stable intent ID. Both are present for compatibility.
- `dataset`: one of `TDD`, `ToothFairy3`, `Private2D`.
- `dataset_path`: expected source dataset path.
- `task_family`: one of `segmentation_2d`, `segmentation_3d`, `detection`, `classification`.
- `task_type`: compatibility mirror of `task_family`.
- `task`: more specific task label.
- `intent_category`: `standard`, `ambiguous`, `trap`, or `boundary`.
- `intent_zh` and `prompt`: natural-language Chinese user intent. Both are present for compatibility.
- `expected_behavior`: evaluator behavior class such as `execute_end_to_end`, `reject_or_explain`, or `warn_and_stop`.
- `expected_terminal_status`: expected terminal status class.
- `reference_workflow_path`: ordered full reference workflow path strings. This is the human-readable source of truth for workflow review.
- `reference_workflow`: canonical path nodes for automated path comparison.
- `reference_workflow_raw`: present in normalized/eval records to preserve the raw path strings.
- `reference_artifacts` and `expected_outputs`: expected artifact classes or path patterns. Both are present for compatibility.
- `success_criteria`: behavioral checks that must hold for the intent.
- `forbidden_paths`: paths that should be scored as incorrect when observed.
- `assertions`: structured scoring hints derived from workflow path, terminal status, artifacts, success criteria, and forbidden paths.
- `data_fact_check`: dataset-specific guardrail for hallucination checks.
- `skill_dependency`: expected high-level skill/agent dependencies.

## Path Matching Guidance

For scoring, compare the actual trace against `reference_workflow` in order and inspect `reference_workflow_path` when diagnosing failures. Exact runtime IDs, dataset IDs, and workspace names can differ, but the agent/skill/entrypoint sequence should match.

Recommended scoring dimensions:

- Dataset routing: the actual path uses the dataset named in `dataset_path`.
- Agent routing: curation tasks go through `data_curator`; training/inference tasks go through `experimentation`; reporting goes through `clinical_result`.
- Maintained entrypoints: nnUNet training/inference uses the DentalClaw spec-driven scripts, not raw lower-level nnUNet calls from main.
- QC gate: intents that require QC must read or produce the relevant QC report before training or declaring readiness.
- Terminal status: unsupported detection/classification requests should stop with a structured missing-label or missing-backend state, not fake metrics or reroute to segmentation.
- Artifact evidence: the actual trace should create or reference the artifacts listed in `reference_artifacts`.
- Forbidden path penalty: any path listed under `forbidden_paths` is a strong negative signal for that intent.

## Important Conventions Captured

TDD segmentation:

- Default tooth segmentation means `teeth_binary`.
- 32-class/FDI-like segmentation must explicitly use `teeth_32class`.
- TDD nnUNet export must use the maintained `tdd-nnunet-export` path.
- TDD detection export uses the TDD curation path and bbox/COCO artifacts, not the nnUNet segmentation exporter.

ToothFairy3:

- CBCT QC must run before supervised 3D training.
- Images-only inference is valid, but supervised training and metrics are not valid without labels.
- No maintained 3D detection or CBCT classification training backend is present in this repository, so the correct path is a structured unsupported terminal state after the appropriate QC/probe step.

Private2D:

- Existing split folders `imagesTr`, `labelsTr`, `imagesVal`, `labelsVal`, and `imagesTs` should be respected.
- Private2D detection/classification requests should not be silently converted into segmentation.
- Missing annotations or missing image-level labels should be reported clearly.
