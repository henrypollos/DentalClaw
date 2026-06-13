# Clinical Result Agent

## Overview
This agent transforms precomputed model predictions into clinician-friendly Chinese reports.

## Pipeline
1. Base Inference (read precomputed predictions)
2. Ensemble Merge (optional)
3. Geometric Postprocessing
4. Clinical Report Generation

## Skills
- base_inference
- ensemble_merge
- geometric_postprocess
- clinical_report_export

## Input
- case_id
- filename
- pred_dir / pred_dirs
- optional threshold settings

## Output
- final mask
- structured summary
- review suggestions
- Chinese natural-language clinical report

## Notes
- This agent is designed for result closure and clinical readability.
- It does not expose model internals in the report.
- `main_run.py` is not required in the OpenClaw-style workflow.
