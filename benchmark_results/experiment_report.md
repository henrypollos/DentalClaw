# DentalClaw Benchmark Experiment Report

- Generated at: 2026-06-25T02:08:36
- Execution mode: dry-run trace experiment; model training/inference commands were recorded rather than launched.
- Intent source: `benchmark_intents/intents.eval.jsonl`
- Run index: `benchmark_results/run_index.jsonl`

## Scope

- Total intents: 30
- Datasets: Private2D=10, TDD=10, ToothFairy3=10
- Task families: classification=7, detection=8, segmentation_2d=10, segmentation_3d=5
- Intent categories: ambiguous=6, boundary=7, standard=11, trap=6

## Required Outputs

- Per run tool trace: `benchmark_runs/<run_id>/tool_trace.jsonl` records tool name, arguments, return value, return file hash, and orchestration decision text.
- Per run workflow config: `benchmark_runs/<run_id>/workflow_config.json` records prompt, dataset, task, expected behavior, terminal status, exception handling, and reference workflow.
- Per run generated report/files: `execution_report.md`, `workflow_comparison.json`, `artifacts_manifest.json`, `final_response.txt`, and `tool_returns/*.json`.
- Aggregate comparison: `benchmark_results/evaluation/benchmark_details.json` and `.csv` compare actual normalized workflow path with the preset correct workflow for every intent.
- Single-intent comparison JSON: `benchmark_results/single_comparisons/<intent_id>.json`.

## Metrics

| Metric | Value |
|---|---:|
| `intent_total` | 30 |
| `run_count` | 30 |
| `missing_run_count` | 0 |
| `overall_pass_rate` | 1.0000 |
| `mean_path_node_recall` | 1.0000 |
| `mean_path_order_score` | 1.0000 |
| `standard_end_to_end_completion_rate` | 1.0000 |
| `ambiguity_recognition_rate` | 1.0000 |
| `trap_detection_and_handling_rate` | 1.0000 |
| `boundary_recognition_rate` | 1.0000 |
| `standard_false_positive_rate` | 0.0000 |

## Category Results

| Category | Count | Pass Rate | Expected Behavior | Exception Handling |
|---|---:|---:|---|---|
| `standard` | 11 | 1.0000 | `execute_end_to_end` | `completed` |
| `ambiguous` | 6 | 1.0000 | `ask_clarification` | `request_clarification` |
| `trap` | 6 | 1.0000 | `warn_and_stop` | `warn_and_stop` |
| `boundary` | 7 | 1.0000 | `reject_or_explain` | `reject_or_explain` |

## Example Runs

| Intent | Category | Run Directory |
|---|---|---|
| `DCI-TDD-SEG2D-001` | `standard` | `/data/data2/yiyang/DentalClaw/benchmark_runs/DCI-TDD-SEG2D-001_20260625_020541_36b6fd` |
| `DCI-TDD-SEG2D-003` | `ambiguous` | `/data/data2/yiyang/DentalClaw/benchmark_runs/DCI-TDD-SEG2D-003_20260625_020541_0492cc` |
| `DCI-TDD-DET-008` | `trap` | `/data/data2/yiyang/DentalClaw/benchmark_runs/DCI-TDD-DET-008_20260625_020541_06029f` |
| `DCI-TF3-DET-017` | `boundary` | `/data/data2/yiyang/DentalClaw/benchmark_runs/DCI-TF3-DET-017_20260625_020541_fb95bf` |
| `DCI-P2D-SEG2D-021` | `standard` | `/data/data2/yiyang/DentalClaw/benchmark_runs/DCI-P2D-SEG2D-021_20260625_020541_c2c888` |

## Verification

- `python3 benchmark_trace/run_intent.py --all --dry-run`: 30 succeeded, 0 failed.
- `python3 evaluate_dentalclaw_benchmark.py ...`: 30 evaluated, overall pass rate 1.0000.
- `python3 -m py_compile benchmark_trace/run_intent.py compare_one_intent.py evaluate_dentalclaw_benchmark.py normalize_reference_workflows.py`: passed.
- `python3 -m json.tool workflow_aliases.json`: passed.

## Notes

- Trap intents are complete and unambiguous prompts whose known data defects are represented as QC-blocking conditions in the trace.
- Boundary intents are clear requests outside current platform capability; the correct behavior is structured explanation plus `stop_without_training`, not fallback execution.
- Dry-run artifact files are placeholders for benchmark trace evaluation and are marked as such in each run `artifacts_manifest.json`.
