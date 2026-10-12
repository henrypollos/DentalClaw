# Data services

Two routes used by the four-route planner: existing-result delivery and binary
2D package QC. This standalone entry point uses conservative,
deterministic intent rules or an explicit `--task`; it does not call an LLM.
Its tests are engineering evidence, not a new independent intent evaluation.

## Existing result delivery

Bind an existing inference summary and optional PNG assets explicitly:

```sh
python -m platform_mvp.data_services.build_binding \
  --summary /path/to/inference_summary.json \
  --dataset-id DATASET --run-id RUN \
  --images /path/to/images --masks /path/to/masks --overlays /path/to/overlays \
  --case-ids sample_a sample_b --output /separate/bindings/results.json

python platform_mvp/run_platform_mvp.py --data-services \
  --intent 'Deliver existing overlays and metrics for the selected cases.' \
  --result-manifest /separate/bindings/results.json \
  --case-ids sample_a sample_b --artifact-kinds summary image mask overlay \
  --expected-dataset-id DATASET --expected-run-id RUN --table \
  --out-dir /separate/fresh_delivery --execute
```

The binding contains `schema_version: 1`, `dataset_id`, `run_id`, and `artifacts`.
Each artifact has a `kind` (`summary`, `image`, `mask`, `overlay`), a `path`, and
SHA-256. Non-summary artifacts also have a `case_id`. Paths may be absolute or
relative to the binding file. Dataset/run IDs are user-supplied bindings, not
independent provenance attestations. Each selected kind must exist for every
selected case. If no cases are selected and user selection is not requested,
all cases in the binding are used. Explicit user selection must never be
replaced by an implicit default such as case 100.

Output: `delivery_manifest.json`, original artifact copies, and `report.html`
with traceable file links and image previews. `--table` additionally renders
existing scalar `metrics_per_case` fields into CSV. Missing fields stay blank;
there is no metric computation, confidence interval, ranking, training or
inference. The original summary JSON is delivered unchanged, including its
full original scope; the optional table filters to selected cases.

The binding helper supports the existing TDD/builtin PNG conventions
(`case.png` and image `case_0000.png`). Other naming schemes can use a manually
reviewed binding; the helper does not guess additional naming conventions.

## Independent binary 2D QC

```sh
python platform_mvp/run_platform_mvp.py --data-services \
  --intent 'Check image-mask pairing, dimension consistency and label values.' \
  --data-root /path/to/package --out-dir /separate/fresh_qc --execute
```

Uses the existing package validator without requiring a model or training
budget. Checks include readable raster files, 0/1 masks, pairing, shapes,
case IDs, exact decoded-image duplicates across splits, and supplied patient
mapping. This is not clinical annotation correctness or near-duplicate image
analysis. No automatic repair, repartitioning or clinical claims.

Output: `qc_report.json`, `qc_issues.csv`, `input_manifest.json`, `report.html`.
`audit_completed: true` and `quality_passed: false` are compatible: an audit
can complete successfully and find bad data. `training_execution_permission`
is always false for this route. Missing input requires clarification; a
missing directory or no inspectable files blocks execution. Quality defects
in a supplied package are reported rather than preventing the audit itself.

## Planning, execution and evidence

Without `--execute`, only plans and normalized decisions are written. Both
routes use `service_preflight`, also reachable via the shared `preflight` API,
and the existing exact-action normalizer. Execution repeats input validation.
The output directory must be new and outside source directories. Delivery
checks copied bytes against the bound hashes and records a failure artifact
if copying fails; it never silently reports partial delivery as complete.

The entry point saves `platform_plan.json`, `prediction.json`, and (when run)
`execution_result.json`. `actual_execution_started` refers to the data service,
not model execution. The execution result explicitly records no training or
inference. Plans do not imply actual execution.

Tests:

```sh
python -m pytest tests/test_data_services.py -q
```

The four-route online entry uses these same services after model planning and
local preflight checks. The standalone deterministic entry is useful for direct
operations, but is not a substitute for the manuscript's online decision tests.
