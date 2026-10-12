# Four-route planner

This entry lets the configured online model select one of four registered routes
directly. It incorporates the refrozen routing prompt and dataset-identity gate
used for the revised evaluation. This portable integration is not a byte-identical
study snapshot. The route choice is not preselected by the legacy keyword matcher.

| Route | Actual delivery | Limits |
|---|---|---|
| `existing_result_delivery` | Copy bound summary, images, masks and overlays; CSV of existing scalar metrics | No new inference, metric calculation, ranking or error analysis |
| `binary2d_package_qc` | Pairing, readability, dimensions, binary labels and exact duplicate checks | Inspection can complete with failed quality checks; this never authorizes training |
| `tdd_2d_segmentation_infer_report` | Existing TDD checkpoint inference and a standard selected-case report | Fixed named assets; no new training or automatic user-data substitution |
| `private_2d_segmentation_train` | TinyUNet2D smoke training and standard test outputs | Paired binary rasters with explicit train/val/test; no CV, automatic split or performance guarantee |

## Invocation

```sh
python3 platform_mvp/run_platform_mvp.py --next-version \
  --intent 'Deliver the saved results without running the model again.' \
  --context /path/to/context.json --out-dir /path/to/fresh/output
```

Planning is the default. Add `--execute` to dispatch a ready data service.
Neural computation additionally requires `--allow-model-execution`. An allowed
route, an attempted launch and successful completion are distinct states.

Context is a local JSON object with optional `data_card`, `data_root`,
`result_manifest`, `selected_case_ids`, `selection_required`, `expected_run_id`,
`tdd_case_only` and `dentalclaw_python`. The accepted data-card fields are defined
by `CARD_KEYS` in `runner.py`; reference labels are rejected. For example:

```json
{
  "data_card": {"dataset_id": "Dataset501_TDDTeethBinary2D"},
  "result_manifest": "/path/to/bound_existing_results.json",
  "selected_case_ids": ["100"]
}
```

Use `platform_mvp.data_services.build_binding` to bind existing artifacts with
SHA-256 digests. Bind package identity to its manifest ID where available.
Outputs must be fresh and outside inputs. A result bundle's global summary
retains its original scope even when only selected images are copied.

The model receives the request and whitelisted factual metadata, available
artifact types and sample indices. The context's local paths and raw artifacts
are not included. User-entered request text is transmitted verbatim: callers
must not place secrets or private paths in it. API credentials remain local.
The response schema accepts only registered route IDs and bounded parameters;
it cannot supply executable code or filesystem paths. Preflight rechecks inputs
before execution. Semantic route selection remains a model decision and needs
online validation; passing a schema check does not prove that it understood the
request correctly.

## Audit and validation

Each online call saves its prompt, request, redacted response, transport audit,
plan and normalized prediction. Failed calls do not become executable decisions.
There is one attempt per invocation and no automatic model or offline fallback.
`--decision-json` explicitly replays a supplied decision for local engineering
checks; it is never evidence of online routing accuracy.

## Optional mandatory-deliverable check

Add `--coverage-check` to use the subsequent development prompt and capability
check. The extended response includes boolean requirements for ground-truth
masks, prediction masks, source-package files, a QC report, smoke training and
additional analysis. `coverage_audit.json` retains the proposed route, extracted
requirements, capability violations and final decision. This does not guarantee
that the model extracted every requirement correctly.

Without this flag the six-field planner response remains compatible with the
refrozen routing schema. Offline replay must match the selected schema. The
coverage check was developed after the final first test; it must not be presented
as the implementation that produced that test's published results.

`runtime_version.json` records the port and whether coverage checking was enabled.
Use `python -m pytest tests -q` for local checks. Online evaluation additionally
requires separately locked requests, references, assets and settings; local tests
and replayed decisions are not independent evaluation results.
