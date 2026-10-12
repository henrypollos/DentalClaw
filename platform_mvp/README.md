# DentalClaw runtime

Use `run_platform_mvp.py --next-version` for the four-route online planner described in the revised manuscript. See the [main README](../README.md) for scope and evaluation results, [planner guide](next_version/README.md) for context and execution options, and [data services](data_services/README.md) for result bindings and deterministic QC.

The [OpenClaw browser integration](../integrations/openclaw/README.md) delegates to this same four-route entry, with explicit input context and execution controls.

## Entry points

| Entry | Decision path | Execution |
|---|---|---|
| `--next-version` | Online model proposal followed by local route/asset checks | `--execute`; neural routes also require `--allow-model-execution` |
| `--data-services` | Deterministic delivery/QC planning | `--execute`; never launches a model |
| Neither flag | Legacy orchestrator and wider registry | Retained for compatibility; not the revised four-route evaluation entry |

The shell wrapper accepts the same flags:

```sh
sh platform_mvp/run_platform_mvp.sh --next-version --help
```

`RUNNER_PYTHON` chooses the wrapper interpreter; `DENTALCLAW_PYTHON` chooses the delegated model interpreter. Defaults use `CONDA_HOME/envs/nnunetv2/bin/python` when configured, otherwise the active conda environment or Python on PATH. The four-route context may override the delegated interpreter with `dentalclaw_python`.

## Contracts and outputs

- `next_version/registry.json`: four registered workflows.
- `route_preflight.py`: shared checks before planning and execution.
- `validate_private2d_package.py` and `private_data_contract.json`: raster inventory, pairing, dimensions, labels and partition checks.
- `tdd_model_binding.py`: checkpoint/model-metadata identity checks.
- `training_status.py`: completion checks; launch alone is not success.
- `evaluation_adapter.py`: normalization of decisions for downstream scoring; does not read reference labels.
- `online_agent.py`: one-attempt API transport and audit records.

Four-route runs write `platform_plan.json`, `prediction.json` and `runtime_version.json`; online calls write `online_agent/`, and dispatched routes write `execution_result.json`. `--coverage-check` adds `coverage_audit.json`. Failed model calls remain errors and do not become executable fallback decisions.

## Binary raster packages

A supported explicitly partitioned package can use:

```text
package/
  imagesTr/   labelsTr/
  imagesVal/  labelsVal/
  imagesTs/   labelsTs/
```

Images and masks must pair by case ID, have matching dimensions and contain supported readable 2D rasters. Masks use labels 0 and 1. The training gate requires a validation partition and test images. Exact cross-split image duplicate checks are not proof of patient independence. A supplied `dataset.json` does not automatically enable an nnU-Net training backend: that path remains unverified for user packages.

## TDD asset binding

The TDD route expects the assets declared in `next_version/registry.json` plus the delivery/QC records and training-run summary used by `mvp_fullflow/run_mvp_fullflow.py`. Its fixed paths are repository-relative. The checkpoint must match the saved nnU-Net model snapshot and metadata. Missing assets block execution; a fresh checkout contains no model weights or study export.

`selected_case_ids` chooses the single report example. Inference covers the test partition unless `tdd_case_only: true` explicitly narrows prediction scope. Existing-result delivery is a separate workflow and does not run inference.

## Version provenance

The publication preparation ports the refrozen routing prompt and known-dataset gate into the four-route runtime. It preserves the original study snapshots outside the source release. The optional mandatory-deliverable check is subsequent development, not a remeasurement of the final 30-request test. Historical meeting notes, readiness scripts and the wider legacy registry are retained but do not expand the revised manuscript's evaluated scope.
