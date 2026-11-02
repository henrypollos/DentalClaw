# DentalClaw

A modular, natural-language-driven workflow platform for auditable dental imaging AI experiments.

DentalClaw connects research requests to registered workflow contracts, deterministic pre-dispatch checks, execution records and reviewable outputs. The browser interface uses OpenClaw; the workflow backend can also run directly from the command line.

> *DentalClaw: Contract-Based Workflow Enforcement for Auditable Dental Imaging AI Experiments* — under review, International Dental Journal.

This repository is the code accompanying the manuscript. It implements the four registered workflow contracts evaluated in the paper. It is not a segmentation model and not a clinical decision support system.

## Supported workflows

| Workflow | Delivery |
|---|---|
| Existing-result delivery | Bound saved summaries, images, masks and overlays; optional CSV of stored metrics |
| Binary 2D package quality control | Pairing, readability, dimensions, binary labels, inventory and exact cross-split image duplicate checks |
| TDD inference and reporting | Bound nnU-Net checkpoint inference, metrics and a selected-case report |
| Binary-raster training | TinyUNet2D training and test outputs from explicitly partitioned image-mask packages |

These are the four routes evaluated in the manuscript. Existing-result delivery does not run inference. Quality control does not repair data or authorize training. TDD inference requires its named assets. Training was evaluated with one-epoch workflow checks, not convergence or clinical performance.

## Installation

Use Python 3.10 or later on Linux. For planning, quality control and regression checks:

```sh
python -m pip install numpy Pillow pytest
```

Neural execution additionally requires PyTorch, SciPy and OpenCV; TDD inference also requires nnU-Net v2 and its bound checkpoint/model metadata. Data and model weights are not included.

For the browser interface, install and configure OpenClaw and a coordinator model. The backend planner has its own settings in `platform_mvp/next_version/settings.json`; provide `DEEPSEEK_API_KEY` to the process running it. The recorded study alias is retained in that file; current availability depends on the provider. Compatible explicit endpoint settings are also supported.

## Browser chat with OpenClaw

The repository includes the custom **DentalClaw Workflow** panel and an installation preparation script. The reference OpenClaw version is **2026.3.13**; other versions require compatibility checking.

```sh
python integrations/openclaw/prepare_ui.py \
  --openclaw-package /absolute/path/to/node_modules/openclaw \
  --out-dir artifacts/openclaw-ui \
  --python /absolute/path/to/python \
  --base-config /absolute/path/to/openclaw.json
```

This prepares a separate Control UI directory and a merged local configuration with a dedicated `dentalclaw` coordinator. It preserves the existing installation and does not restart the live service. The merged config may contain existing credentials and stays outside the served UI directory.

After reviewing the configuration and ensuring its port is available:

```sh
OPENCLAW_CONFIG_PATH=/absolute/path/to/artifacts/openclaw-ui/openclaw.local.json \
  openclaw gateway run

# In another terminal, with the same configuration:
OPENCLAW_CONFIG_PATH=/absolute/path/to/artifacts/openclaw-ui/openclaw.local.json \
  openclaw dashboard
```

For a remote workstation:

```sh
ssh -N -L 18789:127.0.0.1:18789 <user>@<workstation>
```

Open `http://127.0.0.1:18789/`, authenticate, select the **dentalclaw** chat and use the **DentalClaw Workflow** panel:

1. Select an input context and describe the required outputs.
2. Leave execution unchecked to generate a plan only.
3. Enable **Execute if the route is ready** for delivery or quality control; model computation additionally requires **Allow training or inference**.
4. Review the returned decision, execution status and artifact paths. A finished chat is not proof of successful delivery.

For example: “Audit this binary image-mask package for pairing, dimensions and label values; do not train.” The default context is empty. Bind actual server-side contexts with `--profiles` before requesting work on data.

See [setup, context profiles and troubleshooting](integrations/openclaw/README.md). The panel delegates through OpenClaw to `--next-version`; the backend selects the route. It does not preselect a task or default case. This agent-mediated browser path includes an outer coordinator call and is not a one-call reproduction of the paper's routing evaluation.

## Command-line use

Create a local context file:

```json
{"data_root": "/absolute/path/to/binary2d_package"}
```

Plan a request without execution; this makes one backend model call:

```sh
python platform_mvp/run_platform_mvp.py --next-version \
  --intent "Audit pairing, dimensions and binary labels; do not train." \
  --context context.json --out-dir artifacts/plan_001
```

Use a fresh output directory each time. Add `--execute` for an eligible route and `--allow-model-execution` for training/inference. The optional `--coverage-check` enables the subsequent development check of mandatory deliverables.

For direct quality control without a model call:

```sh
python platform_mvp/run_platform_mvp.py --data-services \
  --intent "Check image-mask pairing, dimension consistency and label values." \
  --data-root /absolute/path/to/binary2d_package \
  --out-dir artifacts/qc_001 --execute
```

See [planner contexts](platform_mvp/next_version/README.md), [result bindings and quality control](platform_mvp/data_services/README.md), and [runtime contracts](platform_mvp/README.md).

## Manuscript evaluation

| Stage | Exact decisions | Verified positive deliveries |
|---|---:|---:|
| Development set (n = 35) | 35/35 | 14/14 |
| Set 1, first run (n = 14) | 11/14 | 5/6 |
| Set 1, routing regression (n = 14) | 14/14 | Not rerun |
| Set 2, first run (n = 30) | 24/30 (80.0%) | 4/6 (66.7%) |

Eight requests were dispatched in Set 2: five of the six executable references and three non-executable references that were released and executed. All eight completed their registered sub-workflow, and four of the six executable references met whole-request acceptance. The dispatch count and the delivery count therefore have different denominators and should not be combined.

Incorrect release never occurred in the development set (0/21) and occurred for 2/8 and 3/24 non-executable references in the two prospective sets. The development results describe requests used during development and are not evidence of generalisation. The collections included requests written with awareness of the registered workflow scopes, known assets and author-team reference decisions. Results describe that scope, not clinical diagnostic performance.

This repository integrates the runtime and refrozen routing repairs; it is not a byte-identical evaluation snapshot. The optional coverage check and browser packaging are subsequent integration work. Historical results require the locked requests, references, asset bindings and execution records. The former weighted 30-intent benchmark is legacy material, not the revised manuscript's primary endpoint.

## Data, artifacts and code

Study assets comprise a local TDD binary export (900 training/100 test pairs), a synthetic non-patient package (2/1/1 train/validation/test pairs), and a TDD-derived development package (4/2/2). The derived package is not an independent clinical cohort; available metadata do not establish patient-level split independence for TDD.

Data, checkpoints, credentials and run outputs are excluded from source control. Runs retain plans, normalized decisions, API audit records and execution artifacts. Images and masks remain local. The configured models receive request text and the relevant metadata; the OpenClaw coordinator also sees command paths.

```text
integrations/openclaw/  Browser panel, configuration preparation and setup guide
platform_mvp/          Four-route planner, services, contracts and checks
agents/                Broader agent workspaces and executor skills
skills/                Training and inference implementations
mvp_fullflow/          Bound TDD inference/report executor
schemas/               Structured task, dataset and trace records
benchmark_trace/       Legacy benchmark and retained OpenClaw bridge
benchmark_intents/     Legacy request sets
registry/              Historical registry schemas
tests/                 Local regression checks
```

Legacy adapters are retained where needed by existing entrypoints; they do not expand the revised manuscript's evaluated scope.

## Validation and citation

```sh
python -m pytest tests -q
node --test tests/test_openclaw_workflow.cjs
```

Local checks use fixtures, mocked responses and offline replay. They do not rerun the online study or establish clinical performance.

Please cite the manuscript title above while it is under review. Source is provided for research and peer-review purposes; a final license and DOI-based archival release are planned upon publication/acceptance. OpenClaw retains its own license.
