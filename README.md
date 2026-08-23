# DentalClaw

A modular, natural-language-driven workflow platform for standardized and auditable dental imaging AI experiments.

DentalClaw converts researcher intent into executable, reviewable, and traceable workflows. It is best understood as a **workflow orchestration and validation layer** — not a new segmentation model and not a clinical decision-support system. This repository contains the platform code and the materials needed to reproduce the decision-level evaluation described in the accompanying manuscript:

> *DentalClaw: A Modular Workflow Platform for Standardized and Auditable Dental Imaging AI Experiments* (under review, Journal of Dentistry).

## Architecture

The platform is built around five specialized agents and a set of shared platform services:

| Component | Path | Role |
|---|---|---|
| Main agent | `agents/main/` | Coordinates request handling across the platform |
| Data Curation Agent | `agents/data_curator/` | Dataset audit, quality-control gates, canonical case packaging, deterministic splitting |
| Experimentation Agent | `agents/experimentation/` | Registered training/inference skills (nnU-Net v2, TTA, ensemble) |
| Reporting Agent | `agents/clinical_result/` | Generates structured, review-oriented reports |
| Research Agent | `agents/research/` | Task-aware literature search (Europe PMC / arXiv) |
| Platform orchestrator | `platform_mvp/` | Intent parsing, offline method registry lookup, fail-safe execution policy |

**Key design properties**

- **Decision/execution separation.** A reasoning agent decides *which* workflow should be attempted; a deterministic execution layer runs only the validated route or an approved external proposal.
- **Offline method registry.** Route selection is deterministic and does not depend on live search results. Each validated entry declares its required annotation type, evaluation unit, evaluation metrics, quality-control gates, and contraindications (`platform_mvp/method_registry.json`).
- **Fail-safe policy.** Unsupported or ambiguous requests are rejected or clarified; agent-proposed external solutions remain non-executable until explicitly confirmed with `--allow-external`.
- **Audit trail.** Plan files, QC decisions, execution logs, and generated reports form a per-run record for downstream review.

## Repository layout

```
platform_mvp/          Platform orchestrator, registry, data contracts
agents/                Five specialized agents (code only; run artifacts excluded)
benchmark_trace/       Decision-evaluation framework (scoring rubric, runner)
benchmark_intents/     Prespecified 30-intent evaluation set (JSONL)
registry/              Method registry schemas
schemas/               Platform schema definitions
skills/                Reusable workflow skills
tests/                 Unit tests
mvp_fullflow/          End-to-end MVP runner
```

Data, model weights, training artifacts, and experiment outputs are intentionally not versioned here. See [Data](#data) and [Artifacts](#artifacts) below.

## Quick start

```bash
# Parse and plan a natural-language request (decision layer, no execution)
python platform_mvp/run_platform_mvp.py \
  --intent "Run inference with the existing TDD binary segmentation model on the test set, and report Dice, IoU, HD95, and overlay visualizations."

# Same, with execution enabled (requires a configured dataset; see Data)
python platform_mvp/run_platform_mvp.py \
  --intent "<request>" --execute --allow-external
```

Environment requirements: Python 3.10 on Linux, NVIDIA GPU for training routes, nnU-Net v2 for segmentation routes. The reasoning backend (DeepSeek V4 Pro) is configured through environment variables.

Local paths in code and documentation use the placeholders `$DENTALCLAW_HOME` (repository root), `$NNUNET_HOME` (nnU-Net installation directory), and `$CONDA_HOME` (Python environment directory). Replace them with your local paths (or export the corresponding environment variables) before running workflows.

## Decision evaluation (30 prespecified intents)

The intent-to-workflow decision evaluation reported in the manuscript (weighted score 0.958, 30/30 above the 0.70 pass threshold) can be rerun as follows:

```bash
# Run the full 30-intent evaluation through the platform
python benchmark_trace/run_intent.py --intents-file benchmark_intents/intents.platform_mvp_30.jsonl

# Score the resulting traces with the six-dimension rubric
python benchmark_trace/eval_intents.py --all --csv report.csv
```

The rubric dimensions (planning 0.35, QC blocking 0.20, intent parsing 0.15, external proposal quality 0.15, ambiguity handling 0.10, boundary identification 0.05) and per-dimension scoring rules are documented in `benchmark_trace/eval_intents.py` and the Supplementary Materials of the manuscript. Expected outcomes are prespecified per request; scores are computed by a deterministic script.

## Data

The platform itself does not bundle datasets. The representative tasks in the manuscript use:

| Dataset | Use | Access |
|---|---|---|
| Tufts Dental Database (TDD) | 2D panoramic radiograph segmentation | Public; see the manuscript reference (IEEE DataPort) |
| ToothFairy3 (ToothFairy Challenge) | 3D CBCT segmentation | Public; see [github.com/AImageLab-zip/ToothFairy](https://github.com/AImageLab-zip/ToothFairy) |
| Private panoramic dataset (290 images) | Exploratory task extension and quality review | Available from the corresponding author upon reasonable request, subject to institutional ethical approval (PKUSSIRB-2025,107,012) |

Raw inputs are converted to a canonical case-level representation and pass deterministic QC gates before use; see `agents/data_curator/skills/` and `platform_mvp/private_data_contract.json`.

## Artifacts

Training runs, predictions, reports, and evaluation outputs produced by the platform are stored under per-run directories (by default outside this repository). They are not versioned here but form the audit trail described in the manuscript.

## Citing

If you use this platform or its evaluation materials, please cite the manuscript:

> DentalClaw: A Modular Workflow Platform for Standardized and Auditable Dental Imaging AI Experiments. *Journal of Dentistry* (under review).

## Code availability

The platform source code, method registry, intent set, and scoring scripts are provided in this repository. They will also be deposited in a public repository with an archived DOI upon acceptance.

## License

Source code is provided for research and review purposes; a license will be added upon publication.
