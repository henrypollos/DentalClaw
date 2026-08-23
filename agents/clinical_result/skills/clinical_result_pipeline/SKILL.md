---
name: clinical_result_pipeline
description: Run nnUNet-based dental panoramic analysis and generate a review-oriented workflow report. The main assistant output should be the Final Report text, while the HTML report path is returned separately.
---

# Clinical Result Pipeline

## Purpose
This skill runs the DentalClaw clinical result pipeline for a new dental imaging case.

The pipeline should:
1. anonymize the input image if enabled,
2. run nnUNet-based inference with optional TTA and fold ensemble,
3. generate an overlay figure,
4. compute evaluation metrics when a reference mask is available,
5. generate both Markdown and HTML clinical reports,
6. return the Final Report text as the main assistant output,
7. provide the HTML report path separately for later opening or review.

## Inputs

### Required
- `case_id`

### Optional
- `image_path`
- `label_path`
- `out_dir`
- `model_paths`
- `nnunet_folds`
- `checkpoint_name`
- `use_tta`
- `anonymize`

## Path conventions

If the user provides an explicit `image_path` (absolute or relative), use it directly.
If the user provides only a `case_id`, the skill will look for:

- `cases/{case_id}/image.png` (or `image_0000.png`)
- optionally `cases/{case_id}/label.png`

When no `model_paths` is given, the skill automatically uses the default JoD nnUNet model located at:
`$NNUNET_HOME/nnUNet/nnUNet_results/Dataset106_Teeth32_Labelbox/nnUNetTrainer__nnUNetPlans__2d`
and auto-detects available folds for internal ensemble.

## Workflow

1. Read the case input.
2. Resolve image, label, and output paths.
3. If `anonymize` is enabled, create an anonymized working copy of the input image.
4. Run nnUNet inference on the anonymized or original working image.
5. Apply TTA and fold ensemble when enabled.
6. Generate:
   - overlay image
   - summary JSON
   - review notes
   - Markdown report
   - HTML report
7. If a label path exists, compute evaluation metrics such as Dice, IoU, Precision, and Recall.
8. Return the Final Report text in the assistant response.
9. Return the HTML report path separately.

## Output rules

### Main assistant output
The assistant should display the Final Report text only, or the Final Report text as the primary content.

### File outputs
The skill should also provide these artifacts when available:
- `report.md`
- `report.html`
- `summary.json`
- `review_list.json`
- `overlay.png`

### HTML path
The generated HTML path should be included in the final output metadata or file list so it can be opened later.

## Report content requirements

The final report should include, in a review-oriented style:

- case information
- workflow metadata
- dataset context and governance tags
- imaging findings
- quantitative output summary
- model performance summary if ground truth is available
- review-relevant findings / case notes
- review summary
- suggested review points
- disclaimer

## Style requirements

- Keep the report review-oriented and professional.
- Keep the report English-only.
- Preserve the project focus on nnUNet-based segmentation, TTA, ensemble inference, anonymization, and downstream review support.
- Do not output training instructions unless the user explicitly asks for training.
- Do not replace the Final Report with raw logs.
- Do not ask for unnecessary absolute paths if the case can be resolved from the workspace.
