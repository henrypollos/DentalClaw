---
name: clinical_result_report
description: Generate an English clinical report and overlay figure from a segmentation mask and original image.
---

# Clinical Result Report

Use the local Python report generator to create:
- `report.md`
- `summary.json`
- `review_list.json`
- `overlay.png`

Inputs:
- `case_id`
- `image_path`
- `mask_path`
- `out_dir`

Workflow:
1. Call the maintained CLI with exact flags:
   - required:
     - `--case_id <case_id>`
     - `--mask_path <path>`
     - `--out_dir <path>`
   - optional:
     - `--image_path <path>`
     - `--workspace_dir <path>`
   - canonical pattern:
     - `python run_report.py --case_id <case_id> --mask_path <mask.png> --out_dir <out_dir> [--image_path <image.png>] [--workspace_dir <workspace>]`
2. Read the generated `report.md` and `summary.json`.
3. Return the Markdown report as the main text.
4. Attach the image separately as media.
5. Do not write `MEDIA:` inside the Markdown body.

Important:
- Keep the report concise, clinical, and English-only.
- Save the overlay image into the OpenClaw workspace if possible.
- The final reply should contain the report text, and the image should be handled as an attachment, not plain text.
