---
name: cbct_qc
description: Audit CBCT datasets before downstream training or evaluation, including unlabeled queues, for metadata anomalies, volume consistency, label validity, image quality risks, duplicate scans, and split leakage.
---

# CBCT QC

Use this skill when you need a cautious, publication-ready quality audit for dental or maxillofacial CBCT datasets.

It is designed for:

- labeled or unlabeled CBCT queues
- nnUNet-style datasets such as `imagesTr/` + `labelsTr/`
- pretraining-stage risk review before modeling, registration, evaluation, or reporting

## Script

- `scripts/audit_cbct_dataset.py`
  - discovers and normalizes imported CBCT cases into a case manifest
  - audits metadata, volume consistency, labels, quality/artifact heuristics, duplicates, and split risk
  - separates auto-correctable issues from manual-review and exclusion recommendations
  - writes structured case-level and cohort-level reports without mutating source data

## Outputs

The script writes an audit bundle under the chosen output root:

- `manifest.json`
- `cases.jsonl`
- `normalized_cases.jsonl`
- `corrections.jsonl`
- `cohort_summary.json`
- `cohort_summary.md`

## Typical Usage

```bash
# From agents/data_curator
/home/yiyang/miniconda3/envs/nnunetv2/bin/python skills/core/cbct_qc/scripts/audit_cbct_dataset.py \
  --dataset-root /data/data2/yiyang/JoD/ToothFairy3_LPS \
  --sample-limit 100 \
  --report-key toothfairy3_lps_cbct_qc_smoke \
  --output-root reports/cbct_qc/toothfairy3_lps_smoke
```

Optional flags:

- `--split-json` to audit explicit train/val/test metadata
- `--label-policy optional|required|ignore` to control whether missing annotation volumes are logged as warnings or blocking issues
- `--sample-seed` to make subsampling reproducible
- `--max-voxel-sample` to tune image-quality metric cost

## Notes

- Do not invent scanner or patient metadata when the normalized files no longer contain it.
- If a quality problem is heuristic rather than provable, report it as `suspected`.
- Safe standardization actions are logged as recommendations; the script does not silently rewrite source data.
