# CBCT QC: dataset

## Overview

- Dataset root: `/data/data2/yiyang/DentalClaw/agents/data_curator/reports/cbct_qc/toothfairy3F_images_only_subset/dataset`
- Run timestamp: `2026-04-10T15:27:22+00:00`
- Audited cases: `63`
- Sample limit: `None`
- Label schema source: `None`

## Status Distribution

- needs_manual_review: `20`
- usable: `43`

## Top Findings

- possible_metal_artifact: `13`
- shape_outlier_axis_0: `10`
- shape_outlier_axis_1: `10`
- fov_outlier_axis_0: `10`
- fov_outlier_axis_1: `10`
- shape_outlier_axis_2: `4`
- fov_outlier_axis_2: `4`

## Cohort Risks

- No cohort-level duplicate or leakage findings were confirmed in this run.

## Cases Requiring Attention

- `ToothFairy3F_001` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_003` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_004` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_011` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_012` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_014` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_020` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_024` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_029` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_037` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_038` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_039` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_043` -> `needs_manual_review`: Matrix size on axis 2 is an outlier relative to the cohort. Physical field of view on axis 2 is a cohort outlier.
- `ToothFairy3F_047` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_057` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_058` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_060` -> `needs_manual_review`: Matrix size on axis 0 is an outlier relative to the cohort. Matrix size on axis 1 is an outlier relative to the cohort.
- `ToothFairy3F_061` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_066` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_067` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
