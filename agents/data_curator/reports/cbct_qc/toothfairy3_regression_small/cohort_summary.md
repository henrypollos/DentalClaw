# CBCT QC: ToothFairy3_LPS

## Overview

- Dataset root: `/data/data2/yiyang/JoD/ToothFairy3_LPS`
- Run timestamp: `2026-04-10T09:06:37+00:00`
- Audited cases: `5`
- Sample limit: `5`
- Label schema source: `/data/data2/yiyang/JoD/ToothFairy3_LPS/dataset.json`

## Status Distribution

- needs_manual_review: `2`
- usable: `3`

## Top Findings

- dynamic_range_outlier: `2`
- shape_outlier_axis_2: `1`
- fov_outlier_axis_2: `1`

## Cohort Risks

- No cohort-level duplicate or leakage findings were confirmed in this run.

## Cases Requiring Attention

- `ToothFairy3F_040` -> `needs_manual_review`: Matrix size on axis 2 is an outlier relative to the cohort. Physical field of view on axis 2 is a cohort outlier.
- `ToothFairy3P_498` -> `needs_manual_review`: Sampled intensity dynamic range is a cohort outlier and may reflect low contrast or an unusually wide reconstruction range.
