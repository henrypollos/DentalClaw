# CBCT QC: ToothFairy3_LPS

## Overview

- Dataset root: `/data/data2/yiyang/JoD/ToothFairy3_LPS`
- Run timestamp: `2026-04-09T15:33:09+00:00`
- Audited cases: `10`
- Sample limit: `10`
- Label schema source: `/data/data2/yiyang/JoD/ToothFairy3_LPS/dataset.json`

## Status Distribution

- needs_manual_review: `10`

## Top Findings

- possible_truncation: `10`
- shape_outlier_axis_0: `2`
- shape_outlier_axis_2: `2`
- fov_outlier_axis_0: `2`
- fov_outlier_axis_2: `2`
- possible_metal_artifact: `1`

## Cohort Risks

- No cohort-level duplicate or leakage findings were confirmed in this run.

## Cases Requiring Attention

- `ToothFairy3F_040` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_046` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3P_028` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_086` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_491` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3P_493` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_513` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_518` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_536` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3S_0018` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
