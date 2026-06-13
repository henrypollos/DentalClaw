# CBCT QC: ToothFairy3_LPS

## Overview

- Dataset root: `/data/data2/yiyang/JoD/ToothFairy3_LPS`
- Run timestamp: `2026-04-09T15:34:11+00:00`
- Audited cases: `100`
- Sample limit: `100`
- Label schema source: `/data/data2/yiyang/JoD/ToothFairy3_LPS/dataset.json`

## Status Distribution

- needs_manual_review: `99`
- reject: `1`

## Top Findings

- possible_truncation: `99`
- possible_metal_artifact: `22`
- shape_outlier_axis_2: `17`
- fov_outlier_axis_2: `17`
- shape_outlier_axis_0: `12`
- fov_outlier_axis_0: `12`
- dynamic_range_outlier: `4`
- noise_or_contrast_outlier: `1`
- invalid_label_values: `1`

## Cohort Risks

- No cohort-level duplicate or leakage findings were confirmed in this run.

## Cases Requiring Attention

- `ToothFairy3F_002` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_008` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_015` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_028` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_033` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_037` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3F_038` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.
- `ToothFairy3F_041` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_048` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_059` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_062` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3F_064` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 2 is an outlier relative to the cohort.
- `ToothFairy3P_003` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Sampled intensity dynamic range is a cohort outlier and may reflect low contrast or an unusually wide reconstruction range.
- `ToothFairy3P_004` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_012` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_018` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_050` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3P_056` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view. Matrix size on axis 0 is an outlier relative to the cohort.
- `ToothFairy3P_058` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
- `ToothFairy3P_064` -> `needs_manual_review`: Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.
