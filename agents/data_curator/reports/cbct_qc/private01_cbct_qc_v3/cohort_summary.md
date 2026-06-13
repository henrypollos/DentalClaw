# CBCT QC: private01

## Overview

- Dataset root: `/data/data2/yiyang/DentalClaw/data/private01`
- Run timestamp: `2026-04-10T14:17:17+00:00`
- Audited cases: `139`
- Sample limit: `None`
- Label schema source: `None`

## Status Distribution

- needs_manual_review: `139`

## Top Findings

- repeated_patient_identifier: `139`
- possible_metal_artifact: `44`
- dynamic_range_outlier: `7`
- spacing_outlier_axis_0: `5`
- spacing_outlier_axis_1: `5`
- spacing_outlier_axis_2: `5`
- shape_outlier_axis_0: `5`
- shape_outlier_axis_1: `5`
- shape_outlier_axis_2: `5`
- fov_outlier_axis_0: `5`

## Cohort Risks

- repeated_patient_id: `0000000010, 0000000032, 0000000033, 0000000038, 0000000039, 0000000043`

## Cases Requiring Attention

- `0000000010` -> `needs_manual_review`: Voxel spacing on axis 0 is an outlier relative to the cohort. Voxel spacing on axis 1 is an outlier relative to the cohort.
- `0000000032` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000033` -> `needs_manual_review`: Voxel spacing on axis 0 is an outlier relative to the cohort. Voxel spacing on axis 1 is an outlier relative to the cohort.
- `0000000038` -> `needs_manual_review`: Voxel spacing on axis 0 is an outlier relative to the cohort. Voxel spacing on axis 1 is an outlier relative to the cohort.
- `0000000039` -> `needs_manual_review`: Voxel spacing on axis 0 is an outlier relative to the cohort. Voxel spacing on axis 1 is an outlier relative to the cohort.
- `0000000043` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000054` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000058` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000061` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000216` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000336` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000441` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000471` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000495` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000624` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000632` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000637` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000713` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000723` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `0000000753` -> `needs_manual_review`: High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening. The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
