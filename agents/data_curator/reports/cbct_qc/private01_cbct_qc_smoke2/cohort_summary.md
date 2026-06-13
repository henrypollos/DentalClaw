# CBCT QC: private01

## Overview

- Dataset root: `/data/data2/yiyang/DentalClaw/data/private01`
- Run timestamp: `2026-04-10T13:56:33+00:00`
- Audited cases: `2`
- Sample limit: `2`
- Label schema source: `None`

## Status Distribution

- needs_manual_review: `2`

## Top Findings

- repeated_patient_identifier: `2`

## Cohort Risks

- repeated_patient_id: `0000001139, 3001281243`

## Cases Requiring Attention

- `0000001139` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
- `3001281243` -> `needs_manual_review`: The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.
