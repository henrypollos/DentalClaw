---
name: path_inventory
description: Scan directory trees and summarize file layouts for quick dataset intake triage.
---

# Path Inventory

Use this skill when you need a quick view of an unknown directory tree before more semantic dataset probing.

## Scripts

- `scripts/scan_tree.py`
  - Emit a structured tree listing with path, type, depth, and size
- `scripts/summarize_files.py`
  - Summarize counts, suffix distribution, and directory fan-out

## Notes

- This skill is intentionally generic and does not infer dataset semantics.
- Use it before task-specific probing when the incoming structure is unknown.
