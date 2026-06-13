---
name: file_integrity
description: Compute checksums, verify expected digests, and fingerprint dataset snapshots for reproducible data intake.
---

# File Integrity

Use this skill when you need reproducible checksums or lightweight snapshot fingerprints for downloaded or local datasets.

## Scripts

- `scripts/compute_checksum.py`
  - Compute a digest for a file or a whole directory tree
- `scripts/verify_checksum.py`
  - Compare a file or directory digest against an expected checksum
- `scripts/snapshot_fingerprint.py`
  - Build a compact fingerprint report for a directory snapshot

## Notes

- Directory digests are based on relative file paths plus per-file digests.
- Prefer `sha256` unless you need compatibility with an external checksum source.
