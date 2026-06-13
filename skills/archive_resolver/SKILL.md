---
name: archive_resolver
description: Extract archives, resolve nested archive roots, and flatten redundant directory wrappers for downstream dataset intake.
---

# Archive Resolver

Use this skill when you need to unpack a dataset archive, locate the real dataset root, or remove redundant wrapper directories after extraction.

## Scripts

- `scripts/extract_archive.py`
  - Extract `zip`, `tar`, `tar.gz`, `tar.bz2`, `tar.xz`, and `tgz`
  - Uses safe extraction checks to block path traversal
- `scripts/resolve_archive_root.py`
  - Walk down single-child directory chains to find the effective dataset root
- `scripts/flatten_nested_dirs.py`
  - Copy or move the resolved dataset contents into a clean destination directory

## Notes

- Prefer extracting into a snapshot directory first.
- Resolve the archive root before downstream dataset probing.
- Use `copy` mode by default unless you explicitly want to mutate the extracted tree.
