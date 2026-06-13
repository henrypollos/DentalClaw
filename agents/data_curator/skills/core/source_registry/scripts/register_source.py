#!/usr/bin/env python3
"""Register or update a dataset source record."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import read_json, sanitize_role, utc_now_iso, write_json


def main():
    parser = argparse.ArgumentParser(description='Register or update a dataset source record.')
    parser.add_argument('--registry-path', type=Path, required=True, help='Path to the source registry JSON file.')
    parser.add_argument('--source-type', choices=['local', 'remote'], required=True)
    parser.add_argument('--source-path', type=str, default=None, help='Original local path or remote URL.')
    parser.add_argument('--source-id', type=str, default=None, help='Stable source identifier.')
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--license', type=str, default=None)
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    registry = {'sources': []}
    if args.registry_path.exists():
        registry = read_json(args.registry_path)
    registry.setdefault('sources', [])

    derived = args.dataset_name or (Path(args.source_path).name if args.source_path else 'dataset_source')
    source_id = args.source_id or sanitize_role(derived)

    existing = None
    for record in registry['sources']:
        if record.get('source_id') == source_id:
            existing = record
            break

    payload = existing or {'source_id': source_id, 'registered_at': utc_now_iso()}
    payload.update({
        'source_id': source_id,
        'dataset_name': args.dataset_name or payload.get('dataset_name') or derived,
        'source_type': args.source_type,
        'source_path': args.source_path,
        'version': args.version,
        'license': args.license,
        'notes': args.notes,
        'updated_at': utc_now_iso(),
    })

    if existing is None:
        registry['sources'].append(payload)
    write_json(registry, args.registry_path)
    print(payload)


if __name__ == '__main__':
    main()
