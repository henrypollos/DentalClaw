#!/usr/bin/env python3
"""Create a local dataset intake snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import build_probe_report, ensure_dir, symlink_or_copy, utc_now_iso, write_json


def main():
    parser = argparse.ArgumentParser(description='Create a local dataset intake snapshot.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--source-id', type=str, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['symlink', 'copy', 'index_only'], default='symlink')
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    snapshot_dir = ensure_dir(args.output_root / args.source_id / utc_now_iso().replace(':', '-'))
    snapshot_path = None
    if args.mode in {'symlink', 'copy'}:
        snapshot_path = snapshot_dir / 'source'
        symlink_or_copy(dataset_root, snapshot_path, mode='symlink' if args.mode == 'symlink' else 'copy')
    probe = build_probe_report(dataset_root)
    manifest = {
        'source_id': args.source_id,
        'dataset_root_original': str(dataset_root),
        'snapshot_dir': str(snapshot_dir.resolve()),
        'snapshot_path': str(snapshot_path.resolve()) if snapshot_path else None,
        'mode': args.mode,
        'captured_at': utc_now_iso(),
        'summary': {
            'primary_case_count': probe['primary_case_count'],
            'task_candidates': probe['task_candidates'],
            'primary_image_dir': probe['primary_image_dir'],
        },
    }
    write_json(manifest, snapshot_dir / 'intake_manifest.json')
    print(manifest)


if __name__ == '__main__':
    main()
