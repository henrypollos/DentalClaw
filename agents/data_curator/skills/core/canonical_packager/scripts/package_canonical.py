#!/usr/bin/env python3
"""Package linked cases into a canonical internal dataset format."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import ensure_dir, read_jsonl, symlink_or_copy, utc_now_iso, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description='Package linked cases into a canonical dataset root.')
    parser.add_argument('--linked-cases-jsonl', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dataset-name', type=str, default='canonical_dataset')
    parser.add_argument('--asset-mode', choices=['symlink', 'copy'], default='symlink')
    args = parser.parse_args()

    cases = read_jsonl(args.linked_cases_jsonl)
    output_root = ensure_dir(args.output_root)
    images_root = ensure_dir(output_root / 'assets' / 'images')
    labels_root = ensure_dir(output_root / 'assets' / 'raster_labels')

    canonical_rows = []
    label_roles = Counter()
    for row in cases:
        canonical = {
            'case_id': row['case_id'],
            'images': [],
            'raster_labels': [],
            'bbox_annotations': row.get('bbox_annotations', []),
            'polygon_annotations': row.get('polygon_annotations', []),
            'metadata': row.get('metadata', {}),
        }
        for image in row.get('images', []):
            src = Path(image['path'])
            dst = images_root / image['role'] / src.name
            symlink_or_copy(src, dst, mode=args.asset_mode)
            canonical['images'].append({
                'role': image['role'],
                'path': dst.relative_to(output_root).as_posix(),
                'source_path': str(src),
            })
        for label in row.get('raster_labels', []):
            src = Path(label['path'])
            dst = labels_root / label['role'] / src.name
            symlink_or_copy(src, dst, mode=args.asset_mode)
            canonical['raster_labels'].append({
                'role': label['role'],
                'path': dst.relative_to(output_root).as_posix(),
                'source_path': str(src),
            })
            label_roles[label['role']] += 1
        canonical_rows.append(canonical)

    write_jsonl(canonical_rows, output_root / 'cases.jsonl')
    manifest = {
        'dataset_name': args.dataset_name,
        'created_at': utc_now_iso(),
        'case_count': len(canonical_rows),
        'label_roles': dict(label_roles),
        'asset_mode': args.asset_mode,
        'cases_file': 'cases.jsonl',
    }
    write_json(manifest, output_root / 'manifest.json')
    print(manifest)


if __name__ == '__main__':
    main()
