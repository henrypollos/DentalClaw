#!/usr/bin/env python3
"""Build case records from a dataset root and probe report."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import build_probe_report, case_stem, list_case_paths, read_json, write_json, write_jsonl


def parse_role_path(value: str):
    role, raw_path = value.split('=', 1)
    return role.strip(), Path(raw_path).resolve()


def main():
    parser = argparse.ArgumentParser(description='Build case records from a dataset root.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, default=None)
    parser.add_argument('--probe-json', type=Path, default=None)
    parser.add_argument('--primary-image-dir', type=Path, default=None)
    parser.add_argument('--label-dir', action='append', default=[], help='Override raster label dir as role=path.')
    parser.add_argument('--include-unmatched', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    probe = read_json(args.probe_json) if args.probe_json else build_probe_report(args.dataset_root)
    if args.primary_image_dir:
        primary_dir = args.primary_image_dir.resolve()
    else:
        if not probe.get('primary_image_dir'):
            raise ValueError('No primary image directory could be inferred. Provide --primary-image-dir explicitly.')
        primary_dir = Path(probe['primary_image_dir']).resolve()
    primary_images = list_case_paths(primary_dir)

    label_dirs = []
    if args.label_dir:
        for item in args.label_dir:
            role, path = parse_role_path(item)
            label_dirs.append({'role': role, 'path': str(path)})
    else:
        label_dirs = probe['raster_label_dirs']

    label_maps = {entry['role']: list_case_paths(Path(entry['path'])) for entry in label_dirs}
    case_ids = set(primary_images)
    if args.include_unmatched:
        for mapping in label_maps.values():
            case_ids.update(mapping)
    case_rows = []
    for case_id in sorted(case_ids):
        image_path = primary_images.get(case_id)
        if image_path is None:
            continue
        row = {
            'case_id': case_id,
            'images': [
                {
                    'role': 'primary_image',
                    'path': str(image_path.resolve()),
                    'suffix': image_path.suffix,
                }
            ],
            'raster_labels': [],
            'annotation_placeholders': [],
            'metadata': {},
        }
        for role, mapping in label_maps.items():
            label_path = mapping.get(case_id)
            if label_path is not None:
                row['raster_labels'].append({
                    'role': role,
                    'path': str(label_path.resolve()),
                    'suffix': label_path.suffix,
                })
        case_rows.append(row)
        if args.limit is not None and len(case_rows) >= args.limit:
            break

    write_jsonl(case_rows, args.output_jsonl)
    summary = {
        'dataset_root': str(args.dataset_root.resolve()),
        'primary_image_dir': str(primary_dir),
        'case_count': len(case_rows),
        'label_roles': sorted(label_maps.keys()),
    }
    if args.output_summary:
        write_json(summary, args.output_summary)
    print(summary)


if __name__ == '__main__':
    main()
