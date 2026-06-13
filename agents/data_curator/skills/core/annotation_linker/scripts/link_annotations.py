#!/usr/bin/env python3
"""Link parsed JSON annotations onto case records."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import build_probe_report, load_annotation_mapping, read_json, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description='Link JSON annotations to case records.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--cases-jsonl', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, default=None)
    parser.add_argument('--probe-json', type=Path, default=None)
    parser.add_argument('--include-polygons', action='store_true')
    args = parser.parse_args()

    probe = read_json(args.probe_json) if args.probe_json else build_probe_report(args.dataset_root)
    cases = read_jsonl(args.cases_jsonl)
    combined = {}
    skipped = []
    for entry in probe['json_annotations']:
        if not entry.get('supports_linking'):
            skipped.append(entry)
            continue
        if entry.get('role_hint') == 'polygon' and not args.include_polygons:
            skipped.append(entry)
            continue
        mapping = load_annotation_mapping(entry, include_polygons=args.include_polygons)
        for case_id, annotations in mapping.items():
            bucket = combined.setdefault(case_id, {'bbox_annotations': [], 'polygon_annotations': []})
            bucket['bbox_annotations'].extend(annotations.get('bbox_annotations', []))
            bucket['polygon_annotations'].extend(annotations.get('polygon_annotations', []))

    linked_rows = []
    total_boxes = 0
    total_polygons = 0
    for row in cases:
        case_id = row['case_id']
        annotations = combined.get(case_id, {'bbox_annotations': [], 'polygon_annotations': []})
        row = dict(row)
        row['bbox_annotations'] = annotations['bbox_annotations']
        row['polygon_annotations'] = annotations['polygon_annotations']
        total_boxes += len(row['bbox_annotations'])
        total_polygons += len(row['polygon_annotations'])
        linked_rows.append(row)

    write_jsonl(linked_rows, args.output_jsonl)
    summary = {
        'case_count': len(linked_rows),
        'bbox_annotation_count': total_boxes,
        'polygon_annotation_count': total_polygons,
        'skipped_json_annotations': skipped,
    }
    if args.output_summary:
        write_json(summary, args.output_summary)
    print(summary)


if __name__ == '__main__':
    main()
