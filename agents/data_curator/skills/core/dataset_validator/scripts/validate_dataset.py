#!/usr/bin/env python3
"""Validate datasets for completeness, correspondence, plausibility, and split integrity."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
TDD_LIB_DIR = CURRENT_FILE.parents[3] / 'datasets' / 'tdd-curation' / 'scripts'
if str(TDD_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(TDD_LIB_DIR))

from curation_core import read_json, read_jsonl
from dataset_qc import run_dataset_qc, write_report_bundle
from tdd_common import NUMERIC_TOOTH_LABELS, build_bbox_index, dataset_paths, list_case_paths, load_bbox_items, load_polygon_items


PRIMARY_TOOTH_LABELS = set("ABCDEFGHIJKLMNOPQRST")


def build_canonical_cases(canonical_root: Path):
    rows = read_jsonl(canonical_root / 'cases.jsonl')
    cases = []
    for row in rows:
        cases.append({
            'case_id': row['case_id'],
            'images': [
                {
                    'role': entry.get('role', 'primary_image'),
                    'path': canonical_root / entry['path'],
                }
                for entry in row.get('images', [])
            ],
            'raster_labels': [
                {
                    'role': entry.get('role', 'label'),
                    'path': canonical_root / entry['path'],
                }
                for entry in row.get('raster_labels', [])
            ],
            'bbox_annotations': row.get('bbox_annotations', []),
            'polygon_annotations': row.get('polygon_annotations', []),
            'metadata': row.get('metadata', {}),
        })
    return cases


def build_tdd_cases(dataset_root: Path, *, deep_polygon_scan: bool = False):
    paths = dataset_paths(dataset_root)
    radiographs = list_case_paths(paths['radiographs'])
    teeth_masks = list_case_paths(paths['teeth_mask'])
    maxillo_masks = list_case_paths(paths['maxillomandibular'])
    bbox_payload = load_bbox_items(paths['bbox_json'])
    bbox_index = build_bbox_index(bbox_payload)
    polygon_index = {}
    if deep_polygon_scan:
        polygon_index = build_bbox_index(load_polygon_items(paths['polygon_json']))

    case_ids = sorted(set(radiographs) | set(teeth_masks) | set(maxillo_masks) | set(bbox_index) | set(polygon_index))
    cases = []
    for case_id in case_ids:
        bbox_item = bbox_index.get(case_id, {})
        polygon_item = polygon_index.get(case_id, {}) if deep_polygon_scan else {}
        bbox_annotations = []
        for obj in bbox_item.get('Label', {}).get('objects', []):
            bbox_annotations.append({
                'label': str(obj.get('title', '')).strip(),
                'bbox_xyxy': obj.get('bounding box') or [],
            })
        polygon_annotations = []
        for obj in polygon_item.get('Label', {}).get('objects', []):
            polygon_annotations.append({
                'label': str(obj.get('title', '')).strip(),
                'polygons': obj.get('polygons') or [],
            })
        title_values = [ann['label'] for ann in bbox_annotations] + [ann['label'] for ann in polygon_annotations]
        numeric_labels = sorted({value for value in title_values if value.isdigit()}, key=int)
        primary_labels = sorted({value for value in title_values if value in PRIMARY_TOOTH_LABELS})
        invalid_labels = sorted({
            value for value in title_values
            if (not value.isdigit() and value not in PRIMARY_TOOTH_LABELS)
            or (value.isdigit() and value not in NUMERIC_TOOTH_LABELS)
        })
        cases.append({
            'case_id': case_id,
            'images': [{'role': 'panoramic_image', 'path': radiographs[case_id]}] if case_id in radiographs else [],
            'raster_labels': [
                {'role': 'teeth_mask', 'path': teeth_masks[case_id]} if case_id in teeth_masks else None,
                {'role': 'maxillomandibular_mask', 'path': maxillo_masks[case_id]} if case_id in maxillo_masks else None,
            ],
            'bbox_annotations': bbox_annotations,
            'polygon_annotations': polygon_annotations,
            'metadata': {
                'bbox_json_present': case_id in bbox_index,
                'polygon_json_present': case_id in polygon_index if deep_polygon_scan else None,
                'bbox_count': len(bbox_annotations),
                'polygon_count': len(polygon_annotations),
                'polygon_scan': 'deep' if deep_polygon_scan else 'skipped',
                'tdd_readiness': {
                    'binary_segmentation_ready': bool(case_id in radiographs and case_id in teeth_masks),
                    'maxillomandibular_segmentation_ready': bool(case_id in radiographs and case_id in maxillo_masks),
                    'teeth_32class_ready': bool(case_id in radiographs and case_id in polygon_index) if deep_polygon_scan else None,
                    'detection_ready': bool(case_id in radiographs and case_id in bbox_index),
                },
                'label_profile': {
                    'numeric_labels_present': numeric_labels,
                    'primary_labels_present': primary_labels,
                    'invalid_labels_present': invalid_labels,
                },
            },
        })
    for case in cases:
        case['raster_labels'] = [entry for entry in case['raster_labels'] if entry is not None]
    return cases


def main():
    parser = argparse.ArgumentParser(description='Run dataset QC and write JSON/Markdown reports.')
    parser.add_argument('--canonical-root', type=Path, default=None)
    parser.add_argument('--tdd-root', type=Path, default=None)
    parser.add_argument('--split-json', type=Path, default=None)
    parser.add_argument('--report-key', type=str, default=None)
    parser.add_argument('--report-root', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None, help='Optional case limit for smoke validation.')
    parser.add_argument('--deep-polygon-scan', action='store_true', help='For TDD only: parse the full polygon json for per-case polygon QC.')
    parser.add_argument('--output-json', type=Path, default=None, help='Legacy alias. Optional; if omitted a default QC path is used.')
    parser.add_argument('--output-md', type=Path, default=None, help='Optional explicit markdown report path.')
    args = parser.parse_args()

    if bool(args.canonical_root) == bool(args.tdd_root):
        raise ValueError('Provide exactly one of --canonical-root or --tdd-root.')

    split_payload = read_json(args.split_json) if args.split_json else None

    if args.canonical_root:
        dataset_root = args.canonical_root.resolve()
        dataset_name = dataset_root.name
        dataset_mode = 'canonical'
        report_key = args.report_key or dataset_name
        cases = build_canonical_cases(dataset_root)
        metadata = {
            'input_mode': dataset_mode,
            'split_json': str(args.split_json.resolve()) if args.split_json else None,
            'sample_limit': args.limit,
        }
    else:
        dataset_root = args.tdd_root.resolve()
        dataset_name = f"{dataset_root.name}_source"
        dataset_mode = 'tdd_source'
        report_key = args.report_key or dataset_name
        cases = build_tdd_cases(dataset_root, deep_polygon_scan=args.deep_polygon_scan)
        metadata = {
            'input_mode': dataset_mode,
            'split_json': str(args.split_json.resolve()) if args.split_json else None,
            'tdd_dataset_root': str(dataset_root),
            'sample_limit': args.limit,
            'deep_polygon_scan': args.deep_polygon_scan,
        }

    if args.limit is not None:
        cases = cases[: args.limit]

    report = run_dataset_qc(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        dataset_mode=dataset_mode,
        cases=cases,
        split_payload=split_payload,
        metadata=metadata,
    )

    if args.output_json:
        json_path = args.output_json.resolve()
        md_path = args.output_md.resolve() if args.output_md else json_path.with_suffix('.md')
        bundle = write_report_bundle(report, report_key=report_key, report_root=json_path.parent)
        generated_json = Path(bundle['report_json'])
        generated_md = Path(bundle['report_md'])
        if generated_json != json_path:
            json_path.write_text(generated_json.read_text(encoding='utf-8'), encoding='utf-8')
        if generated_md != md_path:
            md_path.write_text(generated_md.read_text(encoding='utf-8'), encoding='utf-8')
        report_paths = {'report_json': str(json_path), 'report_md': str(md_path)}
    else:
        report_paths = write_report_bundle(report, report_key=report_key, report_root=args.report_root)

    print({
        'dataset_root': str(dataset_root),
        'dataset_mode': dataset_mode,
        'dataset_status': report['summary']['dataset_status'],
        'case_count': report['summary']['case_count'],
        'error_count': report['summary']['error_count'],
        'warning_count': report['summary']['warning_count'],
        **report_paths,
    })


if __name__ == '__main__':
    main()
