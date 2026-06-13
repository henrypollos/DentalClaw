#!/usr/bin/env python3
"""Export canonical datasets into task-specific layouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import sys
CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / '_lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import (
    build_label_to_category_id,
    read_json,
    read_jsonl,
    safe_image_size,
    sort_labels,
    symlink_or_copy,
    threshold_raster_to_png,
    write_json,
)


def split_bucket_name(split_name: str, kind: str) -> str:
    if split_name == 'train':
        return 'imagesTr' if kind == 'image' else 'labelsTr'
    if split_name == 'val':
        return 'imagesVal' if kind == 'image' else 'labelsVal'
    return 'imagesTs' if kind == 'image' else 'labelsTs'


def export_segmentation(cases_by_id, splits, canonical_root: Path, output_root: Path, image_mode: str, threshold: int):
    roles = sorted({label['role'] for case in cases_by_id.values() for label in case.get('raster_labels', [])})
    exports = {}
    for role in roles:
        task_root = output_root / f'segmentation_{role}'
        case_count = 0
        for split_name, case_ids in splits.items():
            image_dir = task_root / split_bucket_name(split_name, 'image')
            label_dir = task_root / split_bucket_name(split_name, 'label')
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for case_id in case_ids:
                case = cases_by_id[case_id]
                image = case['images'][0]
                image_src = canonical_root / image['path']
                image_dst = image_dir / Path(image['path']).name
                symlink_or_copy(image_src, image_dst, mode=image_mode)
                matching_labels = [item for item in case.get('raster_labels', []) if item['role'] == role]
                if matching_labels:
                    label_src = canonical_root / matching_labels[0]['path']
                    threshold_raster_to_png(label_src, label_dir / f'{case_id}.png', threshold=threshold)
                    case_count += 1
        exports[f'segmentation_{role}'] = {
            'task_root': str(task_root),
            'case_count': case_count,
            'threshold': threshold,
        }
    return exports


def export_detection(cases_by_id, splits, canonical_root: Path, output_root: Path, image_mode: str):
    task_root = output_root / 'detection_coco'
    annotations_root = task_root / 'annotations'
    annotations_root.mkdir(parents=True, exist_ok=True)
    all_labels = {ann['label'] for case in cases_by_id.values() for ann in case.get('bbox_annotations', [])}
    label_to_category_id = build_label_to_category_id(all_labels)
    categories = [
        {'id': label_to_category_id[label], 'name': f'tooth_{label}', 'label': label}
        for label in sort_labels(all_labels)
    ]
    result = {'task_root': str(task_root), 'category_count': len(categories), 'splits': {}}
    for split_name, case_ids in splits.items():
        images_dir = task_root / 'images' / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        ann_id = 1
        for image_id, case_id in enumerate(case_ids, start=1):
            case = cases_by_id[case_id]
            image = case['images'][0]
            image_src = canonical_root / image['path']
            image_dst = images_dir / Path(image['path']).name
            symlink_or_copy(image_src, image_dst, mode=image_mode)
            size = safe_image_size(image_src)
            width, height = size if size else (None, None)
            images.append({
                'id': image_id,
                'file_name': f'{split_name}/{image_dst.name}',
                'width': width,
                'height': height,
                'case_id': case_id,
            })
            for ann in case.get('bbox_annotations', []):
                x1, y1, x2, y2 = ann['bbox_xyxy']
                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': label_to_category_id[str(ann['label'])],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'tooth_label': ann['label'],
                })
                ann_id += 1
        payload = {'images': images, 'annotations': annotations, 'categories': categories}
        ann_path = annotations_root / f'instances_{split_name}.json'
        write_json(payload, ann_path)
        result['splits'][split_name] = {
            'image_count': len(images),
            'annotation_count': len(annotations),
            'annotation_path': str(ann_path),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description='Export canonical datasets into task-specific layouts.')
    parser.add_argument('--canonical-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--splits-json', type=Path, default=None)
    parser.add_argument('--tasks', nargs='*', choices=['segmentation', 'detection'], default=['segmentation', 'detection'])
    parser.add_argument('--image-mode', choices=['symlink', 'copy'], default='symlink')
    parser.add_argument('--threshold', type=int, default=127)
    args = parser.parse_args()

    canonical_root = args.canonical_root.resolve()
    cases = read_jsonl(canonical_root / 'cases.jsonl')
    splits_payload = read_json(args.splits_json) if args.splits_json else {'splits': {'train': [row['case_id'] for row in cases], 'val': [], 'test': []}}
    splits = splits_payload['splits']
    cases_by_id = {row['case_id']: row for row in cases}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    result = {
        'canonical_root': str(canonical_root),
        'output_root': str(output_root),
        'tasks': {},
    }
    if 'segmentation' in args.tasks:
        result['tasks']['segmentation'] = export_segmentation(cases_by_id, splits, canonical_root, output_root, args.image_mode, args.threshold)
    if 'detection' in args.tasks:
        result['tasks']['detection'] = export_detection(cases_by_id, splits, canonical_root, output_root, args.image_mode)
    write_json(result, output_root / 'export_manifest.json')
    print(result)


if __name__ == '__main__':
    main()
