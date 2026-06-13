#!/usr/bin/env python3
"""Shared helpers for data curator core skills."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image


RASTER_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VOLUME_EXTS = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}
DICOM_EXTS = {".dcm"}
IMAGE_EXTS = RASTER_IMAGE_EXTS | VOLUME_EXTS | DICOM_EXTS

PRIMARY_POSITIVE_KEYWORDS = ["radiograph", "radiographs", "image", "images", "scan", "scans", "img"]
PRIMARY_NEGATIVE_KEYWORDS = ["mask", "masks", "label", "labels", "seg", "segmentation", "annotation", "annotations", "gt"]
MASK_ROLE_KEYWORDS = {
    "teeth_mask": ["teeth_mask", "tooth_mask", "teeth", "tooth"],
    "maxillomandibular": ["maxillomandibular", "maxillo", "mandible", "jaw", "jaws"],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalized_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith('.nii.gz'):
        return '.nii.gz'
    if name.endswith('.tar.gz'):
        return '.tar.gz'
    return path.suffix.lower()


def case_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith('.nii.gz'):
        return name[:-7]
    return path.stem


def is_image_file(path: Path) -> bool:
    return normalized_suffix(path) in IMAGE_EXTS


def is_raster_image(path: Path) -> bool:
    return normalized_suffix(path) in RASTER_IMAGE_EXTS


def sanitize_role(name: str) -> str:
    value = re.sub(r'[^a-zA-Z0-9]+', '_', name.strip().lower())
    value = re.sub(r'_+', '_', value).strip('_')
    return value or 'asset'


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(payload, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: Path):
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def safe_image_size(path: Path):
    if not is_raster_image(path):
        return None
    with Image.open(path) as image:
        return image.size


def list_case_paths(folder: Path) -> Dict[str, Path]:
    mapping = {}
    if not folder.is_dir():
        return mapping
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if path.is_file() and is_image_file(path):
            mapping[case_stem(path)] = path
    return mapping


def symlink_or_copy(src: Path, dst: Path, mode: str = 'symlink'):
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == 'symlink':
        dst.symlink_to(src.resolve())
    elif mode == 'copy':
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f'Unsupported mode: {mode}')


def threshold_raster_to_png(src: Path, dst: Path, threshold: int = 127):
    ensure_dir(dst.parent)
    with Image.open(src) as image:
        gray = image.convert('L')
        binary = gray.point(lambda x: 255 if x > threshold else 0, mode='L')
        binary.save(dst)
    return dst


def scan_directory_summaries(dataset_root: Path) -> List[dict]:
    summaries = []
    for root, dirs, files in os.walk(dataset_root):
        path = Path(root)
        file_paths = [path / f for f in files]
        image_files = [p for p in file_paths if is_image_file(p)]
        json_files = [p for p in file_paths if p.suffix.lower() == '.json']
        summaries.append({
            'path': str(path.resolve()),
            'relative_path': path.resolve().relative_to(dataset_root.resolve()).as_posix() if path.resolve() != dataset_root.resolve() else '.',
            'depth': len(path.resolve().relative_to(dataset_root.resolve()).parts) if path.resolve() != dataset_root.resolve() else 0,
            'file_count': len(file_paths),
            'image_file_count': len(image_files),
            'json_file_count': len(json_files),
            'extensions': dict(Counter(normalized_suffix(p) for p in file_paths)),
        })
    return summaries


def _score_primary_dir(summary: dict) -> tuple:
    rel = summary['relative_path'].lower()
    name = Path(summary['path']).name.lower()
    score = summary['image_file_count'] * 10
    if any(keyword in rel for keyword in PRIMARY_POSITIVE_KEYWORDS):
        score += 500
    if any(keyword in rel for keyword in PRIMARY_NEGATIVE_KEYWORDS):
        score -= 600
    score -= summary['depth'] * 5
    return (score, summary['image_file_count'], -summary['depth'], name)


def guess_primary_image_dir(dataset_root: Path, dir_summaries: List[dict]) -> Optional[Path]:
    candidates = [summary for summary in dir_summaries if summary['image_file_count'] > 0]
    if not candidates:
        return None
    best = max(candidates, key=_score_primary_dir)
    return Path(best['path'])


def guess_mask_role(path: Path) -> str:
    rel = path.as_posix().lower()
    for role, keywords in MASK_ROLE_KEYWORDS.items():
        if any(keyword in rel for keyword in keywords):
            return role
    return sanitize_role(path.name)


def guess_raster_label_dirs(dataset_root: Path, primary_dir: Optional[Path], dir_summaries: List[dict]) -> List[dict]:
    if primary_dir is None:
        return []
    primary_map = list_case_paths(primary_dir)
    primary_stems = set(primary_map)
    results = []
    for summary in dir_summaries:
        candidate = Path(summary['path'])
        if candidate == primary_dir or summary['image_file_count'] == 0:
            continue
        candidate_map = list_case_paths(candidate)
        if not candidate_map:
            continue
        overlap = primary_stems & set(candidate_map)
        overlap_ratio = (len(overlap) / len(primary_stems)) if primary_stems else 0.0
        if overlap_ratio < 0.05 and len(overlap) < 5:
            continue
        results.append({
            'role': guess_mask_role(candidate.relative_to(dataset_root)),
            'path': str(candidate.resolve()),
            'file_count': len(candidate_map),
            'matched_case_count': len(overlap),
            'overlap_ratio': overlap_ratio,
        })
    results.sort(key=lambda item: (-item['matched_case_count'], item['role']))
    return results


def sniff_json_annotation(path: Path) -> dict:
    prefix = path.read_text(encoding='utf-8', errors='ignore')[:4096]
    name = path.name.lower()
    role_hint = 'annotation_json'
    if 'bbox' in name:
        role_hint = 'bbox'
    elif 'polygon' in name:
        role_hint = 'polygon'
    elif 'keypoint' in name or 'landmark' in name:
        role_hint = 'landmark'
    detected_format = 'unknown'
    supports_linking = False
    if 'External ID' in prefix and 'Label' in prefix:
        detected_format = 'tdd_like'
        supports_linking = True
    elif '"images"' in prefix and '"annotations"' in prefix:
        detected_format = 'coco'
        supports_linking = True
    return {
        'path': str(path.resolve()),
        'role_hint': role_hint,
        'detected_format': detected_format,
        'supports_linking': supports_linking,
        'size_bytes': path.stat().st_size,
    }


def infer_task_candidates(raster_label_dirs: List[dict], json_annotations: List[dict], dir_summaries: List[dict]) -> List[str]:
    tasks = []
    if raster_label_dirs:
        tasks.append('segmentation')
    if any(entry['role_hint'] == 'bbox' or entry['detected_format'] == 'coco' for entry in json_annotations):
        tasks.append('detection')
    image_dirs = [summary for summary in dir_summaries if summary['image_file_count'] > 0]
    if len(image_dirs) >= 2:
        rel_names = ' '.join(summary['relative_path'].lower() for summary in image_dirs)
        if 'fixed' in rel_names or 'moving' in rel_names or 'register' in rel_names:
            tasks.append('registration')
    return sorted(set(tasks))


def build_probe_report(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    dir_summaries = scan_directory_summaries(dataset_root)
    primary_dir = guess_primary_image_dir(dataset_root, dir_summaries)
    raster_label_dirs = guess_raster_label_dirs(dataset_root, primary_dir, dir_summaries)
    json_annotations = []
    for path in sorted(dataset_root.rglob('*.json')):
        json_annotations.append(sniff_json_annotation(path))
    primary_case_count = len(list_case_paths(primary_dir)) if primary_dir else 0
    file_extension_counts = Counter()
    for summary in dir_summaries:
        for ext, count in summary['extensions'].items():
            file_extension_counts[ext] += count
    return {
        'dataset_root': str(dataset_root),
        'directory_summaries': dir_summaries,
        'primary_image_dir': str(primary_dir.resolve()) if primary_dir else None,
        'primary_case_count': primary_case_count,
        'raster_label_dirs': raster_label_dirs,
        'json_annotations': json_annotations,
        'file_extension_counts': dict(file_extension_counts),
        'task_candidates': infer_task_candidates(raster_label_dirs, json_annotations, dir_summaries),
        'generated_at': utc_now_iso(),
    }


def parse_tdd_like_annotations(path: Path, include_polygons: bool = False) -> Dict[str, dict]:
    data = read_json(path)
    mapping = defaultdict(lambda: {'bbox_annotations': [], 'polygon_annotations': []})
    for item in data:
        external_id = item.get('External ID')
        if not external_id:
            continue
        case_id = case_stem(Path(external_id))
        objects = item.get('Label', {}).get('objects', [])
        for obj in objects:
            label = str(obj.get('title', '')).strip()
            bbox = obj.get('bounding box') or []
            if len(bbox) == 4:
                mapping[case_id]['bbox_annotations'].append({
                    'label': label,
                    'bbox_xyxy': bbox,
                    'source_json': str(path.resolve()),
                })
            if include_polygons:
                polygons = obj.get('polygons') or []
                if polygons:
                    mapping[case_id]['polygon_annotations'].append({
                        'label': label,
                        'polygons': polygons,
                        'source_json': str(path.resolve()),
                    })
    return dict(mapping)


def parse_coco_annotations(path: Path) -> Dict[str, dict]:
    payload = read_json(path)
    images = {image['id']: image for image in payload.get('images', [])}
    categories = {category['id']: category.get('name', str(category['id'])) for category in payload.get('categories', [])}
    mapping = defaultdict(lambda: {'bbox_annotations': [], 'polygon_annotations': []})
    for ann in payload.get('annotations', []):
        image = images.get(ann.get('image_id'))
        if not image:
            continue
        case_id = case_stem(Path(image.get('file_name', 'unknown')))
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        mapping[case_id]['bbox_annotations'].append({
            'label': categories.get(ann.get('category_id'), str(ann.get('category_id'))),
            'bbox_xyxy': [x, y, x + w, y + h],
            'source_json': str(path.resolve()),
        })
    return dict(mapping)


def load_annotation_mapping(annotation_entry: dict, include_polygons: bool = False) -> Dict[str, dict]:
    path = Path(annotation_entry['path'])
    detected_format = annotation_entry.get('detected_format')
    role_hint = annotation_entry.get('role_hint')
    if detected_format == 'tdd_like':
        if role_hint == 'polygon' and not include_polygons:
            return {}
        return parse_tdd_like_annotations(path, include_polygons=include_polygons)
    if detected_format == 'coco':
        return parse_coco_annotations(path)
    return {}


def sort_labels(labels: Iterable[str]) -> List[str]:
    return sorted(labels, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))


def build_label_to_category_id(labels: Iterable[str]) -> Dict[str, int]:
    labels = list(dict.fromkeys(str(label) for label in labels))
    mapping: Dict[str, int] = {}
    next_id = 1
    for label in sort_labels(labels):
        if label.isdigit():
            mapping[label] = int(label)
            next_id = max(next_id, int(label) + 1)
    for label in sort_labels(labels):
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
    return mapping


def split_case_ids(case_ids: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[str]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError('Split ratios must sum to a positive value.')
    normalized = [train_ratio / total, val_ratio / total, test_ratio / total]
    ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(round(n * normalized[0]))
    val_end = train_end + int(round(n * normalized[1]))
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    return {
        'train': sorted(train_ids),
        'val': sorted(val_ids),
        'test': sorted(test_ids),
    }
