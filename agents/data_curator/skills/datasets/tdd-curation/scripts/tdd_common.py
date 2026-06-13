#!/usr/bin/env python3
"""Common helpers for TDD dataset probing and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


RADIOGRAPHS_DIRNAME = "Radiographs"
SEG_DIRNAME = "Segmentation"
TEETH_MASK_DIRNAME = "teeth_mask"
MAXILLO_DIRNAME = "maxillomandibular"
BBOX_JSON = "teeth_bbox.json"
POLYGON_JSON = "teeth_polygon.json"
NUMERIC_TOOTH_LABELS = tuple(str(index) for index in range(1, 33))


def resolve_dataset_root(dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def dataset_paths(dataset_root: Path) -> Dict[str, Path]:
    root = resolve_dataset_root(dataset_root)
    seg_root = root / SEG_DIRNAME
    return {
        "root": root,
        "radiographs": root / RADIOGRAPHS_DIRNAME,
        "seg_root": seg_root,
        "teeth_mask": seg_root / TEETH_MASK_DIRNAME,
        "maxillomandibular": seg_root / MAXILLO_DIRNAME,
        "bbox_json": seg_root / BBOX_JSON,
        "polygon_json": seg_root / POLYGON_JSON,
    }


def list_case_paths(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    mapping = {}
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if path.is_file():
            mapping[path.stem] = path
    return mapping


def load_bbox_items(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in bbox json: {path}")
    return data


def load_polygon_items(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in polygon json: {path}")
    return data


def build_bbox_index(items: Iterable[dict]) -> Dict[str, dict]:
    index = {}
    for item in items:
        external_id = item.get("External ID")
        if not external_id:
            continue
        index[Path(external_id).stem] = item
    return index


def load_cases(dataset_root: Path) -> List[dict]:
    paths = dataset_paths(dataset_root)
    radiographs = list_case_paths(paths["radiographs"])
    teeth_masks = list_case_paths(paths["teeth_mask"])
    maxillo_masks = list_case_paths(paths["maxillomandibular"])
    bbox_index = build_bbox_index(load_bbox_items(paths["bbox_json"]))
    polygon_index = build_bbox_index(load_polygon_items(paths["polygon_json"]))

    case_ids = sorted(set(radiographs) | set(teeth_masks) | set(maxillo_masks) | set(bbox_index) | set(polygon_index))
    cases = []
    for case_id in case_ids:
        bbox_item = bbox_index.get(case_id, {})
        polygon_item = polygon_index.get(case_id, {})
        objects = bbox_item.get("Label", {}).get("objects", [])
        polygon_objects = polygon_item.get("Label", {}).get("objects", [])
        cases.append({
            "case_id": case_id,
            "radiograph": radiographs.get(case_id),
            "teeth_mask": teeth_masks.get(case_id),
            "maxillomandibular_mask": maxillo_masks.get(case_id),
            "bbox_item": bbox_item if bbox_item else None,
            "polygon_item": polygon_item if polygon_item else None,
            "bbox_count": len(objects),
            "polygon_count": len(polygon_objects),
            "has_all_assets": all([
                case_id in radiographs,
                case_id in teeth_masks,
                case_id in maxillo_masks,
                case_id in bbox_index,
            ]),
        })
    return cases


def open_grayscale(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "L":
        image = image.convert("L")
    return image


def image_size(path: Path):
    with Image.open(path) as image:
        return image.size


def threshold_mask(input_path: Path, output_path: Path, threshold: int = 127) -> dict:
    gray = open_grayscale(input_path)
    binary = gray.point(lambda x: 255 if x > threshold else 0, mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary.save(output_path)
    histogram = binary.histogram()
    fg = histogram[255] if len(histogram) > 255 else 0
    total = sum(histogram)
    return {
        "output_path": str(output_path),
        "foreground_ratio": (fg / total) if total else 0.0,
    }


def ensure_link(src: Path, dst: Path, mode: str = "symlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        from shutil import copy2
        copy2(src, dst)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def objects_for_case(case: dict) -> List[dict]:
    bbox_item = case.get("bbox_item") or {}
    return bbox_item.get("Label", {}).get("objects", [])


def polygon_objects_for_case(case: dict) -> List[dict]:
    polygon_item = case.get("polygon_item") or {}
    return polygon_item.get("Label", {}).get("objects", [])
