#!/usr/bin/env python3
"""Export the TDD dataset into training-ready subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tdd_common import (
    NUMERIC_TOOTH_LABELS,
    ensure_link,
    image_size,
    load_cases,
    objects_for_case,
    polygon_objects_for_case,
    threshold_mask,
)


TASKS = {
    "segmentation_teeth_binary",
    "segmentation_maxillomandibular_binary",
    "segmentation_teeth_32class",
    "detection_teeth_coco",
}

NUMERIC_TOOTH_LABEL_SET = set(NUMERIC_TOOTH_LABELS)


def sort_labels(labels):
    return sorted(labels, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))


def build_label_to_category_id(labels):
    mapping = {}
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


def write_cases(cases, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            row = {
                "case_id": case["case_id"],
                "radiograph": str(case["radiograph"].resolve()),
                "teeth_mask": str(case["teeth_mask"].resolve()),
                "maxillomandibular_mask": str(case["maxillomandibular_mask"].resolve()),
                "bbox_count": case["bbox_count"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_segmentation(cases, output_root: Path, image_mode: str, threshold: int, mask_key: str, task_name: str):
    task_root = output_root / task_name
    images_dir = task_root / "images"
    masks_dir = task_root / "masks"
    stats = []
    for case in cases:
        image_name = f"{case['case_id']}{case['radiograph'].suffix.lower()}"
        ensure_link(case["radiograph"], images_dir / image_name, mode=image_mode)
        stat = threshold_mask(case[mask_key], masks_dir / f"{case['case_id']}.png", threshold=threshold)
        stats.append({
            "case_id": case["case_id"],
            "foreground_ratio": stat["foreground_ratio"],
        })
    return {
        "task_name": task_name,
        "case_count": len(cases),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "foreground_ratio_avg": (sum(item["foreground_ratio"] for item in stats) / len(stats)) if stats else 0.0,
    }


def export_multiclass_segmentation(cases, output_root: Path, image_mode: str):
    task_root = output_root / "segmentation_teeth_32class"
    images_dir = task_root / "images"
    masks_dir = task_root / "masks"
    label_values = set()
    skipped_labels = {}
    empty_cases = []

    for case in cases:
        if case.get("polygon_item") is None:
            raise ValueError(f"Case {case['case_id']} is missing polygon annotations for 32-class export.")
        image_name = f"{case['case_id']}{case['radiograph'].suffix.lower()}"
        ensure_link(case["radiograph"], images_dir / image_name, mode=image_mode)
        width, height = case["image_size"]
        label = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(label)
        case_values = set()
        for obj in polygon_objects_for_case(case):
            tooth_label = str(obj.get("title", "")).strip()
            if tooth_label not in NUMERIC_TOOTH_LABEL_SET:
                if tooth_label:
                    skipped_labels[tooth_label] = skipped_labels.get(tooth_label, 0) + 1
                continue
            class_id = int(tooth_label)
            for polygon in obj.get("polygons") or []:
                points = [
                    (float(point[0]), float(point[1]))
                    for point in polygon
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
                if len(points) < 3:
                    continue
                draw.polygon(points, fill=class_id)
                case_values.add(class_id)
        if not case_values:
            empty_cases.append(case["case_id"])
        label_values.update(case_values)
        label.save(masks_dir / f"{case['case_id']}.png")

    return {
        "task_name": "segmentation_teeth_32class",
        "case_count": len(cases),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "label_values_present": sorted(label_values),
        "skipped_labels": dict(sorted(skipped_labels.items())),
        "empty_cases": empty_cases[:50],
    }


def export_detection(cases, output_root: Path, image_mode: str):
    task_root = output_root / "detection_teeth_coco"
    images_dir = task_root / "images"
    ann_dir = task_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        str(obj.get("title", "")).strip()
        for case in cases
        for obj in objects_for_case(case)
        if str(obj.get("title", "")).strip()
    }
    label_to_category_id = build_label_to_category_id(labels)

    images = []
    annotations = []
    ann_id = 1

    for image_id, case in enumerate(cases, start=1):
        image_name = f"{case['case_id']}{case['radiograph'].suffix.lower()}"
        ensure_link(case["radiograph"], images_dir / image_name, mode=image_mode)
        width, height = case["image_size"]
        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
            "case_id": case["case_id"],
        })
        for obj in objects_for_case(case):
            label = str(obj.get("title", "")).strip()
            bbox = obj.get("bounding box") or []
            if len(bbox) != 4 or not label:
                continue
            x1, y1, x2, y2 = bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": label_to_category_id[label],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "tooth_label": label,
            })
            ann_id += 1

    categories = [
        {"id": label_to_category_id[label], "name": f"tooth_{label}", "label": label}
        for label in sort_labels(labels)
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ann_path = ann_dir / "instances.json"
    ann_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "task_name": "detection_teeth_coco",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "annotation_path": str(ann_path),
        "non_numeric_labels": [label for label in sort_labels(labels) if not label.isdigit()],
    }


def main():
    parser = argparse.ArgumentParser(description="Export TDD into training-ready subsets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the TDD dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Export destination root.")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS), choices=sorted(TASKS), help="Tasks to export.")
    parser.add_argument("--image-mode", choices=["symlink", "copy"], default="symlink", help="How to place source radiographs into exports.")
    parser.add_argument("--threshold", type=int, default=127, help="Threshold for JPEG mask binarization.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of matched cases to export.")
    args = parser.parse_args()

    cases = [case for case in load_cases(args.dataset_root) if case["has_all_assets"]]
    for case in cases:
        case["image_size"] = image_size(case["radiograph"])
    if args.limit is not None:
        cases = cases[:args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_cases(cases, args.output_root / "cases.jsonl")

    manifest = {
        "dataset_name": "TDD",
        "dataset_root": str(args.dataset_root.resolve()),
        "case_count": len(cases),
        "tasks": args.tasks,
        "image_mode": args.image_mode,
        "threshold": args.threshold,
        "exports": {},
    }

    if "segmentation_teeth_binary" in args.tasks:
        manifest["exports"]["segmentation_teeth_binary"] = export_segmentation(
            cases,
            args.output_root,
            image_mode=args.image_mode,
            threshold=args.threshold,
            mask_key="teeth_mask",
            task_name="segmentation_teeth_binary",
        )
    if "segmentation_maxillomandibular_binary" in args.tasks:
        manifest["exports"]["segmentation_maxillomandibular_binary"] = export_segmentation(
            cases,
            args.output_root,
            image_mode=args.image_mode,
            threshold=args.threshold,
            mask_key="maxillomandibular_mask",
            task_name="segmentation_maxillomandibular_binary",
        )
    if "segmentation_teeth_32class" in args.tasks:
        manifest["exports"]["segmentation_teeth_32class"] = export_multiclass_segmentation(
            cases,
            args.output_root,
            image_mode=args.image_mode,
        )
    if "detection_teeth_coco" in args.tasks:
        manifest["exports"]["detection_teeth_coco"] = export_detection(
            cases,
            args.output_root,
            image_mode=args.image_mode,
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
