#!/usr/bin/env python3
"""Inspect the local TDD dataset and write a profile report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from tdd_common import dataset_paths, image_size, load_bbox_items, load_cases, load_polygon_items


def summarize(dataset_root: Path) -> dict:
    paths = dataset_paths(dataset_root)
    cases = load_cases(dataset_root)
    matched_cases = [case for case in cases if case["has_all_assets"]]
    radiograph_sizes = Counter(image_size(case["radiograph"]) for case in matched_cases[:50])
    bbox_items = load_bbox_items(paths["bbox_json"])
    polygon_items = load_polygon_items(paths["polygon_json"])
    bbox_counts = [len(item.get("Label", {}).get("objects", [])) for item in bbox_items]
    polygon_counts = [len(item.get("Label", {}).get("objects", [])) for item in polygon_items]
    category_counter = Counter()
    for item in polygon_items:
        for obj in item.get("Label", {}).get("objects", []):
            title = str(obj.get("title", "")).strip()
            if title:
                category_counter[title] += 1

    sample_teeth = paths["teeth_mask"] / f"{matched_cases[0]['case_id']}.jpg"
    sample_maxillo = paths["maxillomandibular"] / f"{matched_cases[0]['case_id']}.jpg"
    with Image.open(sample_teeth) as image:
        teeth_mode = image.mode
    with Image.open(sample_maxillo) as image:
        maxillo_mode = image.mode

    return {
        "dataset_name": "TDD",
        "dataset_root": str(paths["root"]),
        "case_count": len(cases),
        "fully_matched_case_count": len(matched_cases),
        "counts": {
            "radiographs": len(list(paths["radiographs"].iterdir())),
            "teeth_mask_files": len(list(paths["teeth_mask"].iterdir())),
            "maxillomandibular_mask_files": len(list(paths["maxillomandibular"].iterdir())),
            "bbox_entries": len(bbox_items),
            "polygon_entries": len(polygon_items),
        },
        "modalities": ["2d panoramic radiograph"],
        "task_candidates": [
            "teeth_detection_coco",
            "teeth_binary_segmentation",
            "maxillomandibular_binary_segmentation",
            "teeth_32class_segmentation",
        ],
        "image_size_sample_distribution": [
            {"size": list(size), "count": count}
            for size, count in radiograph_sizes.items()
        ],
        "annotation_sources": {
            "teeth_bbox_json": str(paths["bbox_json"]),
            "teeth_polygon_json": str(paths["polygon_json"]),
            "teeth_mask_dir": str(paths["teeth_mask"]),
            "maxillomandibular_dir": str(paths["maxillomandibular"]),
        },
        "mask_notes": {
            "teeth_mask_mode": teeth_mode,
            "maxillomandibular_mode": maxillo_mode,
            "export_strategy": "JPEG masks are thresholded to binary PNG during export (default threshold=127).",
        },
        "bbox_stats": {
            "min_boxes_per_case": min(bbox_counts) if bbox_counts else 0,
            "max_boxes_per_case": max(bbox_counts) if bbox_counts else 0,
            "avg_boxes_per_case": (sum(bbox_counts) / len(bbox_counts)) if bbox_counts else 0.0,
            "tooth_categories_present": sorted([x for x in category_counter.keys() if x.isdigit()], key=lambda x: int(x)),
            "non_numeric_categories_present": sorted([x for x in category_counter.keys() if not x.isdigit()]),
        },
        "polygon_stats": {
            "min_polygons_per_case": min(polygon_counts) if polygon_counts else 0,
            "max_polygons_per_case": max(polygon_counts) if polygon_counts else 0,
            "avg_polygons_per_case": (sum(polygon_counts) / len(polygon_counts)) if polygon_counts else 0.0,
        },
        "integrity": {
            "all_cases_have_radiograph": all(case["radiograph"] is not None for case in matched_cases),
            "all_cases_have_teeth_mask": all(case["teeth_mask"] is not None for case in matched_cases),
            "all_cases_have_maxillomandibular_mask": all(case["maxillomandibular_mask"] is not None for case in matched_cases),
            "all_cases_have_bbox": all(case["bbox_item"] is not None for case in matched_cases),
            "all_cases_have_polygon": all(case["polygon_item"] is not None for case in matched_cases),
            "missing_asset_cases": [
                case["case_id"] for case in cases if not case["has_all_assets"]
            ][:50],
        },
        "recommended_exports": {
            "detection": "COCO-style teeth detection from teeth_bbox.json",
            "segmentation_teeth_binary": "Binary PNG masks thresholded from Segmentation/teeth_mask",
            "segmentation_maxillomandibular_binary": "Binary PNG masks thresholded from Segmentation/maxillomandibular",
            "segmentation_teeth_32class": "32-class PNG masks rasterized from teeth_polygon.json numeric tooth labels 1..32",
        },
    }


def write_markdown(summary: dict, path: Path):
    lines = [
        "# TDD Dataset Profile",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Total cases: {summary['case_count']}",
        f"- Fully matched cases: {summary['fully_matched_case_count']}",
        "",
        "## Counts",
        "",
        f"- Radiographs: {summary['counts']['radiographs']}",
        f"- Teeth masks: {summary['counts']['teeth_mask_files']}",
        f"- Maxillomandibular masks: {summary['counts']['maxillomandibular_mask_files']}",
        f"- BBox entries: {summary['counts']['bbox_entries']}",
        f"- Polygon entries: {summary['counts']['polygon_entries']}",
        "",
        "## Supported Export Targets",
        "",
        f"- Detection: {summary['recommended_exports']['detection']}",
        f"- Teeth segmentation: {summary['recommended_exports']['segmentation_teeth_binary']}",
        f"- Maxillomandibular segmentation: {summary['recommended_exports']['segmentation_maxillomandibular_binary']}",
        f"- Teeth 32-class segmentation: {summary['recommended_exports']['segmentation_teeth_32class']}",
        "",
        "## Notes",
        "",
        f"- Teeth mask mode: {summary['mask_notes']['teeth_mask_mode']}",
        f"- Maxillomandibular mask mode: {summary['mask_notes']['maxillomandibular_mode']}",
        f"- Export strategy: {summary['mask_notes']['export_strategy']}",
        "",
        "## BBox Stats",
        "",
        f"- Min boxes/case: {summary['bbox_stats']['min_boxes_per_case']}",
        f"- Max boxes/case: {summary['bbox_stats']['max_boxes_per_case']}",
        f"- Avg boxes/case: {summary['bbox_stats']['avg_boxes_per_case']:.2f}",
        f"- Non-numeric labels: {', '.join(summary['bbox_stats']['non_numeric_categories_present']) if summary['bbox_stats']['non_numeric_categories_present'] else 'None'}",
        "",
        "## Polygon Stats",
        "",
        f"- Min polygons/case: {summary['polygon_stats']['min_polygons_per_case']}",
        f"- Max polygons/case: {summary['polygon_stats']['max_polygons_per_case']}",
        f"- Avg polygons/case: {summary['polygon_stats']['avg_polygons_per_case']:.2f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Probe the local TDD dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the TDD dataset root.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional Markdown report output path.")
    args = parser.parse_args()

    summary = summarize(args.dataset_root)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(summary, args.output_md)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
