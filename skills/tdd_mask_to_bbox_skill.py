from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def _case_key(path: Path) -> str:
    stem = path.stem

    # 兼容nnU-Net常见命名，例如case001_0000.png
    if stem.endswith("_0000"):
        stem = stem[:-5]

    return stem


def _collect_files(
    directory: Path,
    suffixes: Optional[set] = None,
) -> Dict[str, Path]:
    if not directory.exists():
        return {}

    result: Dict[str, Path] = {}

    for path in directory.iterdir():
        if not path.is_file():
            continue

        if suffixes is not None:
            if path.suffix.lower() not in suffixes:
                continue

        result[_case_key(path)] = path

    return result


def _link_or_copy(
    source: Path,
    target: Path,
    mode: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        target.unlink()

    if mode == "symlink":
        target.symlink_to(source.resolve())
        return

    if mode == "hardlink":
        try:
            os.link(str(source), str(target))
            return
        except OSError:
            shutil.copy2(source, target)
            return

    if mode == "copy":
        shutil.copy2(source, target)
        return

    raise ValueError(
        "Unsupported link mode: {}".format(mode)
    )


def _mask_to_yolo_boxes(
    mask: np.ndarray,
    class_count: int,
    min_area: int,
) -> Tuple[List[str], Dict[str, Any]]:
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be 2D, got shape {}".format(
                mask.shape
            )
        )

    height, width = mask.shape
    values = np.unique(mask)

    invalid_values = [
        int(value)
        for value in values.tolist()
        if int(value) < 0 or int(value) > class_count
    ]

    lines: List[str] = []
    removed_small = 0

    for raw_class_id in values.tolist():
        class_id = int(raw_class_id)

        # 0为背景
        if class_id == 0:
            continue

        if class_id < 1 or class_id > class_count:
            continue

        ys, xs = np.where(mask == class_id)

        if len(xs) == 0:
            continue

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        box_width = x_max - x_min
        box_height = y_max - y_min
        area = box_width * box_height

        if area < min_area:
            removed_small += 1
            continue

        x_center = (
            (x_min + x_max) / 2.0
        ) / float(width)

        y_center = (
            (y_min + y_max) / 2.0
        ) / float(height)

        norm_width = box_width / float(width)
        norm_height = box_height / float(height)

        # 原掩码类别1-32转换为YOLO类别0-31
        yolo_class_id = class_id - 1

        lines.append(
            "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
                yolo_class_id,
                x_center,
                y_center,
                norm_width,
                norm_height,
            )
        )

    summary = {
        "image_width": width,
        "image_height": height,
        "label_values": [
            int(value) for value in values.tolist()
        ],
        "invalid_values": invalid_values,
        "box_count": len(lines),
        "removed_small_boxes": removed_small,
    }

    return lines, summary


def _write_data_yaml(
    output_root: Path,
    class_names: List[str],
    has_test: bool,
) -> Path:
    yaml_path = output_root / "data.yaml"

    lines = [
        "path: {}".format(output_root.resolve()),
        "train: images/train",
        "val: images/val",
    ]

    if has_test:
        lines.append("test: images/test")

    lines.extend(
        [
            "nc: {}".format(len(class_names)),
            "names:",
        ]
    )

    for index, name in enumerate(class_names):
        safe_name = str(name).replace('"', '\\"')
        lines.append(
            '  {}: "{}"'.format(index, safe_name)
        )

    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return yaml_path


def convert_tdd_masks_to_yolo(
    source_root: str,
    output_root: str,
    class_count: int = 32,
    min_area: int = 16,
    link_mode: str = "symlink",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    source_path = Path(source_root).resolve()
    output_path = Path(output_root).resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if class_names is None:
        class_names = [
            "tooth_{:02d}".format(index)
            for index in range(1, class_count + 1)
        ]

    if len(class_names) != class_count:
        raise ValueError(
            "class_names length {} does not match "
            "class_count {}".format(
                len(class_names),
                class_count,
            )
        )

    split_definitions = {
        "train": ("imagesTr", "labelsTr", True),
        "val": ("imagesVal", "labelsVal", True),
        "test": ("imagesTs", "labelsTs", False),
    }

    audit: Dict[str, Any] = {
        "source_root": str(source_path),
        "output_root": str(output_path),
        "class_count": class_count,
        "min_area": min_area,
        "link_mode": link_mode,
        "splits": {},
        "issues": [],
    }

    for (
        split_name,
        (
            image_folder_name,
            label_folder_name,
            labels_required,
        ),
    ) in split_definitions.items():
        image_dir = source_path / image_folder_name
        label_dir = source_path / label_folder_name

        image_map = _collect_files(
            image_dir,
            IMAGE_SUFFIXES,
        )

        label_map = _collect_files(
            label_dir,
            IMAGE_SUFFIXES,
        )

        destination_images = (
            output_path / "images" / split_name
        )
        destination_labels = (
            output_path / "labels" / split_name
        )

        destination_images.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_report: Dict[str, Any] = {
            "image_count": len(image_map),
            "label_count": len(label_map),
            "converted_cases": 0,
            "total_boxes": 0,
            "missing_labels": [],
            "empty_annotations": [],
            "invalid_label_cases": [],
            "case_summaries": [],
        }

        for case_id, image_path in sorted(
            image_map.items()
        ):
            output_image = (
                destination_images
                / "{}{}".format(
                    case_id,
                    image_path.suffix.lower(),
                )
            )

            _link_or_copy(
                image_path,
                output_image,
                link_mode,
            )

            mask_path = label_map.get(case_id)

            if mask_path is None:
                if labels_required:
                    split_report[
                        "missing_labels"
                    ].append(case_id)

                    audit["issues"].append(
                        {
                            "split": split_name,
                            "case_id": case_id,
                            "issue_type": (
                                "missing_mask_annotation"
                            ),
                        }
                    )
                continue

            mask = np.asarray(
                Image.open(mask_path)
            )

            yolo_lines, case_summary = (
                _mask_to_yolo_boxes(
                    mask=mask,
                    class_count=class_count,
                    min_area=min_area,
                )
            )

            output_label = (
                destination_labels
                / "{}.txt".format(case_id)
            )

            output_label.write_text(
                "\n".join(yolo_lines)
                + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

            if not yolo_lines:
                split_report[
                    "empty_annotations"
                ].append(case_id)

            if case_summary["invalid_values"]:
                split_report[
                    "invalid_label_cases"
                ].append(
                    {
                        "case_id": case_id,
                        "invalid_values": case_summary[
                            "invalid_values"
                        ],
                    }
                )

            split_report["converted_cases"] += 1
            split_report["total_boxes"] += (
                case_summary["box_count"]
            )

            split_report[
                "case_summaries"
            ].append(
                {
                    "case_id": case_id,
                    **case_summary,
                }
            )

        audit["splits"][split_name] = split_report

    has_test = (
        audit["splits"]["test"]["image_count"] > 0
    )

    data_yaml = _write_data_yaml(
        output_root=output_path,
        class_names=class_names,
        has_test=has_test,
    )

    audit["data_yaml"] = str(data_yaml)

    audit_path = output_path / "conversion_audit.json"
    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit["audit_path"] = str(audit_path)

    return audit
