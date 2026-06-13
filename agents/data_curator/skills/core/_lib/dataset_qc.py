#!/usr/bin/env python3
"""Shared dataset QC helpers for Data Curator skills."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from curation_core import case_stem, normalized_suffix

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # pragma: no cover - optional dependency
    scipy_ndimage = None  # type: ignore[assignment]

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None  # type: ignore[assignment]

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency
    sitk = None  # type: ignore[assignment]

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency
    pydicom = None  # type: ignore[assignment]


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[5]
DEFAULT_QC_REPORT_ROOT = REPO_ROOT / "artifacts" / "results" / "reports" / "datasets_qc"

NUMERIC_TOOTH_LABELS = {str(index) for index in range(1, 33)}
PRIMARY_TOOTH_LABELS = set("ABCDEFGHIJKLMNOPQRST")
DEFAULT_TINY_FRAGMENT_PIXELS_2D = 16
MAX_COMPONENT_ANALYSIS_PIXELS = 900_000
NEAREST_RESAMPLE = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_report_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "dataset_qc"


def build_default_report_paths(report_key: str, report_root: Path | None = None) -> Tuple[Path, Path]:
    root = ensure_dir((report_root or DEFAULT_QC_REPORT_ROOT).resolve())
    key = sanitize_report_key(report_key)
    return root / f"{key}.qc.json", root / f"{key}.qc.md"


def normalize_case_token(case_id: str) -> str:
    lowered = str(case_id).strip().lower()
    lowered = re.sub(r"(?:_copy|-copy|copy|duplicate|dup|aug|augment|flip|rot\d+)$", "", lowered)
    return re.sub(r"[^a-z0-9]+", "", lowered)


def normalized_asset_case_id(path: Path) -> str:
    stem = case_stem(path)
    # nnU-Net names image channels like case_0000.png, case_0001.png, ...
    return re.sub(r"_\d{4}$", "", stem)


def safe_sha1(path: Path, max_bytes: int = 128 * 1024 * 1024) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        hasher = hashlib.sha1()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def detect_numbering_schema(labels: Iterable[str]) -> Dict[str, Any]:
    values = [str(label).strip() for label in labels if str(label).strip()]
    numeric = sorted({label for label in values if label.isdigit()}, key=int)
    primary = sorted({label for label in values if label in PRIMARY_TOOTH_LABELS})
    invalid = sorted({
        label for label in values
        if (not label.isdigit()) and (label not in PRIMARY_TOOTH_LABELS)
    })
    if numeric and primary:
        schema = "mixed"
    elif numeric:
        schema = "numeric"
    elif primary:
        schema = "primary"
    elif values:
        schema = "nonstandard"
    else:
        schema = "none"
    invalid_numeric = sorted(label for label in numeric if label not in NUMERIC_TOOTH_LABELS)
    return {
        "schema": schema,
        "numeric_labels": numeric,
        "primary_labels": primary,
        "invalid_labels": sorted(set(invalid + invalid_numeric)),
    }


def mask_allowed_values_for_role(role: str) -> Optional[set[int]]:
    role_lower = role.lower()
    if "32class" in role_lower or "multiclass" in role_lower:
        return set(range(33))
    if any(token in role_lower for token in ("mask", "binary", "seg", "label", "teeth_mask", "maxillomandibular")):
        return {0, 1, 255}
    return None


def _component_summary(mask_array: Any, label_values: Sequence[int]) -> Optional[Dict[str, Any]]:
    if np is None or scipy_ndimage is None:
        return None
    summary: Dict[str, Any] = {}
    foreground = mask_array > 0
    labeled, count = scipy_ndimage.label(foreground)
    sizes = np.bincount(labeled.ravel())[1:] if count else np.array([], dtype=int)
    summary["foreground_component_count"] = int(count)
    summary["tiny_foreground_component_count"] = int(np.sum(sizes < DEFAULT_TINY_FRAGMENT_PIXELS_2D)) if sizes.size else 0
    summary["largest_foreground_component_pixels"] = int(sizes.max()) if sizes.size else 0
    per_label = {}
    for label_value in label_values:
        if label_value == 0:
            continue
        label_mask = mask_array == label_value
        label_labeled, label_count = scipy_ndimage.label(label_mask)
        label_sizes = np.bincount(label_labeled.ravel())[1:] if label_count else np.array([], dtype=int)
        per_label[str(label_value)] = {
            "component_count": int(label_count),
            "tiny_component_count": int(np.sum(label_sizes < DEFAULT_TINY_FRAGMENT_PIXELS_2D)) if label_sizes.size else 0,
            "pixel_count": int(label_mask.sum()),
        }
    summary["per_label"] = per_label
    return summary


def inspect_raster_label(path: Path) -> Dict[str, Any]:
    with Image.open(path) as image:
        label = image if image.mode in {"1", "L", "I", "P"} else image.convert("L")
        histogram = label.histogram()
        values = [index for index, count in enumerate(histogram) if count]
        foreground_pixels = sum(histogram[1:])
        pixel_count = sum(histogram)
        payload = {
            "readable": True,
            "mode": label.mode,
            "size": [int(label.size[0]), int(label.size[1])],
            "unique_values": values,
            "foreground_pixels": foreground_pixels,
            "foreground_ratio": (foreground_pixels / pixel_count) if pixel_count else 0.0,
            "empty": foreground_pixels == 0,
        }
        if np is not None and scipy_ndimage is not None:
            analysis_image = label
            sampled = False
            width, height = label.size
            area = width * height
            if area > MAX_COMPONENT_ANALYSIS_PIXELS:
                scale = (MAX_COMPONENT_ANALYSIS_PIXELS / float(area)) ** 0.5
                resized = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                analysis_image = label.resize(resized, NEAREST_RESAMPLE)
                sampled = True
            label_array = np.asarray(analysis_image)
            payload["components"] = {
                **(_component_summary(label_array, values) or {}),
                "sampled": sampled,
                "analysis_size": [int(analysis_image.size[0]), int(analysis_image.size[1])],
            }
        return payload


def inspect_raster_image(path: Path) -> Dict[str, Any]:
    with Image.open(path) as image:
        return {
            "readable": True,
            "mode": image.mode,
            "size": [int(image.size[0]), int(image.size[1])],
        }


def inspect_volume(path: Path) -> Dict[str, Any]:
    suffix = normalized_suffix(path)
    if suffix in {".nii", ".nii.gz"} and nib is not None:
        image = nib.load(str(path))
        return {
            "readable": True,
            "shape": [int(dim) for dim in image.shape],
            "ndim": int(len(image.shape)),
            "spacing": [float(value) for value in image.header.get_zooms()[: len(image.shape)]],
            "reader": "nibabel",
        }
    if suffix in {".mha", ".mhd", ".nrrd"} and sitk is not None:
        image = sitk.ReadImage(str(path))
        return {
            "readable": True,
            "shape": [int(dim) for dim in image.GetSize()],
            "ndim": int(image.GetDimension()),
            "spacing": [float(value) for value in image.GetSpacing()],
            "reader": "SimpleITK",
        }
    if suffix == ".dcm" and pydicom is not None:
        ds = pydicom.dcmread(str(path), force=True)
        shape = []
        if hasattr(ds, "NumberOfFrames"):
            shape.append(int(ds.NumberOfFrames))
        if hasattr(ds, "Rows"):
            shape.append(int(ds.Rows))
        if hasattr(ds, "Columns"):
            shape.append(int(ds.Columns))
        spacing = []
        if hasattr(ds, "PixelSpacing"):
            spacing.extend(float(value) for value in ds.PixelSpacing)
        if hasattr(ds, "SliceThickness"):
            spacing.append(float(ds.SliceThickness))
        return {
            "readable": True,
            "shape": shape,
            "ndim": len(shape),
            "spacing": spacing or None,
            "reader": "pydicom",
            "modality": getattr(ds, "Modality", None),
        }
    raise RuntimeError(f"No available reader for volume path: {path}")


def _record_issue(
    issues: List[Dict[str, Any]],
    case_issues: Dict[str, List[Dict[str, Any]]],
    *,
    severity: str,
    category: str,
    issue_type: str,
    message: str,
    case_id: Optional[str] = None,
    **details: Any,
) -> None:
    issue = {
        "severity": severity,
        "category": category,
        "type": issue_type,
        "message": message,
    }
    if case_id is not None:
        issue["case_id"] = case_id
    if details:
        issue["details"] = details
    issues.append(issue)
    if case_id is not None:
        case_issues.setdefault(case_id, []).append(issue)


def _classify_case_status(case_issue_list: Sequence[Dict[str, Any]]) -> str:
    severities = {issue["severity"] for issue in case_issue_list}
    if "error" in severities:
        return "blocked"
    if "warning" in severities:
        return "manual_review"
    return "ready"


def _flatten_polygons(annotation: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    polygons = annotation.get("polygons")
    if polygons:
        result = []
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)):
                continue
            points = []
            for point in polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if points:
                result.append(points)
        return result
    polygon = annotation.get("polygon")
    if isinstance(polygon, (list, tuple)):
        points = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        return [points] if points else []
    return []


def _build_split_lookup(split_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not split_payload:
        return {}
    splits = split_payload.get("splits", split_payload)
    lookup = {}
    if not isinstance(splits, dict):
        return lookup
    for split_name, case_ids in splits.items():
        for case_id in case_ids or []:
            lookup[str(case_id)] = str(split_name)
    return lookup


def run_dataset_qc(
    *,
    dataset_root: Path,
    dataset_name: str,
    dataset_mode: str,
    cases: Sequence[Dict[str, Any]],
    split_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    issues: List[Dict[str, Any]] = []
    case_issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    split_lookup = _build_split_lookup(split_payload)
    modality_counter: Counter[str] = Counter()
    numbering_schema_counter: Counter[str] = Counter()
    duplicate_role_counter: Counter[str] = Counter()

    case_records = []
    primary_fingerprints: Dict[str, Dict[str, Any]] = {}
    normalized_case_ids: Dict[str, List[str]] = defaultdict(list)

    for case in cases:
        case_id = str(case["case_id"])
        normalized_case_ids[normalize_case_token(case_id)].append(case_id)

        image_entries = [
            {"role": str(entry.get("role", "image")), "path": Path(entry["path"]).resolve()}
            for entry in case.get("images", [])
        ]
        raster_entries = [
            {"role": str(entry.get("role", "label")), "path": Path(entry["path"]).resolve()}
            for entry in case.get("raster_labels", [])
        ]
        bbox_entries = list(case.get("bbox_annotations", []))
        polygon_entries = list(case.get("polygon_annotations", []))
        case_report = {
            "case_id": case_id,
            "split": split_lookup.get(case_id),
            "image_roles": [entry["role"] for entry in image_entries],
            "label_roles": [entry["role"] for entry in raster_entries],
            "issues": [],
            "metrics": {},
        }

        if not image_entries:
            _record_issue(
                issues,
                case_issues,
                severity="error",
                category="completeness",
                issue_type="missing_image",
                case_id=case_id,
                message="Case is missing its primary image/volume.",
            )
            case_records.append(case_report)
            continue

        role_counts = Counter(entry["role"] for entry in image_entries)
        duplicate_image_roles = [role for role, count in role_counts.items() if count > 1]
        if duplicate_image_roles:
            duplicate_role_counter["image_roles"] += len(duplicate_image_roles)
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="correspondence",
                issue_type="duplicate_image_roles",
                case_id=case_id,
                message="Case has multiple image files under the same role.",
                roles=duplicate_image_roles,
            )

        label_role_counts = Counter(entry["role"] for entry in raster_entries)
        duplicate_label_roles = [role for role, count in label_role_counts.items() if count > 1]
        if duplicate_label_roles:
            duplicate_role_counter["label_roles"] += len(duplicate_label_roles)
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="correspondence",
                issue_type="duplicate_label_roles",
                case_id=case_id,
                message="Case has multiple label files under the same role.",
                roles=duplicate_label_roles,
            )

        primary_image = image_entries[0]
        primary_path = primary_image["path"]
        primary_suffix = normalized_suffix(primary_path)
        image_info: Dict[str, Any]
        try:
            if primary_suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                image_info = inspect_raster_image(primary_path)
                modality = "2d"
            else:
                image_info = inspect_volume(primary_path)
                modality = "3d"
        except Exception as exc:
            _record_issue(
                issues,
                case_issues,
                severity="error",
                category="completeness",
                issue_type="unreadable_image",
                case_id=case_id,
                message="Primary image/volume could not be read.",
                path=str(primary_path),
                error=str(exc),
            )
            case_records.append(case_report)
            continue

        modality_counter[modality] += 1
        case_report["metrics"]["primary_image"] = {
            "path": str(primary_path),
            **image_info,
        }

        if normalized_asset_case_id(primary_path) != case_id:
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="correspondence",
                issue_type="image_case_id_mismatch",
                case_id=case_id,
                message="Primary image stem does not match case id.",
                image_stem=normalized_asset_case_id(primary_path),
            )

        if modality == "2d":
            image_size = tuple(image_info["size"])
        else:
            image_size = tuple(image_info.get("shape", []))
            spacing = image_info.get("spacing")
            if not spacing:
                _record_issue(
                    issues,
                    case_issues,
                    severity="warning",
                    category="metadata_consistency",
                    issue_type="missing_spacing",
                    case_id=case_id,
                    message="Volume spacing metadata is missing.",
                    path=str(primary_path),
                )
            ndim = int(image_info.get("ndim", 0))
            if ndim not in {3, 4}:
                _record_issue(
                    issues,
                    case_issues,
                    severity="warning",
                    category="metadata_consistency",
                    issue_type="unexpected_dimensionality",
                    case_id=case_id,
                    message="Volume dimensionality is unusual for downstream medical imaging workflows.",
                    ndim=ndim,
                )

        if primary_path.is_file():
            primary_fingerprints[case_id] = {
                "path": str(primary_path),
                "sha1": safe_sha1(primary_path),
                "normalized_case_id": normalize_case_token(case_id),
            }

        if not raster_entries and not bbox_entries and not polygon_entries:
            _record_issue(
                issues,
                case_issues,
                severity="error",
                category="completeness",
                issue_type="missing_annotation",
                case_id=case_id,
                message="Case has no labels or annotations and cannot enter a labeled training flow.",
            )

        numbering_schema = detect_numbering_schema(
            [ann.get("label", "") for ann in bbox_entries] + [ann.get("label", "") for ann in polygon_entries]
        )
        numbering_schema_counter[numbering_schema["schema"]] += 1
        case_report["metrics"]["numbering_schema"] = numbering_schema
        if numbering_schema["invalid_labels"]:
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="annotation_integrity",
                issue_type="invalid_annotation_labels",
                case_id=case_id,
                message="Annotation labels fall outside the expected numbering schema.",
                invalid_labels=numbering_schema["invalid_labels"],
            )

        if numbering_schema["schema"] == "mixed":
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="numbering_schema",
                issue_type="mixed_numbering_schema",
                case_id=case_id,
                message="Case mixes permanent numeric labels and primary-letter labels.",
            )

        for label_entry in raster_entries:
            label_path = label_entry["path"]
            label_role = label_entry["role"]
            if normalized_asset_case_id(label_path) != case_id:
                _record_issue(
                    issues,
                    case_issues,
                    severity="warning",
                    category="correspondence",
                    issue_type="label_case_id_mismatch",
                    case_id=case_id,
                    message="Label stem does not match case id.",
                    label_role=label_role,
                    label_stem=normalized_asset_case_id(label_path),
                )
            try:
                if modality == "2d":
                    label_info = inspect_raster_label(label_path)
                else:
                    label_info = inspect_volume(label_path)
            except Exception as exc:
                _record_issue(
                    issues,
                    case_issues,
                    severity="error",
                    category="completeness",
                    issue_type="unreadable_annotation",
                    case_id=case_id,
                    message="Label file could not be read.",
                    path=str(label_path),
                    label_role=label_role,
                    error=str(exc),
                )
                continue

            case_report["metrics"].setdefault("labels", []).append({
                "role": label_role,
                "path": str(label_path),
                **label_info,
            })

            label_shape = tuple(label_info.get("size") or label_info.get("shape") or [])
            if image_size and label_shape and image_size != label_shape:
                _record_issue(
                    issues,
                    case_issues,
                    severity="error",
                    category="correspondence",
                    issue_type="image_label_shape_mismatch",
                    case_id=case_id,
                    message="Image and label shape/size do not match.",
                    image_shape=list(image_size),
                    label_shape=list(label_shape),
                    label_role=label_role,
                )

            if modality == "2d":
                if label_info.get("empty"):
                    _record_issue(
                        issues,
                        case_issues,
                        severity="warning",
                        category="annotation_integrity",
                        issue_type="empty_mask",
                        case_id=case_id,
                        message="Mask is empty.",
                        label_role=label_role,
                    )
                allowed_values = mask_allowed_values_for_role(label_role)
                unique_values = set(label_info.get("unique_values", []))
                if allowed_values is not None and not unique_values.issubset(allowed_values):
                    _record_issue(
                        issues,
                        case_issues,
                        severity="warning",
                        category="annotation_integrity",
                        issue_type="invalid_label_values",
                        case_id=case_id,
                        message="Mask contains label values outside the expected binary schema.",
                        label_role=label_role,
                        unique_values=sorted(unique_values),
                        expected_values=sorted(allowed_values),
                    )
                foreground_ratio = float(label_info.get("foreground_ratio", 0.0))
                if foreground_ratio > 0.98:
                    _record_issue(
                        issues,
                        case_issues,
                        severity="warning",
                        category="plausibility",
                        issue_type="implausible_foreground_ratio",
                        case_id=case_id,
                        message="Mask foreground almost fills the whole image.",
                        label_role=label_role,
                        foreground_ratio=foreground_ratio,
                    )
                components = label_info.get("components") or {}
                if components.get("tiny_foreground_component_count", 0) > 100:
                    _record_issue(
                        issues,
                        case_issues,
                        severity="warning",
                        category="plausibility",
                        issue_type="many_tiny_fragments",
                        case_id=case_id,
                        message="Mask contains many tiny disconnected fragments.",
                        label_role=label_role,
                        tiny_component_count=components.get("tiny_foreground_component_count"),
                    )
                for class_id, component_summary in (components.get("per_label") or {}).items():
                    if allowed_values is not None:
                        continue
                    if component_summary.get("component_count", 0) > 4 and component_summary.get("pixel_count", 0) > 0:
                        _record_issue(
                            issues,
                            case_issues,
                            severity="warning",
                            category="plausibility",
                            issue_type="disconnected_components",
                            case_id=case_id,
                            message="A label class has several disconnected components.",
                            label_role=label_role,
                            class_id=class_id,
                            component_count=component_summary.get("component_count"),
                        )
            else:
                spacing = label_info.get("spacing")
                image_spacing = image_info.get("spacing")
                if spacing and image_spacing and tuple(spacing) != tuple(image_spacing):
                    _record_issue(
                        issues,
                        case_issues,
                        severity="warning",
                        category="metadata_consistency",
                        issue_type="spacing_mismatch",
                        case_id=case_id,
                        message="Volume and label spacing metadata differ.",
                        image_spacing=image_spacing,
                        label_spacing=spacing,
                        label_role=label_role,
                    )

        bbox_counter = Counter(str(ann.get("label", "")).strip() for ann in bbox_entries if str(ann.get("label", "")).strip())
        duplicated_bbox_labels = sorted(label for label, count in bbox_counter.items() if count > 1)
        if duplicated_bbox_labels:
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="annotation_integrity",
                issue_type="duplicated_bbox_labels",
                case_id=case_id,
                message="Bounding-box annotations reuse the same label multiple times.",
                duplicated_labels=duplicated_bbox_labels,
            )

        for annotation in bbox_entries:
            bbox = annotation.get("bbox_xyxy") or []
            if len(bbox) != 4:
                _record_issue(
                    issues,
                    case_issues,
                    severity="error",
                    category="annotation_integrity",
                    issue_type="invalid_bbox_format",
                    case_id=case_id,
                    message="Bounding box does not contain four coordinates.",
                    bbox=bbox,
                )
                continue
            if modality == "2d" and image_size:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                width, height = image_size
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                    _record_issue(
                        issues,
                        case_issues,
                        severity="error",
                        category="plausibility",
                        issue_type="bbox_out_of_bounds",
                        case_id=case_id,
                        message="Bounding box lies outside the image extent.",
                        bbox=bbox,
                        image_size=list(image_size),
                    )

        polygon_counter = Counter(str(ann.get("label", "")).strip() for ann in polygon_entries if str(ann.get("label", "")).strip())
        duplicated_polygon_labels = sorted(label for label, count in polygon_counter.items() if count > 1)
        if duplicated_polygon_labels:
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="annotation_integrity",
                issue_type="duplicated_polygon_labels",
                case_id=case_id,
                message="Polygon annotations reuse the same label multiple times.",
                duplicated_labels=duplicated_polygon_labels,
            )

        for annotation in polygon_entries:
            for polygon in _flatten_polygons(annotation):
                if len(polygon) < 3:
                    _record_issue(
                        issues,
                        case_issues,
                        severity="error",
                        category="annotation_integrity",
                        issue_type="invalid_polygon",
                        case_id=case_id,
                        message="Polygon has fewer than three points.",
                        label=annotation.get("label"),
                    )
                    continue
                if modality == "2d" and image_size:
                    width, height = image_size
                    if any(point[0] < 0 or point[1] < 0 or point[0] > width or point[1] > height for point in polygon):
                        _record_issue(
                            issues,
                            case_issues,
                            severity="error",
                            category="plausibility",
                            issue_type="polygon_out_of_bounds",
                            case_id=case_id,
                            message="Polygon contains points outside the image extent.",
                            label=annotation.get("label"),
                        )

        case_records.append(case_report)

    # Dataset-level numbering and identifier consistency
    for normalized_token, members in sorted(normalized_case_ids.items()):
        if normalized_token and len(members) > 1:
            _record_issue(
                issues,
                case_issues,
                severity="warning",
                category="split_integrity",
                issue_type="repeated_case_identifier",
                message="Multiple case ids collapse to the same normalized identifier.",
                normalized_case_id=normalized_token,
                case_ids=sorted(members),
            )

    if split_payload:
        split_lists = split_payload.get("splits", split_payload)
        seen_case_to_split: Dict[str, str] = {}
        seen_hash_to_split: Dict[str, Tuple[str, str]] = {}
        for split_name, case_ids in (split_lists.items() if isinstance(split_lists, dict) else []):
            split_name = str(split_name)
            for case_id in case_ids or []:
                case_id = str(case_id)
                existing = seen_case_to_split.get(case_id)
                if existing is not None and existing != split_name:
                    _record_issue(
                        issues,
                        case_issues,
                        severity="error",
                        category="split_integrity",
                        issue_type="case_in_multiple_splits",
                        case_id=case_id,
                        message="The same case id appears in multiple splits.",
                        first_split=existing,
                        second_split=split_name,
                    )
                seen_case_to_split[case_id] = split_name

                fingerprint = primary_fingerprints.get(case_id, {})
                sha1 = fingerprint.get("sha1")
                if sha1:
                    existing_hash = seen_hash_to_split.get(str(sha1))
                    if existing_hash is not None and existing_hash[0] != split_name:
                        _record_issue(
                            issues,
                            case_issues,
                            severity="error",
                            category="split_integrity",
                            issue_type="content_leakage",
                            case_id=case_id,
                            message="The same image content appears across different splits.",
                            first_split=existing_hash[0],
                            first_case_id=existing_hash[1],
                            second_split=split_name,
                            sha1=sha1,
                        )
                    else:
                        seen_hash_to_split[str(sha1)] = (split_name, case_id)

        normalized_split_members: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for case_id, split_name in seen_case_to_split.items():
            normalized_split_members[normalize_case_token(case_id)][split_name].append(case_id)
        for normalized_token, split_members in normalized_split_members.items():
            if len(split_members) > 1:
                _record_issue(
                    issues,
                    case_issues,
                    severity="warning",
                    category="split_integrity",
                    issue_type="normalized_identifier_leakage",
                    message="Potential train/test leakage through renamed duplicates.",
                    normalized_case_id=normalized_token,
                    split_members={key: sorted(value) for key, value in split_members.items()},
                )

    # Finalize case statuses.
    ready_count = 0
    manual_review_count = 0
    blocked_count = 0
    for case_report in case_records:
        current_issues = case_issues.get(case_report["case_id"], [])
        case_report["issues"] = current_issues
        status = _classify_case_status(current_issues)
        case_report["status"] = status
        if status == "ready":
            ready_count += 1
        elif status == "manual_review":
            manual_review_count += 1
        else:
            blocked_count += 1

    category_summary: Dict[str, Dict[str, int]] = {}
    for issue in issues:
        bucket = category_summary.setdefault(issue["category"], {"error_count": 0, "warning_count": 0})
        if issue["severity"] == "error":
            bucket["error_count"] += 1
        elif issue["severity"] == "warning":
            bucket["warning_count"] += 1

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    if error_count:
        dataset_status = "fail"
    elif warning_count:
        dataset_status = "review"
    else:
        dataset_status = "pass"

    report = {
        "generated_at": utc_now_iso(),
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "dataset_mode": dataset_mode,
        "summary": {
            "dataset_status": dataset_status,
            "case_count": len(case_records),
            "ready_case_count": ready_count,
            "manual_review_case_count": manual_review_count,
            "blocked_case_count": blocked_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "modality_breakdown": dict(modality_counter),
            "numbering_schema_breakdown": dict(numbering_schema_counter),
            "duplicate_role_warnings": dict(duplicate_role_counter),
        },
        "checks": category_summary,
        "issues": issues,
        "cases": case_records,
        "split_summary": split_payload,
        "metadata": metadata or {},
    }
    return report


def build_markdown_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Dataset QC: {report['dataset_name']}",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Dataset mode: `{report['dataset_mode']}`",
        f"- Status: `{summary['dataset_status']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Ready: `{summary['ready_case_count']}`",
        f"- Manual review: `{summary['manual_review_case_count']}`",
        f"- Blocked: `{summary['blocked_case_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        "",
        "## Check Summary",
        "",
    ]
    if report["checks"]:
        for category, counts in sorted(report["checks"].items()):
            lines.append(
                f"- `{category}`: errors={counts.get('error_count', 0)}, warnings={counts.get('warning_count', 0)}"
            )
    else:
        lines.append("- No issues detected")

    top_errors = [issue for issue in report["issues"] if issue["severity"] == "error"][:20]
    top_warnings = [issue for issue in report["issues"] if issue["severity"] == "warning"][:20]

    lines.extend(["", "## Top Errors", ""])
    if top_errors:
        for issue in top_errors:
            case_prefix = f"[{issue['case_id']}] " if issue.get("case_id") else ""
            lines.append(f"- {case_prefix}{issue['type']}: {issue['message']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Warnings", ""])
    if top_warnings:
        for issue in top_warnings:
            case_prefix = f"[{issue['case_id']}] " if issue.get("case_id") else ""
            lines.append(f"- {case_prefix}{issue['type']}: {issue['message']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Blocked Cases", ""])
    blocked_cases = [case for case in report["cases"] if case["status"] == "blocked"][:30]
    if blocked_cases:
        for case in blocked_cases:
            lines.append(f"- `{case['case_id']}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Manual Review Cases", ""])
    manual_review_cases = [case for case in report["cases"] if case["status"] == "manual_review"][:30]
    if manual_review_cases:
        for case in manual_review_cases:
            lines.append(f"- `{case['case_id']}`")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_report_bundle(
    report: Dict[str, Any],
    *,
    report_key: str,
    report_root: Path | None = None,
) -> Dict[str, str]:
    output_json, output_md = build_default_report_paths(report_key, report_root=report_root)
    payload = dict(report)
    payload["report_key"] = sanitize_report_key(report_key)
    payload["report_json"] = str(output_json)
    payload["report_md"] = str(output_md)
    write_json(output_json, payload)
    output_md.write_text(build_markdown_report(payload), encoding="utf-8")
    return {
        "report_json": str(output_json),
        "report_md": str(output_md),
    }
