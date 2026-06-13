#!/usr/bin/env python3
"""Fast audit for unlabeled CBCT cohorts stored as one DICOM volume per file."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / "_lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import ensure_dir, write_json, write_jsonl  # noqa: E402

import numpy as np
import SimpleITK as sitk


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def partial_sha1(path: Path, chunk_bytes: int = 1024 * 1024) -> Optional[str]:
    try:
        size = path.stat().st_size
        hasher = hashlib.sha1()
        with path.open("rb") as handle:
            head = handle.read(chunk_bytes)
            hasher.update(head)
            if size > chunk_bytes:
                if size > 2 * chunk_bytes:
                    handle.seek(max(0, size - chunk_bytes))
                tail = handle.read(chunk_bytes)
                hasher.update(tail)
        hasher.update(str(size).encode("utf-8"))
        return hasher.hexdigest()
    except Exception:
        return None


def foreground_bbox(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if mask.size == 0 or not np.any(mask):
        return None
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    shape = np.array(mask.shape, dtype=int)
    touch_faces = 0
    touch_axes = 0
    for axis in range(mask.ndim):
        axis_touches = 0
        if mins[axis] <= 0:
            touch_faces += 1
            axis_touches += 1
        if maxs[axis] >= shape[axis] - 1:
            touch_faces += 1
            axis_touches += 1
        if axis_touches:
            touch_axes += 1
    return {
        "min_index": mins.tolist(),
        "max_index": maxs.tolist(),
        "extent_voxels": (maxs - mins + 1).tolist(),
        "touch_faces": int(touch_faces),
        "touch_axes": int(touch_axes),
    }


def quantiles(values: np.ndarray, probs: Sequence[float]) -> List[float]:
    if values.size == 0:
        return [0.0 for _ in probs]
    return [float(value) for value in np.quantile(values, probs)]


def summarize_sampled_image(sampled: np.ndarray) -> Dict[str, Any]:
    flat = sampled[np.isfinite(sampled)]
    if flat.size == 0:
        return {"sample_voxels": 0, "finite_voxels": 0, "empty_or_corrupt": True}
    q001, q01, q05, q50, q95, q99, q999 = quantiles(flat, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    dynamic_range = q99 - q01
    threshold = q05 + 0.15 * max(1e-6, (q95 - q05))
    foreground = sampled > threshold
    bbox = foreground_bbox(foreground)
    slice_means = [float(value) for value in sampled.mean(axis=(0, 1))] if sampled.ndim == 3 else []
    slice_deltas = np.abs(np.diff(slice_means)) if len(slice_means) > 1 else np.asarray([], dtype=np.float32)
    return {
        "sample_voxels": int(sampled.size),
        "finite_voxels": int(flat.size),
        "empty_or_corrupt": bool(dynamic_range <= 1e-6 and float(np.std(flat)) <= 1e-6),
        "sample_mean": float(np.mean(flat)),
        "sample_std": float(np.std(flat)),
        "dynamic_range_q99_q01": float(dynamic_range),
        "upper_tail_ratio": float((q999 - q99) / max(1e-6, dynamic_range)),
        "foreground_ratio": float(np.mean(foreground)),
        "foreground_bbox": bbox,
        "zero_fraction": float(np.mean(flat == 0)),
        "slice_mean_delta_p95": float(np.quantile(slice_deltas, 0.95)) if slice_deltas.size else 0.0,
    }


def robust_bounds(values: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 5:
        return None
    q1 = float(np.quantile(clean, 0.25))
    q3 = float(np.quantile(clean, 0.75))
    median = float(np.median(clean))
    iqr = q3 - q1
    if iqr <= 1e-6:
        spread = max(abs(median) * 0.1, 1e-3)
        return median - spread, median + spread, median
    return q1 - 3.0 * iqr, q3 + 3.0 * iqr, median


def add_finding(case: Dict[str, Any], *, domain: str, code: str, message: str, severity: str, confidence: str, impact: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    case.setdefault("findings", []).append({
        "domain": domain,
        "code": code,
        "message": message,
        "severity": severity,
        "confidence": confidence,
        "impact": impact,
        "status": status,
        "details": details or {},
    })


def classify_case(case: Dict[str, Any]) -> None:
    findings = case.get("findings") or []
    blocking_codes = {"unreadable_image", "invalid_spacing", "unexpected_dimensionality", "empty_or_corrupt_volume"}
    if any(f["code"] in blocking_codes for f in findings):
        case["status"] = "reject"
    elif findings:
        case["status"] = "needs_manual_review"
    else:
        case["status"] = "usable"


def read_case(path_str: str, max_voxel_sample: int) -> Dict[str, Any]:
    path = Path(path_str)
    case = {
        "case_id": path.stem,
        "source_image_path": str(path.resolve()),
        "image_sha1_partial": partial_sha1(path),
        "ingest_status": "complete",
        "image_format": ".dcm",
        "findings": [],
    }
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        if array.ndim == 3:
            z, y, x = array.shape
            total_voxels = max(1, z * y * x)
            stride = max(1, int(math.ceil((float(total_voxels) / float(max_voxel_sample)) ** (1.0 / 3.0))))
            sampled = array[::stride, ::stride, ::stride]
        else:
            sampled = array
        sampled = np.asarray(sampled, dtype=np.float32)
        orientation = None
        try:
            orientation = str(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection()))
        except Exception:
            orientation = None
        metadata = {str(k): reader.GetMetaData(k) for k in reader.GetMetaDataKeys()}
        case["image_summary"] = {
            "reader": "sitk_dicom",
            "shape": [int(v) for v in image.GetSize()],
            "spacing": [float(v) for v in image.GetSpacing()],
            "orientation": orientation,
            "modality": metadata.get("0008|0060"),
            "patient_id": metadata.get("0010|0020"),
            "study_uid": metadata.get("0020|000d"),
            "series_uid": metadata.get("0020|000e"),
            "study_date": metadata.get("0008|0020"),
            "study_time": metadata.get("0008|0030"),
            "manufacturer": metadata.get("0008|0070"),
            "manufacturer_model_name": metadata.get("0008|1090"),
            "series_description": metadata.get("0008|103e"),
            "image_quality_sample": summarize_sampled_image(sampled),
            "sample_hash": hashlib.sha1(sampled.tobytes()).hexdigest() if sampled.size else None,
        }
    except Exception as exc:
        case["ingest_status"] = "unreadable"
        add_finding(
            case,
            domain="intake",
            code="unreadable_image",
            message="Primary image could not be parsed as a CBCT volume.",
            severity="severe",
            confidence="high",
            impact="major",
            status="confirmed",
            details={"error": str(exc)},
        )
        classify_case(case)
        return case

    image = case["image_summary"]
    spacing = image.get("spacing") or []
    shape = image.get("shape") or []
    sample = image.get("image_quality_sample") or {}

    if not spacing or any((not math.isfinite(float(v))) or float(v) <= 0 for v in spacing):
        add_finding(case, domain="metadata", code="invalid_spacing", message="Voxel spacing is missing or non-positive.", severity="severe", confidence="high", impact="major", status="confirmed", details={"spacing": spacing})
    if len(shape) != 3:
        add_finding(case, domain="volume_consistency", code="unexpected_dimensionality", message="Volume dimensionality is unusual for a 3D CBCT case.", severity="severe", confidence="high", impact="major", status="confirmed", details={"shape": shape})
    missing_keys = [key for key in ("patient_id", "study_uid", "series_uid") if not str(image.get(key, "") or "").strip()]
    if missing_keys:
        add_finding(case, domain="metadata", code="missing_key_metadata", message="Critical DICOM identifier fields are missing.", severity="moderate", confidence="high", impact="moderate", status="confirmed", details={"missing_keys": missing_keys})
    modality = str(image.get("modality", "") or "").strip().upper()
    if not modality:
        add_finding(case, domain="metadata", code="missing_modality_metadata", message="DICOM modality tag is missing.", severity="moderate", confidence="high", impact="moderate", status="confirmed")
    elif modality != "CT":
        add_finding(case, domain="metadata", code="metadata_inconsistency", message="DICOM modality is not CT, which is inconsistent with a CBCT cohort.", severity="moderate", confidence="high", impact="moderate", status="confirmed", details={"modality": image.get("modality")})
    if sample.get("empty_or_corrupt"):
        add_finding(case, domain="volume_consistency", code="empty_or_corrupt_volume", message="Volume appears empty or corrupted in the sampled intensity profile.", severity="severe", confidence="high", impact="major", status="confirmed")
    bbox = sample.get("foreground_bbox") or {}
    if bbox and int(bbox.get("touch_faces", 0)) >= 5 and (float(sample.get("zero_fraction", 0.0)) > 0.05 or float(sample.get("foreground_ratio", 0.0)) < 0.08 or float(sample.get("foreground_ratio", 0.0)) > 0.95):
        add_finding(case, domain="volume_consistency", code="possible_truncation", message="Foreground touches multiple borders, suggesting cropped or truncated FOV.", severity="moderate", confidence="medium", impact="moderate", status="suspected")
    if sample.get("foreground_ratio", 0.0) < 0.002:
        add_finding(case, domain="volume_consistency", code="very_low_foreground_ratio", message="Foreground occupies a very small fraction of the sampled volume.", severity="moderate", confidence="medium", impact="moderate", status="suspected")
    if sample.get("upper_tail_ratio", 0.0) > 0.55:
        add_finding(case, domain="image_quality", code="possible_metal_artifact", message="High-intensity tail is unusually strong and may reflect metal artifact or beam hardening.", severity="moderate", confidence="low", impact="moderate", status="suspected")
    if sample.get("slice_mean_delta_p95", 0.0) > max(1.0, 0.3 * sample.get("dynamic_range_q99_q01", 0.0)):
        add_finding(case, domain="image_quality", code="possible_motion_or_slice_inconsistency", message="Slice-to-slice intensity shifts are elevated.", severity="moderate", confidence="low", impact="moderate", status="suspected")
    classify_case(case)
    return case


def render_markdown(summary: Dict[str, Any], problem_cases: Dict[str, List[str]]) -> str:
    lines = [
        "# Institutional Unlabeled CBCT Audit",
        "",
        "## Summary",
        "",
        f"- Imported: `{summary['imported']}`",
        f"- Parsed: `{summary['parsed']}`",
        f"- Usable after audit: `{summary['usable_after_audit']}`",
        f"- Rejected: `{summary['rejected']}`",
        f"- Metadata issues: `{summary['n_meta']}`",
        f"- Volumetric issues: `{summary['n_cons']}`",
        f"- Duplicate-ID / leakage-risk warnings: `{summary['n_dup']}`",
        f"- Artifact-heavy / image-quality flags: `{summary['n_art']}`",
        "",
        "## Cases With Issues",
        "",
    ]
    for key in ("metadata_issues", "volumetric_issues", "duplicate_risk", "artifact_or_quality"):
        values = problem_cases.get(key) or []
        joined = ", ".join(values[:40])
        if len(values) > 40:
            joined += ", ..."
        lines.append(f"- {key}: `{len(values)}` -> {joined if joined else 'None'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-voxel-sample", type=int, default=120000)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    paths = sorted(dataset_root.glob("*.dcm"))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        cases = list(pool.map(read_case, [str(p) for p in paths], [args.max_voxel_sample] * len(paths)))

    spacing_axes: Dict[int, List[float]] = defaultdict(list)
    shape_axes: Dict[int, List[float]] = defaultdict(list)
    fov_axes: Dict[int, List[float]] = defaultdict(list)
    range_values: List[float] = []
    std_values: List[float] = []
    file_hash_map: Dict[str, List[str]] = defaultdict(list)
    sample_hash_map: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    patient_id_map: Dict[str, List[str]] = defaultdict(list)
    study_uid_map: Dict[str, List[str]] = defaultdict(list)
    series_uid_map: Dict[str, List[str]] = defaultdict(list)

    for case in cases:
        image = case.get("image_summary") or {}
        sample = image.get("image_quality_sample") or {}
        shape = image.get("shape") or []
        spacing = image.get("spacing") or []
        for axis, value in enumerate(shape):
            shape_axes[axis].append(float(value))
        for axis, value in enumerate(spacing):
            spacing_axes[axis].append(float(value))
        if shape and spacing and len(shape) == len(spacing):
            extents = np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)
            for axis, value in enumerate(extents):
                fov_axes[axis].append(float(value))
        if sample:
            range_values.append(float(sample.get("dynamic_range_q99_q01", 0.0)))
            std_values.append(float(sample.get("sample_std", 0.0)))
        if case.get("image_sha1_partial"):
            file_hash_map[str(case["image_sha1_partial"])].append(case["case_id"])
        if image.get("sample_hash"):
            sample_hash_map[(str(image["sample_hash"]), tuple(shape), tuple(round(float(v), 6) for v in spacing))].append(case["case_id"])
        patient_id = str(image.get("patient_id", "") or "").strip()
        study_uid = str(image.get("study_uid", "") or "").strip()
        series_uid = str(image.get("series_uid", "") or "").strip()
        if patient_id:
            patient_id_map[patient_id].append(case["case_id"])
        if study_uid:
            study_uid_map[study_uid].append(case["case_id"])
        if series_uid:
            series_uid_map[series_uid].append(case["case_id"])

    spacing_bounds = {axis: robust_bounds(values) for axis, values in spacing_axes.items()}
    shape_bounds = {axis: robust_bounds(values) for axis, values in shape_axes.items()}
    fov_bounds = {axis: robust_bounds(values) for axis, values in fov_axes.items()}
    range_bounds = robust_bounds(range_values)
    std_bounds = robust_bounds(std_values)

    for case in cases:
        image = case.get("image_summary") or {}
        sample = image.get("image_quality_sample") or {}
        shape = image.get("shape") or []
        spacing = image.get("spacing") or []
        for axis, value in enumerate(spacing):
            bounds = spacing_bounds.get(axis)
            if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                add_finding(case, domain="metadata", code=f"spacing_outlier_axis_{axis}", message=f"Voxel spacing on axis {axis} is a cohort outlier.", severity="moderate", confidence="medium", impact="moderate", status="suspected", details={"value": value, "cohort_median": bounds[2]})
        for axis, value in enumerate(shape):
            bounds = shape_bounds.get(axis)
            if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                add_finding(case, domain="volume_consistency", code=f"shape_outlier_axis_{axis}", message=f"Matrix size on axis {axis} is an outlier relative to the cohort.", severity="moderate", confidence="medium", impact="moderate", status="suspected", details={"value": value, "cohort_median": bounds[2]})
        if shape and spacing and len(shape) == len(spacing):
            extents = np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)
            for axis, value in enumerate(extents):
                bounds = fov_bounds.get(axis)
                if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                    add_finding(case, domain="volume_consistency", code=f"fov_outlier_axis_{axis}", message=f"Physical field of view on axis {axis} is a cohort outlier.", severity="moderate", confidence="medium", impact="moderate", status="suspected", details={"value_mm": float(value), "cohort_median_mm": bounds[2]})
        if sample and range_bounds and not (range_bounds[0] <= float(sample.get("dynamic_range_q99_q01", 0.0)) <= range_bounds[1]):
            add_finding(case, domain="image_quality", code="dynamic_range_outlier", message="Sampled intensity dynamic range is a cohort outlier.", severity="moderate", confidence="medium", impact="moderate", status="suspected", details={"value": sample.get("dynamic_range_q99_q01"), "cohort_median": range_bounds[2]})
        if sample and std_bounds and not (std_bounds[0] <= float(sample.get("sample_std", 0.0)) <= std_bounds[1]):
            add_finding(case, domain="image_quality", code="noise_or_contrast_outlier", message="Sampled intensity spread is a cohort outlier.", severity="mild", confidence="medium", impact="moderate", status="suspected", details={"value": sample.get("sample_std"), "cohort_median": std_bounds[2]})

    for mapping_name, mapping in (("repeated_patient_identifier", patient_id_map), ("repeated_study_uid", study_uid_map), ("duplicate_series_uid", series_uid_map)):
        for key, case_ids in sorted(mapping.items()):
            unique_case_ids = sorted(set(case_ids))
            if len(unique_case_ids) > 1:
                for case in cases:
                    if case["case_id"] in unique_case_ids:
                        severity = "severe" if mapping_name == "duplicate_series_uid" else "moderate"
                        impact = "major" if mapping_name == "duplicate_series_uid" else "moderate"
                        status = "confirmed" if mapping_name != "repeated_patient_identifier" else "suspected"
                        add_finding(case, domain="duplicate_risk", code=mapping_name, message=f"{mapping_name} detected across multiple cases.", severity=severity, confidence="high" if mapping_name != "repeated_patient_identifier" else "medium", impact=impact, status=status, details={"key": key, "case_ids": unique_case_ids})

    exact_duplicate_sets = set()
    for digest, case_ids in sorted(file_hash_map.items()):
        unique_case_ids = sorted(set(case_ids))
        if len(unique_case_ids) > 1:
            exact_duplicate_sets.add(tuple(unique_case_ids))
            for case in cases:
                if case["case_id"] in unique_case_ids:
                    add_finding(case, domain="duplicate_risk", code="exact_duplicate_scan", message="Image file fingerprint matches another case, suggesting a duplicated scan.", severity="moderate", confidence="high", impact="moderate", status="confirmed", details={"duplicate_cases": unique_case_ids})

    for sample_key, case_ids in sorted(sample_hash_map.items(), key=lambda item: item[1]):
        unique_case_ids = tuple(sorted(set(case_ids)))
        if len(unique_case_ids) > 1 and unique_case_ids not in exact_duplicate_sets:
            for case in cases:
                if case["case_id"] in unique_case_ids:
                    add_finding(case, domain="duplicate_risk", code="suspected_near_duplicate_scan", message="Sampled image content matches another case after accounting for shape and spacing.", severity="moderate", confidence="medium", impact="moderate", status="suspected", details={"duplicate_cases": list(unique_case_ids)})

    for case in cases:
        classify_case(case)

    finding_counter = Counter(f["code"] for case in cases for f in case.get("findings") or [])
    parsed = sum(1 for case in cases if case.get("ingest_status") == "complete")
    rejected = sum(1 for case in cases if case.get("status") == "reject")
    problem_cases = {
        "metadata_issues": sorted({case["case_id"] for case in cases for f in case.get("findings") or [] if f["domain"] == "metadata"}),
        "volumetric_issues": sorted({case["case_id"] for case in cases for f in case.get("findings") or [] if f["domain"] == "volume_consistency"}),
        "duplicate_risk": sorted({case["case_id"] for case in cases for f in case.get("findings") or [] if f["domain"] == "duplicate_risk"}),
        "artifact_or_quality": sorted({case["case_id"] for case in cases for f in case.get("findings") or [] if f["domain"] == "image_quality"}),
    }
    summary = {
        "dataset_root": str(dataset_root),
        "generated_at": utc_now_iso(),
        "imported": len(paths),
        "parsed": parsed,
        "usable_after_audit": len(paths) - rejected,
        "rejected": rejected,
        "n_meta": len(problem_cases["metadata_issues"]),
        "n_cons": len(problem_cases["volumetric_issues"]),
        "n_dup": len(problem_cases["duplicate_risk"]),
        "n_art": len(problem_cases["artifact_or_quality"]),
        "status_counter": dict(Counter(case.get("status") for case in cases)),
        "finding_counter": dict(finding_counter.most_common(40)),
        "problem_cases": problem_cases,
    }

    markdown = render_markdown(summary, problem_cases)
    write_json(summary, output_root / "cohort_summary.json")
    write_json({"dataset_root": str(dataset_root), "output_root": str(output_root), "workers": args.workers, "max_voxel_sample": args.max_voxel_sample}, output_root / "manifest.json")
    write_json(problem_cases, output_root / "problem_cases.json")
    write_jsonl(cases, output_root / "cases.jsonl")
    (output_root / "cohort_summary.md").write_text(markdown, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
