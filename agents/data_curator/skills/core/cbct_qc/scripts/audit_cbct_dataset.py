#!/usr/bin/env python3
"""Audit CBCT datasets for metadata, geometry, label, quality, and duplicate risks."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CURRENT_FILE = Path(__file__).resolve()
LIB_DIR = CURRENT_FILE.parents[2] / "_lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from curation_core import case_stem, ensure_dir, normalized_suffix, read_json, write_json, write_jsonl  # noqa: E402

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("numpy is required for CBCT QC.") from exc

try:
    import nibabel as nib
except Exception:
    nib = None  # type: ignore[assignment]

try:
    import SimpleITK as sitk
except Exception:
    sitk = None  # type: ignore[assignment]

try:
    import pydicom
except Exception:
    pydicom = None  # type: ignore[assignment]


VOLUME_EXTS = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"}
IMAGE_DIR_NAMES = ("imagestr", "imagests", "images", "scans", "volumes")
LABEL_DIR_NAMES = ("labelstr", "labelsts", "labels", "masks", "segmentations", "annotations")
DEFAULT_MAX_VOXEL_SAMPLE = 450_000
DEFAULT_REPORT_ROOT = CURRENT_FILE.parents[4] / "reports" / "cbct_qc"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_case_id(raw_case_id: str) -> str:
    cleaned = str(raw_case_id).strip()
    if cleaned.endswith("_0000"):
        cleaned = cleaned[:-5]
    return cleaned


def duplicate_identifier_key(raw_case_id: str) -> str:
    cleaned = canonical_case_id(raw_case_id).strip().lower()
    cleaned = re.sub(r"(?:_copy|-copy|copy|duplicate|dup)$", "", cleaned)
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def is_volume_file(path: Path) -> bool:
    return normalized_suffix(path) in VOLUME_EXTS


def canonical_filename(case_id: str, path: Path, *, is_image: bool) -> str:
    suffix = normalized_suffix(path)
    if is_image:
        return f"{case_id}_0000{suffix}"
    return f"{case_id}{suffix}"


def strip_nnunet_channel_suffix(stem: str) -> str:
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def case_id_from_image_path(path: Path) -> str:
    return canonical_case_id(strip_nnunet_channel_suffix(case_stem(path)))


def case_id_from_label_path(path: Path) -> str:
    return canonical_case_id(case_stem(path))


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


def build_case_record(case_id: str) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "normalized_case_id": canonical_case_id(case_id),
        "image_paths": [],
        "label_paths": [],
        "split": None,
        "source_filenames": [],
        "source_structure": [],
    }


def discover_cases(dataset_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_root = dataset_root.resolve()
    cases: Dict[str, Dict[str, Any]] = {}
    discovery = {
        "image_dir_count": 0,
        "label_dir_count": 0,
        "fallback_walk_used": False,
        "orphan_labels": [],
    }

    def get_case(case_id: str) -> Dict[str, Any]:
        if case_id not in cases:
            cases[case_id] = build_case_record(case_id)
        return cases[case_id]

    image_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.lower() in IMAGE_DIR_NAMES] if dataset_root.is_dir() else []
    label_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.lower() in LABEL_DIR_NAMES] if dataset_root.is_dir() else []
    discovery["image_dir_count"] = len(image_dirs)
    discovery["label_dir_count"] = len(label_dirs)

    if image_dirs:
        for folder in image_dirs:
            split_name = folder.name
            for path in sorted(folder.rglob("*")):
                if not (path.is_file() and is_volume_file(path)):
                    continue
                case_id = case_id_from_image_path(path)
                case = get_case(case_id)
                case["image_paths"].append(path)
                case["source_filenames"].append(path.name)
                case["source_structure"].append({
                    "role": "image",
                    "path": safe_relpath(path, dataset_root),
                    "folder": split_name,
                })
                if case["split"] is None:
                    if split_name.lower() == "imagestr":
                        case["split"] = "train"
                    elif split_name.lower() == "imagests":
                        case["split"] = "test"
                    else:
                        case["split"] = split_name
    else:
        discovery["fallback_walk_used"] = True
        for path in sorted(dataset_root.rglob("*")):
            if not (path.is_file() and is_volume_file(path)):
                continue
            rel_parts = [part.lower() for part in path.relative_to(dataset_root).parts[:-1]]
            if any(part in LABEL_DIR_NAMES for part in rel_parts):
                continue
            case_id = case_id_from_image_path(path)
            case = get_case(case_id)
            case["image_paths"].append(path)
            case["source_filenames"].append(path.name)
            case["source_structure"].append({
                "role": "image",
                "path": safe_relpath(path, dataset_root),
                "folder": rel_parts[0] if rel_parts else ".",
            })

    for folder in label_dirs:
        split_name = folder.name
        for path in sorted(folder.rglob("*")):
            if not (path.is_file() and is_volume_file(path)):
                continue
            case_id = case_id_from_label_path(path)
            case = get_case(case_id)
            case["label_paths"].append(path)
            case["source_filenames"].append(path.name)
            case["source_structure"].append({
                "role": "label",
                "path": safe_relpath(path, dataset_root),
                "folder": split_name,
            })
            if not case["image_paths"]:
                discovery["orphan_labels"].append(case_id)

    ordered_cases = [cases[key] for key in sorted(cases)]
    for case in ordered_cases:
        case["source_filenames"] = sorted(set(case["source_filenames"]))
        case["source_structure"] = sorted(case["source_structure"], key=lambda item: (item["role"], item["path"]))
        case["image_paths"] = sorted(case["image_paths"], key=lambda path: path.name)
        case["label_paths"] = sorted(case["label_paths"], key=lambda path: path.name)
    return ordered_cases, discovery


def load_label_schema(dataset_root: Path) -> Dict[str, Any]:
    dataset_json = dataset_root / "dataset.json"
    if not dataset_json.is_file():
        return {
            "source": None,
            "allowed_values": None,
            "label_map": None,
            "channel_names": None,
        }
    payload = read_json(dataset_json)
    labels = payload.get("labels") or {}
    allowed_values = set()
    label_map = {}
    for key, value in labels.items():
        if isinstance(value, int):
            label_map[int(value)] = str(key)
            allowed_values.add(int(value))
        elif isinstance(key, str) and key.isdigit():
            allowed_values.add(int(key))
            label_map[int(key)] = str(value)
    if allowed_values:
        allowed_values.add(0)
    return {
        "source": str(dataset_json.resolve()),
        "allowed_values": sorted(allowed_values) if allowed_values else None,
        "label_map": label_map or None,
        "channel_names": payload.get("channel_names"),
        "dataset_json": payload,
    }


def normalize_shape_and_spacing(shape: Sequence[int], spacing: Sequence[float]) -> Tuple[List[int], List[float]]:
    dims = [int(value) for value in shape]
    zooms = [float(value) for value in spacing[: len(dims)]]
    if len(dims) == 4 and dims[-1] == 1:
        dims = dims[:-1]
        zooms = zooms[:3]
    return dims, zooms


def sample_volume_proxy(proxy: Any, shape: Sequence[int], max_voxels: int) -> np.ndarray:
    spatial_shape = tuple(int(value) for value in shape[:3])
    total_voxels = int(np.prod(spatial_shape))
    if total_voxels <= 0:
        return np.zeros((0,), dtype=np.float32)
    stride = max(1, int(math.ceil((float(total_voxels) / float(max_voxels)) ** (1.0 / 3.0))))
    slicer: List[Any] = [slice(None, None, stride), slice(None, None, stride), slice(None, None, stride)]
    if len(shape) == 4:
        slicer.append(0)
    sampled = np.asarray(proxy[tuple(slicer)], dtype=np.float32)
    if sampled.ndim == 4 and sampled.shape[-1] == 1:
        sampled = sampled[..., 0]
    return sampled


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
    extents = (maxs - mins + 1).tolist()
    return {
        "min_index": mins.tolist(),
        "max_index": maxs.tolist(),
        "extent_voxels": extents,
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
        return {
            "sample_voxels": 0,
            "finite_voxels": 0,
            "empty_or_corrupt": True,
        }
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
        "sample_min": float(np.min(flat)),
        "sample_max": float(np.max(flat)),
        "sample_mean": float(np.mean(flat)),
        "sample_std": float(np.std(flat)),
        "quantiles": {
            "q001": q001,
            "q01": q01,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "q99": q99,
            "q999": q999,
        },
        "dynamic_range_q99_q01": float(dynamic_range),
        "upper_tail_ratio": float((q999 - q99) / max(1e-6, dynamic_range)),
        "foreground_ratio": float(np.mean(foreground)),
        "foreground_bbox": bbox,
        "zero_fraction": float(np.mean(flat == 0)),
        "slice_mean_delta_mean": float(np.mean(slice_deltas)) if slice_deltas.size else 0.0,
        "slice_mean_delta_p95": float(np.quantile(slice_deltas, 0.95)) if slice_deltas.size else 0.0,
    }


def read_nifti_volume(path: Path, *, max_voxel_sample: int, with_sample: bool) -> Dict[str, Any]:
    if nib is None:
        raise RuntimeError("nibabel is unavailable for NIfTI reading.")
    image = nib.load(str(path))
    shape, spacing = normalize_shape_and_spacing(image.shape, image.header.get_zooms())
    affine = image.affine
    qform_affine, qform_code = image.get_qform(coded=True)
    sform_affine, sform_code = image.get_sform(coded=True)
    summary = {
        "reader": "nibabel",
        "path": str(path.resolve()),
        "format": normalized_suffix(path),
        "shape": shape,
        "spacing": spacing,
        "ndim": len(shape),
        "dtype": str(image.header.get_data_dtype()),
        "orientation": "".join(nib.aff2axcodes(affine)),
        "qform_code": int(qform_code),
        "sform_code": int(sform_code),
        "xyzt_units": int(image.header["xyzt_units"]),
        "descrip": image.header["descrip"].tobytes().decode("utf-8", errors="ignore").strip("\x00 ").strip(),
        "aux_file": image.header["aux_file"].tobytes().decode("utf-8", errors="ignore").strip("\x00 ").strip(),
    }
    if qform_affine is not None and int(qform_code) > 0:
        summary["qform_orientation"] = "".join(nib.aff2axcodes(qform_affine))
    if sform_affine is not None and int(sform_code) > 0:
        summary["sform_orientation"] = "".join(nib.aff2axcodes(sform_affine))
    if qform_affine is not None and sform_affine is not None and int(qform_code) > 0 and int(sform_code) > 0:
        summary["qform_sform_max_abs_diff"] = float(np.max(np.abs(np.asarray(qform_affine) - np.asarray(sform_affine))))
    if with_sample:
        sampled = sample_volume_proxy(image.dataobj, image.shape, max_voxel_sample)
        summary["image_quality_sample"] = summarize_sampled_image(sampled)
        summary["sample_hash"] = hashlib.sha1(sampled.tobytes()).hexdigest() if sampled.size else None
    return summary


def read_itk_volume(path: Path, *, max_voxel_sample: int, with_sample: bool) -> Dict[str, Any]:
    if sitk is None:
        raise RuntimeError("SimpleITK is unavailable for ITK volume reading.")
    image = sitk.ReadImage(str(path))
    size = [int(value) for value in image.GetSize()]
    spacing = [float(value) for value in image.GetSpacing()]
    direction = [float(value) for value in image.GetDirection()]
    summary = {
        "reader": "SimpleITK",
        "path": str(path.resolve()),
        "format": normalized_suffix(path),
        "shape": size,
        "spacing": spacing,
        "ndim": int(image.GetDimension()),
        "dtype": str(image.GetPixelIDTypeAsString()),
        "direction": direction,
        "origin": [float(value) for value in image.GetOrigin()],
        "metadata_keys": sorted(list(image.GetMetaDataKeys()))[:24],
    }
    if with_sample:
        array = sitk.GetArrayFromImage(image)
        if array.ndim == 3:
            z, y, x = array.shape
            total_voxels = max(1, z * y * x)
            stride = max(1, int(math.ceil((float(total_voxels) / float(max_voxel_sample)) ** (1.0 / 3.0))))
            sampled = array[::stride, ::stride, ::stride]
        else:
            sampled = array
        sampled = np.asarray(sampled, dtype=np.float32)
        summary["image_quality_sample"] = summarize_sampled_image(sampled)
        summary["sample_hash"] = hashlib.sha1(sampled.tobytes()).hexdigest() if sampled.size else None
    return summary


def read_dicom_with_sitk(path: Path, *, max_voxel_sample: int, with_sample: bool) -> Dict[str, Any]:
    if sitk is None:
        raise RuntimeError("SimpleITK is unavailable for DICOM reading.")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    image = sitk.ReadImage(str(path))
    size = [int(value) for value in image.GetSize()]
    spacing = [float(value) for value in image.GetSpacing()]
    direction = [float(value) for value in image.GetDirection()]
    metadata = {str(key): reader.GetMetaData(key) for key in reader.GetMetaDataKeys()}
    orientation = None
    try:
        orientation = str(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection()))
    except Exception:
        orientation = None
    summary = {
        "reader": "sitk_dicom",
        "path": str(path.resolve()),
        "format": normalized_suffix(path),
        "shape": size,
        "spacing": spacing,
        "ndim": int(image.GetDimension()),
        "dtype": str(image.GetPixelIDTypeAsString()),
        "direction": direction,
        "origin": [float(value) for value in image.GetOrigin()],
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
        "metadata_keys": sorted(list(metadata.keys()))[:24],
    }
    if with_sample:
        array = sitk.GetArrayFromImage(image)
        if array.ndim == 3:
            z, y, x = array.shape
            total_voxels = max(1, z * y * x)
            stride = max(1, int(math.ceil((float(total_voxels) / float(max_voxel_sample)) ** (1.0 / 3.0))))
            sampled = array[::stride, ::stride, ::stride]
        else:
            sampled = array
        sampled = np.asarray(sampled, dtype=np.float32)
        summary["image_quality_sample"] = summarize_sampled_image(sampled)
        summary["sample_hash"] = hashlib.sha1(sampled.tobytes()).hexdigest() if sampled.size else None
    return summary


def read_dicom_single(path: Path, *, max_voxel_sample: int, with_sample: bool) -> Dict[str, Any]:
    if pydicom is None:
        return read_dicom_with_sitk(path, max_voxel_sample=max_voxel_sample, with_sample=with_sample)
    ds = pydicom.dcmread(str(path), force=True)
    shape: List[int] = []
    if getattr(ds, "NumberOfFrames", None) is not None:
        shape.append(int(ds.NumberOfFrames))
    if getattr(ds, "Rows", None) is not None:
        shape.append(int(ds.Rows))
    if getattr(ds, "Columns", None) is not None:
        shape.append(int(ds.Columns))
    spacing = []
    if getattr(ds, "SliceThickness", None) is not None:
        spacing.append(float(ds.SliceThickness))
    if getattr(ds, "PixelSpacing", None) is not None:
        spacing.extend(float(value) for value in ds.PixelSpacing)
    summary = {
        "reader": "pydicom",
        "path": str(path.resolve()),
        "format": normalized_suffix(path),
        "shape": shape,
        "spacing": spacing,
        "ndim": len(shape),
        "dtype": str(getattr(ds, "BitsAllocated", "")),
        "modality": getattr(ds, "Modality", None),
        "patient_id": getattr(ds, "PatientID", None),
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "series_uid": getattr(ds, "SeriesInstanceUID", None),
        "study_date": getattr(ds, "StudyDate", None),
        "study_time": getattr(ds, "StudyTime", None),
        "manufacturer": getattr(ds, "Manufacturer", None),
        "manufacturer_model_name": getattr(ds, "ManufacturerModelName", None),
        "series_description": getattr(ds, "SeriesDescription", None),
    }
    if with_sample and getattr(ds, "pixel_array", None) is not None:
        sampled = np.asarray(ds.pixel_array, dtype=np.float32)
        if sampled.ndim == 3:
            z, y, x = sampled.shape
            total_voxels = max(1, z * y * x)
            stride = max(1, int(math.ceil((float(total_voxels) / float(max_voxel_sample)) ** (1.0 / 3.0))))
            sampled = sampled[::stride, ::stride, ::stride]
        summary["image_quality_sample"] = summarize_sampled_image(sampled)
        summary["sample_hash"] = hashlib.sha1(sampled.tobytes()).hexdigest() if sampled.size else None
    return summary


def read_volume(path: Path, *, max_voxel_sample: int, with_sample: bool) -> Dict[str, Any]:
    suffix = normalized_suffix(path)
    if suffix in {".nii", ".nii.gz"}:
        return read_nifti_volume(path, max_voxel_sample=max_voxel_sample, with_sample=with_sample)
    if suffix in {".mha", ".mhd", ".nrrd"}:
        return read_itk_volume(path, max_voxel_sample=max_voxel_sample, with_sample=with_sample)
    if suffix == ".dcm":
        return read_dicom_single(path, max_voxel_sample=max_voxel_sample, with_sample=with_sample)
    raise RuntimeError(f"Unsupported CBCT volume format: {path}")


def summarize_label(path: Path, *, max_voxel_sample: int) -> Dict[str, Any]:
    summary = read_volume(path, max_voxel_sample=max_voxel_sample, with_sample=False)
    suffix = normalized_suffix(path)
    if suffix in {".nii", ".nii.gz"}:
        if nib is None:
            raise RuntimeError("nibabel is unavailable for label reading.")
        label = nib.load(str(path))
        data = np.asarray(label.dataobj)
    elif suffix in {".mha", ".mhd", ".nrrd"}:
        if sitk is None:
            raise RuntimeError("SimpleITK is unavailable for label reading.")
        data = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    elif suffix == ".dcm":
        if pydicom is not None:
            ds = pydicom.dcmread(str(path), force=True)
            data = np.asarray(ds.pixel_array)
        elif sitk is not None:
            data = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        else:
            raise RuntimeError("Neither pydicom nor SimpleITK is available for DICOM label reading.")
    else:
        raise RuntimeError(f"Unsupported label format: {path}")

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    unique_values = np.unique(data)
    nonzero_voxels = int(np.count_nonzero(data))
    bbox = foreground_bbox(data > 0)
    rounded = np.round(unique_values)
    integer_like = bool(np.all(np.isfinite(unique_values)) and np.allclose(unique_values, rounded))
    summary.update({
        "unique_values": [int(value) for value in unique_values[:256]] if integer_like else [float(value) for value in unique_values[:256]],
        "unique_value_count": int(unique_values.size),
        "nonzero_voxels": nonzero_voxels,
        "foreground_ratio": float(nonzero_voxels / float(data.size)) if data.size else 0.0,
        "empty_mask": bool(nonzero_voxels == 0),
        "integer_like": integer_like,
        "foreground_bbox": bbox,
    })
    return summary


def robust_bounds(values: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
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


def add_finding(
    case: Dict[str, Any],
    *,
    domain: str,
    code: str,
    message: str,
    severity: str,
    confidence: str,
    impact: str,
    status: str,
    auto_correctable: bool = False,
    suggested_action: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    finding = {
        "domain": domain,
        "code": code,
        "message": message,
        "severity": severity,
        "confidence": confidence,
        "impact": impact,
        "status": status,
        "auto_correctable": auto_correctable,
    }
    if suggested_action:
        finding["suggested_action"] = suggested_action
    if details:
        finding["details"] = details
    case.setdefault("findings", []).append(finding)


def classify_case(case: Dict[str, Any]) -> None:
    findings = case.get("findings") or []
    if not findings:
        case["status"] = "usable"
        case["status_rationale"] = "No blocking issues were found in the available image, metadata, or label checks."
        return

    severe_blockers = [
        finding for finding in findings
        if finding["severity"] == "severe"
        and finding["impact"] == "major"
        and finding["status"] == "confirmed"
        and not finding.get("auto_correctable", False)
    ]
    manual_review = [
        finding for finding in findings
        if finding["status"] == "suspected" or finding["severity"] in {"moderate", "severe"}
    ]
    if severe_blockers:
        case["status"] = "reject"
    elif manual_review:
        case["status"] = "needs_manual_review"
    elif findings:
        case["status"] = "usable_with_warnings"
    else:
        case["status"] = "usable"

    top_messages = [finding["message"] for finding in findings[:2]]
    case["status_rationale"] = " ".join(top_messages) if top_messages else "QC completed."


def apply_case_level_checks(case: Dict[str, Any], label_schema: Dict[str, Any]) -> None:
    image_paths = case["image_paths"]
    label_paths = case["label_paths"]
    case["findings"] = []
    case["auto_corrections"] = []
    case["manual_review_reasons"] = []
    case["recommended_exclusion_reasons"] = []

    if not image_paths:
        add_finding(
            case,
            domain="intake",
            code="missing_image",
            message="Case has no readable image reference and cannot be audited as a CBCT volume.",
            severity="severe",
            confidence="high",
            impact="major",
            status="confirmed",
            suggested_action="Exclude the case until the source image is restored.",
        )
        classify_case(case)
        return

    if len(image_paths) > 1:
        add_finding(
            case,
            domain="intake",
            code="multiple_image_files",
            message="Case has multiple image files and needs manual review to determine the canonical CBCT volume.",
            severity="moderate",
            confidence="high",
            impact="moderate",
            status="confirmed",
            suggested_action="Select or normalize a single primary CBCT volume for this case.",
            details={"image_paths": [str(path) for path in image_paths]},
        )

    image_path = image_paths[0]
    case["canonical_image_name"] = canonical_filename(case["normalized_case_id"], image_path, is_image=True)
    case["image_sha1_partial"] = partial_sha1(image_path)
    case["ingest_status"] = "complete"
    case["image_format"] = normalized_suffix(image_path)
    case["source_image_path"] = str(image_path.resolve())

    try:
        image_summary = read_volume(image_path, max_voxel_sample=args.max_voxel_sample, with_sample=True)
        case["image_summary"] = image_summary
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
            suggested_action="Exclude the case or re-export the image volume.",
            details={"error": str(exc), "path": str(image_path)},
        )
        classify_case(case)
        return

    image_summary = case["image_summary"]
    spacing = image_summary.get("spacing") or []
    shape = image_summary.get("shape") or []
    sample = image_summary.get("image_quality_sample") or {}

    if not spacing or any((not math.isfinite(float(value))) or float(value) <= 0 for value in spacing):
        add_finding(
            case,
            domain="metadata",
            code="invalid_spacing",
            message="Voxel spacing is missing or non-positive.",
            severity="severe",
            confidence="high",
            impact="major",
            status="confirmed",
            suggested_action="Repair spatial metadata before downstream use.",
            details={"spacing": spacing},
        )

    if len(shape) not in {3}:
        add_finding(
            case,
            domain="volume_consistency",
            code="unexpected_dimensionality",
            message="Volume dimensionality is unusual for a 3D CBCT case.",
            severity="moderate",
            confidence="high",
            impact="moderate",
            status="confirmed",
            suggested_action="Verify whether this file is a full CBCT reconstruction or an auxiliary export.",
            details={"shape": shape},
        )

    if image_summary.get("reader") == "nibabel":
        if int(image_summary.get("qform_code", 0)) == 0 and int(image_summary.get("sform_code", 0)) == 0:
            add_finding(
                case,
                domain="metadata",
                code="missing_spatial_transform_codes",
                message="NIfTI header lacks both qform and sform codes, so orientation metadata may be unreliable.",
                severity="moderate",
                confidence="medium",
                impact="moderate",
                status="confirmed",
                suggested_action="Verify orientation against the upstream export before training or registration.",
            )
        qform_orientation = image_summary.get("qform_orientation")
        sform_orientation = image_summary.get("sform_orientation")
        qform_sform_max_abs_diff = float(image_summary.get("qform_sform_max_abs_diff", 0.0) or 0.0)
        if (
            int(image_summary.get("qform_code", 0)) > 0
            and int(image_summary.get("sform_code", 0)) > 0
            and (
                (qform_orientation and sform_orientation and qform_orientation != sform_orientation)
                or qform_sform_max_abs_diff > 1e-3
            )
        ):
            add_finding(
                case,
                domain="metadata",
                code="metadata_inconsistency",
                message="NIfTI qform and sform headers disagree, so spatial metadata may be inconsistent.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Resolve header inconsistency before registration-sensitive downstream use.",
                details={
                    "qform_orientation": qform_orientation,
                    "sform_orientation": sform_orientation,
                    "qform_sform_max_abs_diff": qform_sform_max_abs_diff,
                    "subtype": "qform_sform_mismatch",
                },
            )

    if image_summary.get("reader") in {"pydicom", "sitk_dicom"}:
        missing_keys = [
            key
            for key in ("patient_id", "study_uid", "series_uid")
            if not str(image_summary.get(key, "") or "").strip()
        ]
        if missing_keys:
            add_finding(
                case,
                domain="metadata",
                code="missing_key_metadata",
                message="Critical DICOM identifier fields are missing.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Verify the export and restore identifier metadata needed for cohort governance.",
                details={"missing_keys": missing_keys},
            )
        modality = str(image_summary.get("modality", "") or "").strip().upper()
        if not modality:
            add_finding(
                case,
                domain="metadata",
                code="missing_modality_metadata",
                message="DICOM modality tag is missing.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Restore modality metadata before downstream governance or reporting.",
            )
        elif modality != "CT":
            add_finding(
                case,
                domain="metadata",
                code="metadata_inconsistency",
                message="DICOM modality is not CT, which is inconsistent with a CBCT cohort.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Verify that this export belongs to the intended CBCT cohort.",
                details={"modality": image_summary.get("modality"), "subtype": "unexpected_dicom_modality"},
            )

    if sample.get("empty_or_corrupt"):
        add_finding(
            case,
            domain="volume_consistency",
            code="empty_or_corrupt_volume",
            message="Volume appears empty, near-empty, or numerically corrupted in the sampled intensity profile.",
            severity="severe",
            confidence="high",
            impact="major",
            status="confirmed",
            suggested_action="Exclude the case until the source volume is repaired.",
        )

    bbox = (sample.get("foreground_bbox") or {})
    if (
        bbox
        and int(bbox.get("touch_faces", 0)) >= 5
        and (
            float(sample.get("zero_fraction", 0.0)) > 0.05
            or float(sample.get("foreground_ratio", 0.0)) < 0.08
            or float(sample.get("foreground_ratio", 0.0)) > 0.95
        )
    ):
        add_finding(
            case,
            domain="volume_consistency",
            code="possible_truncation",
            message="Foreground touches multiple volume borders, suggesting a cropped or truncated head field of view.",
            severity="moderate",
            confidence="medium",
            impact="moderate",
            status="suspected",
            suggested_action="Manually review whether craniofacial coverage is sufficient for the intended task.",
            details={"foreground_bbox": bbox},
        )

    if sample.get("foreground_ratio", 0.0) < 0.002:
        add_finding(
            case,
            domain="volume_consistency",
            code="very_low_foreground_ratio",
            message="Foreground occupies a very small fraction of the sampled volume, which may indicate an incomplete or mostly empty scan.",
            severity="moderate",
            confidence="medium",
            impact="moderate",
            status="suspected",
            suggested_action="Review the case for empty coverage, strong padding, or export failure.",
            details={"foreground_ratio": sample.get("foreground_ratio")},
        )

    if sample.get("upper_tail_ratio", 0.0) > 0.55:
        add_finding(
            case,
            domain="image_quality",
            code="possible_metal_artifact",
            message="High-intensity tail is unusually strong, which may reflect metal-related artifact or beam hardening.",
            severity="moderate",
            confidence="low",
            impact="moderate",
            status="suspected",
            suggested_action="Review the scan for restorations, implants, or streak artifacts before direct downstream use.",
            details={"upper_tail_ratio": sample.get("upper_tail_ratio")},
        )

    if sample.get("slice_mean_delta_p95", 0.0) > max(1.0, 0.3 * sample.get("dynamic_range_q99_q01", 0.0)):
        add_finding(
            case,
            domain="image_quality",
            code="possible_motion_or_slice_inconsistency",
            message="Slice-to-slice intensity shifts are elevated, which may indicate motion or reconstruction inconsistency.",
            severity="moderate",
            confidence="low",
            impact="moderate",
            status="suspected",
            suggested_action="Inspect the reconstruction visually before relying on it for analysis.",
            details={"slice_mean_delta_p95": sample.get("slice_mean_delta_p95")},
        )

    label_status = "missing"
    case["label_status"] = label_status
    if not label_paths and args.label_policy != "ignore":
        add_finding(
            case,
            domain="label_awareness",
            code="missing_annotation_volume",
            message="Case has no annotation volume.",
            severity="severe" if args.label_policy == "required" else "mild",
            confidence="high",
            impact="major" if args.label_policy == "required" else "moderate",
            status="confirmed",
            suggested_action=(
                "Exclude the case from supervised segmentation until a label volume is available."
                if args.label_policy == "required"
                else "Allowed for unlabeled or pretraining queues, but not for supervised segmentation."
            ),
            details={"label_policy": args.label_policy},
        )
    if label_paths:
        if len(label_paths) > 1:
            add_finding(
                case,
                domain="label_awareness",
                code="multiple_label_files",
                message="Case has multiple label files and needs manual review to determine the canonical segmentation volume.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Select or normalize a single label file for this case.",
                details={"label_paths": [str(path) for path in label_paths]},
            )
        label_path = label_paths[0]
        case["source_label_path"] = str(label_path.resolve())
        case["canonical_label_name"] = canonical_filename(case["normalized_case_id"], label_path, is_image=False)
        case["label_sha1_partial"] = partial_sha1(label_path)
        try:
            label_summary = summarize_label(label_path, max_voxel_sample=args.max_voxel_sample)
            case["label_summary"] = label_summary
            case["label_status"] = "present"
        except Exception as exc:
            case["label_status"] = "unreadable"
            add_finding(
                case,
                domain="label_awareness",
                code="unreadable_label",
                message="Label volume exists but could not be parsed.",
                severity="severe",
                confidence="high",
                impact="major",
                status="confirmed",
                suggested_action="Exclude the label from supervised use or re-export it.",
                details={"error": str(exc), "path": str(label_path)},
            )
            classify_case(case)
            return

        label_summary = case["label_summary"]
        if (image_summary.get("shape") or []) != (label_summary.get("shape") or []):
            add_finding(
                case,
                domain="label_awareness",
                code="image_label_shape_mismatch",
                message="Image and label volumes do not share the same shape.",
                severity="severe",
                confidence="high",
                impact="major",
                status="confirmed",
                suggested_action="Do not use the label until geometry is corrected.",
                details={
                    "image_shape": image_summary.get("shape"),
                    "label_shape": label_summary.get("shape"),
                },
            )

        if tuple(round(float(value), 6) for value in (image_summary.get("spacing") or [])) != tuple(round(float(value), 6) for value in (label_summary.get("spacing") or [])):
            add_finding(
                case,
                domain="label_awareness",
                code="image_label_spacing_mismatch",
                message="Image and label spacing metadata do not match.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Verify that label geometry was exported from the same reconstruction.",
                details={
                    "image_spacing": image_summary.get("spacing"),
                    "label_spacing": label_summary.get("spacing"),
                },
            )
            add_finding(
                case,
                domain="metadata",
                code="metadata_inconsistency",
                message="Image and label headers disagree on voxel spacing metadata.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Reconcile image and label spacing metadata before supervised use.",
                details={
                    "image_spacing": image_summary.get("spacing"),
                    "label_spacing": label_summary.get("spacing"),
                    "subtype": "image_label_spacing_mismatch",
                },
            )

        if image_summary.get("orientation") and label_summary.get("orientation") and image_summary.get("orientation") != label_summary.get("orientation"):
            add_finding(
                case,
                domain="label_awareness",
                code="image_label_orientation_mismatch",
                message="Image and label orientations differ.",
                severity="moderate",
                confidence="medium",
                impact="moderate",
                status="confirmed",
                suggested_action="Check whether one of the files was reoriented without updating its counterpart.",
                details={
                    "image_orientation": image_summary.get("orientation"),
                    "label_orientation": label_summary.get("orientation"),
                },
            )
            add_finding(
                case,
                domain="metadata",
                code="metadata_inconsistency",
                message="Image and label headers disagree on orientation metadata.",
                severity="moderate",
                confidence="high",
                impact="moderate",
                status="confirmed",
                suggested_action="Reconcile image and label orientation metadata before supervised use.",
                details={
                    "image_orientation": image_summary.get("orientation"),
                    "label_orientation": label_summary.get("orientation"),
                    "subtype": "image_label_orientation_mismatch",
                },
            )

        if label_summary.get("empty_mask"):
            add_finding(
                case,
                domain="label_awareness",
                code="empty_label",
                message="Label volume is empty.",
                severity="severe",
                confidence="high",
                impact="major",
                status="confirmed",
                suggested_action="Exclude the label or verify that the export was not corrupted.",
            )

        if label_summary.get("foreground_ratio", 0.0) < 0.0001 and not label_summary.get("empty_mask"):
            add_finding(
                case,
                domain="label_awareness",
                code="near_empty_label",
                message="Label foreground is extremely sparse and may need review.",
                severity="moderate",
                confidence="medium",
                impact="moderate",
                status="suspected",
                suggested_action="Inspect whether the target anatomy is plausibly represented.",
                details={"foreground_ratio": label_summary.get("foreground_ratio")},
            )

        if not label_summary.get("integer_like", True):
            add_finding(
                case,
                domain="label_awareness",
                code="non_integer_label_values",
                message="Label values are not integer-like, which is inconsistent with segmentation masks.",
                severity="severe",
                confidence="high",
                impact="major",
                status="confirmed",
                suggested_action="Map the label to an integer schema before supervised use.",
            )

        allowed_values = label_schema.get("allowed_values")
        if allowed_values is not None:
            observed = set(int(value) for value in (label_summary.get("unique_values") or []))
            invalid_values = sorted(observed - set(allowed_values))
            if invalid_values:
                add_finding(
                    case,
                    domain="label_awareness",
                    code="invalid_label_values",
                    message="Label contains values outside the declared dataset schema.",
                    severity="severe",
                    confidence="high",
                    impact="major",
                    status="confirmed",
                    auto_correctable=True,
                    suggested_action="Map or remove invalid label values before supervised training.",
                    details={"invalid_values": invalid_values},
                )

    if case["normalized_case_id"] != case["case_id"]:
        case["auto_corrections"].append({
            "type": "canonical_case_id",
            "from": case["case_id"],
            "to": case["normalized_case_id"],
            "applied": False,
            "safe_to_apply": True,
        })

    classify_case(case)


def apply_cohort_level_checks(cases: List[Dict[str, Any]], split_lookup: Dict[str, str]) -> Dict[str, Any]:
    cohort_findings: List[Dict[str, Any]] = []
    spacing_axes: Dict[int, List[float]] = defaultdict(list)
    shape_axes: Dict[int, List[float]] = defaultdict(list)
    range_values: List[float] = []
    std_values: List[float] = []
    fov_axes: Dict[int, List[float]] = defaultdict(list)
    file_hash_map: Dict[str, List[str]] = defaultdict(list)
    sample_hash_map: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    id_map: Dict[str, List[str]] = defaultdict(list)
    patient_id_map: Dict[str, List[str]] = defaultdict(list)
    study_uid_map: Dict[str, List[str]] = defaultdict(list)
    series_uid_map: Dict[str, List[str]] = defaultdict(list)
    manufacturer_counter: Counter[str] = Counter()
    orientation_counter: Counter[str] = Counter()

    for case in cases:
        id_map[case["normalized_case_id"]].append(case["case_id"])
        image = case.get("image_summary") or {}
        sample = image.get("image_quality_sample") or {}
        shape = image.get("shape") or []
        spacing = image.get("spacing") or []
        for axis, value in enumerate(shape):
            shape_axes[axis].append(float(value))
        for axis, value in enumerate(spacing):
            spacing_axes[axis].append(float(value))
        if shape and spacing and len(shape) == len(spacing):
            for axis, value in enumerate(np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)):
                fov_axes[axis].append(float(value))
        if sample:
            range_values.append(float(sample.get("dynamic_range_q99_q01", 0.0)))
            std_values.append(float(sample.get("sample_std", 0.0)))
        if image.get("sample_hash"):
            sample_hash_map[(str(image["sample_hash"]), tuple(image.get("shape") or []), tuple(round(float(value), 6) for value in (image.get("spacing") or [])))].append(case["case_id"])
        if case.get("image_sha1_partial"):
            file_hash_map[str(case["image_sha1_partial"])].append(case["case_id"])
        if image.get("orientation"):
            orientation_counter[str(image["orientation"])] += 1
        manufacturer = image.get("manufacturer") or image.get("series_description")
        if manufacturer:
            manufacturer_counter[str(manufacturer)] += 1
        patient_id = str(image.get("patient_id", "") or "").strip()
        if patient_id:
            patient_id_map[patient_id].append(case["case_id"])
        study_uid = str(image.get("study_uid", "") or "").strip()
        if study_uid:
            study_uid_map[study_uid].append(case["case_id"])
        series_uid = str(image.get("series_uid", "") or "").strip()
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
                add_finding(
                    case,
                    domain="metadata",
                    code=f"spacing_outlier_axis_{axis}",
                    message=f"Voxel spacing on axis {axis} is an outlier relative to the cohort.",
                    severity="moderate",
                    confidence="medium",
                    impact="moderate",
                    status="suspected",
                    suggested_action="Review whether this scan belongs to a distinct acquisition subgroup.",
                    details={"value": value, "cohort_median": bounds[2]},
                )

        for axis, value in enumerate(shape):
            bounds = shape_bounds.get(axis)
            if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                add_finding(
                    case,
                    domain="volume_consistency",
                    code=f"shape_outlier_axis_{axis}",
                    message=f"Matrix size on axis {axis} is an outlier relative to the cohort.",
                    severity="moderate",
                    confidence="medium",
                    impact="moderate",
                    status="suspected",
                    suggested_action="Check whether this case is cropped, padded, or reconstructed differently.",
                    details={"value": value, "cohort_median": bounds[2]},
                )

        if shape and spacing and len(shape) == len(spacing):
            extents = np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)
            for axis, value in enumerate(extents):
                bounds = fov_bounds.get(axis)
                if bounds and not (bounds[0] <= float(value) <= bounds[1]):
                    add_finding(
                        case,
                        domain="volume_consistency",
                        code=f"fov_outlier_axis_{axis}",
                        message=f"Physical field of view on axis {axis} is a cohort outlier.",
                        severity="moderate",
                        confidence="medium",
                        impact="moderate",
                        status="suspected",
                        suggested_action="Review whether anatomy coverage is incomplete for this task.",
                        details={"value_mm": float(value), "cohort_median_mm": bounds[2]},
                    )

        if sample and range_bounds and not (range_bounds[0] <= float(sample.get("dynamic_range_q99_q01", 0.0)) <= range_bounds[1]):
            low_side = float(sample.get("dynamic_range_q99_q01", 0.0)) < range_bounds[2]
            add_finding(
                case,
                domain="image_quality",
                code="dynamic_range_outlier",
                message="Sampled intensity dynamic range is a cohort outlier and may reflect low contrast or an unusually wide reconstruction range.",
                severity="moderate",
                confidence="medium",
                impact="moderate",
                status="suspected",
                suggested_action="Inspect the intensity profile before direct downstream use.",
                details={"direction": "low" if low_side else "high", "value": sample.get("dynamic_range_q99_q01"), "cohort_median": range_bounds[2]},
            )

        if sample and std_bounds and not (std_bounds[0] <= float(sample.get("sample_std", 0.0)) <= std_bounds[1]):
            add_finding(
                case,
                domain="image_quality",
                code="noise_or_contrast_outlier",
                message="Sampled intensity spread is a cohort outlier and may reflect unusual noise or contrast.",
                severity="mild",
                confidence="medium",
                impact="moderate",
                status="suspected",
                suggested_action="Review this case if it is selected for sensitive downstream evaluation.",
                details={"value": sample.get("sample_std"), "cohort_median": std_bounds[2]},
            )

    for normalized_case_id, raw_ids in sorted(id_map.items()):
        if len(raw_ids) > 1:
            cohort_findings.append({
                "type": "duplicate_case_id",
                "normalized_case_id": normalized_case_id,
                "case_ids": raw_ids,
                "severity": "moderate",
                "confidence": "high",
                "impact": "moderate",
            })
            for case in cases:
                if case["case_id"] in raw_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="duplicate_case_identifier",
                        message="Normalized case identifier collides with another case in the cohort.",
                        severity="moderate",
                        confidence="high",
                        impact="moderate",
                        status="confirmed",
                        suggested_action="Resolve naming collisions before split assignment or training.",
                        details={"case_ids": raw_ids},
                    )
            split_names = sorted({
                split_lookup.get(case_id) or next((case.get("split") for case in cases if case["case_id"] == case_id), None)
                for case_id in raw_ids
                if split_lookup.get(case_id) or next((case.get("split") for case in cases if case["case_id"] == case_id), None)
            })
            if len(split_names) > 1:
                cohort_findings.append({
                    "type": "split_leakage_duplicate_identifier",
                    "normalized_case_id": normalized_case_id,
                    "case_ids": raw_ids,
                    "splits": split_names,
                    "severity": "severe",
                    "confidence": "high",
                    "impact": "major",
                })
                for case in cases:
                    if case["case_id"] in raw_ids:
                        add_finding(
                            case,
                            domain="duplicate_risk",
                            code="split_leakage",
                            message="Duplicate normalized identifiers appear across different data splits.",
                            severity="severe",
                            confidence="high",
                            impact="major",
                            status="confirmed",
                            suggested_action="Remove identifier collisions across train/val/test before evaluation.",
                            details={"case_ids": raw_ids, "splits": split_names, "subtype": "duplicate_identifier"},
                        )

    duplicate_id_map: Dict[str, List[str]] = defaultdict(list)
    for case in cases:
        duplicate_id_map[duplicate_identifier_key(case["case_id"])].append(case["case_id"])

    for duplicate_key, raw_ids in sorted(duplicate_id_map.items()):
        if len(raw_ids) > 1:
            cohort_findings.append({
                "type": "duplicate_identifier_key",
                "duplicate_key": duplicate_key,
                "case_ids": raw_ids,
                "severity": "moderate",
                "confidence": "high",
                "impact": "moderate",
            })
            for case in cases:
                if case["case_id"] in raw_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="duplicate_case_identifier",
                        message="Case identifier collides with another case after normalization.",
                        severity="moderate",
                        confidence="high",
                        impact="moderate",
                        status="confirmed",
                        suggested_action="Resolve identifier variants before split assignment or training.",
                        details={"case_ids": raw_ids, "duplicate_key": duplicate_key},
                    )
            split_names = sorted({
                split_lookup.get(case_id) or next((case.get("split") for case in cases if case["case_id"] == case_id), None)
                for case_id in raw_ids
                if split_lookup.get(case_id) or next((case.get("split") for case in cases if case["case_id"] == case_id), None)
            })
            if len(split_names) > 1:
                cohort_findings.append({
                    "type": "split_leakage_duplicate_identifier",
                    "duplicate_key": duplicate_key,
                    "case_ids": raw_ids,
                    "splits": split_names,
                    "severity": "severe",
                    "confidence": "high",
                    "impact": "major",
                })
                for case in cases:
                    if case["case_id"] in raw_ids:
                        add_finding(
                            case,
                            domain="duplicate_risk",
                            code="split_leakage",
                            message="Duplicate identifiers appear across different data splits.",
                            severity="severe",
                            confidence="high",
                            impact="major",
                            status="confirmed",
                            suggested_action="Remove identifier collisions across train/val/test before evaluation.",
                            details={"case_ids": raw_ids, "splits": split_names, "subtype": "duplicate_identifier"},
                        )

    for patient_id, case_ids in sorted(patient_id_map.items()):
        unique_case_ids = sorted(set(case_ids))
        if len(unique_case_ids) > 1:
            cohort_findings.append({
                "type": "repeated_patient_id",
                "patient_id": patient_id,
                "case_ids": unique_case_ids,
                "severity": "moderate",
                "confidence": "medium",
                "impact": "moderate",
            })
            for case in cases:
                if case["case_id"] in unique_case_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="repeated_patient_identifier",
                        message="The same patient identifier appears in multiple cases, which may indicate repeat scans or leakage risk across future splits.",
                        severity="moderate",
                        confidence="medium",
                        impact="moderate",
                        status="suspected",
                        suggested_action="Group repeated-patient cases before train/val/test assignment.",
                        details={"patient_id": patient_id, "case_ids": unique_case_ids},
                    )

    for study_uid, case_ids in sorted(study_uid_map.items()):
        unique_case_ids = sorted(set(case_ids))
        if len(unique_case_ids) > 1:
            cohort_findings.append({
                "type": "repeated_study_uid",
                "study_uid": study_uid,
                "case_ids": unique_case_ids,
                "severity": "moderate",
                "confidence": "high",
                "impact": "moderate",
            })
            for case in cases:
                if case["case_id"] in unique_case_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="repeated_study_uid",
                        message="The same DICOM StudyInstanceUID appears in multiple exported cases.",
                        severity="moderate",
                        confidence="high",
                        impact="moderate",
                        status="confirmed",
                        suggested_action="Verify whether these files are duplicate exports from the same study.",
                        details={"study_uid": study_uid, "case_ids": unique_case_ids},
                    )

    for series_uid, case_ids in sorted(series_uid_map.items()):
        unique_case_ids = sorted(set(case_ids))
        if len(unique_case_ids) > 1:
            cohort_findings.append({
                "type": "duplicate_series_uid",
                "series_uid": series_uid,
                "case_ids": unique_case_ids,
                "severity": "severe",
                "confidence": "high",
                "impact": "major",
            })
            for case in cases:
                if case["case_id"] in unique_case_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="duplicate_series_uid",
                        message="The same DICOM SeriesInstanceUID appears in multiple cases, suggesting duplicate or split exports of one acquisition.",
                        severity="severe",
                        confidence="high",
                        impact="major",
                        status="confirmed",
                        suggested_action="Deduplicate or regroup these cases before downstream use.",
                        details={"series_uid": series_uid, "case_ids": unique_case_ids},
                    )

    for digest, case_ids in sorted(file_hash_map.items()):
        if len(case_ids) > 1:
            cohort_findings.append({
                "type": "exact_duplicate_scan",
                "case_ids": case_ids,
                "severity": "severe",
                "confidence": "high",
                "impact": "major",
            })
            split_names = sorted({split_lookup.get(case_id) for case_id in case_ids if split_lookup.get(case_id)})
            for case in cases:
                if case["case_id"] in case_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="exact_duplicate_scan",
                        message="Image file fingerprint matches another case, suggesting a duplicated scan.",
                        severity="severe" if len(split_names) > 1 else "moderate",
                        confidence="high",
                        impact="major" if len(split_names) > 1 else "moderate",
                        status="confirmed",
                        suggested_action="Deduplicate the scan before model development.",
                        details={"duplicate_cases": case_ids, "splits": split_names},
                    )
                    if len(split_names) > 1:
                        add_finding(
                            case,
                            domain="duplicate_risk",
                            code="split_leakage",
                            message="Exact duplicate scans appear across different data splits.",
                            severity="severe",
                            confidence="high",
                            impact="major",
                            status="confirmed",
                            suggested_action="Remove duplicate scans across train/val/test before evaluation.",
                            details={"duplicate_cases": case_ids, "splits": split_names, "subtype": "exact_duplicate_scan"},
                        )

    exact_duplicate_sets = {tuple(sorted(case_ids)) for case_ids in file_hash_map.values() if len(case_ids) > 1}
    for sample_key, case_ids in sorted(sample_hash_map.items(), key=lambda item: item[1]):
        if len(case_ids) > 1 and tuple(sorted(case_ids)) not in exact_duplicate_sets:
            cohort_findings.append({
                "type": "suspected_near_duplicate_scan",
                "case_ids": case_ids,
                "severity": "moderate",
                "confidence": "medium",
                "impact": "moderate",
            })
            for case in cases:
                if case["case_id"] in case_ids:
                    add_finding(
                        case,
                        domain="duplicate_risk",
                        code="suspected_near_duplicate_scan",
                        message="Sampled image content matches another case after accounting for shape and spacing, suggesting a duplicate or repeated reconstruction.",
                        severity="moderate",
                        confidence="medium",
                        impact="moderate",
                        status="suspected",
                        suggested_action="Review both scans before placing them in different splits.",
                        details={"duplicate_cases": case_ids},
                    )
                    split_names = sorted({split_lookup.get(case_id) for case_id in case_ids if split_lookup.get(case_id)})
                    if len(split_names) > 1:
                        add_finding(
                            case,
                            domain="duplicate_risk",
                            code="split_leakage",
                            message="Suspected near-duplicate scans appear across different data splits.",
                            severity="moderate",
                            confidence="medium",
                            impact="major",
                            status="suspected",
                            suggested_action="Review possible leakage before reporting evaluation results.",
                            details={"duplicate_cases": case_ids, "splits": split_names, "subtype": "suspected_near_duplicate_scan"},
                        )

    for case in cases:
        classify_case(case)

    split_counter = Counter(case.get("split") or "unspecified" for case in cases)
    status_counter = Counter(case.get("status") or "unknown" for case in cases)
    finding_counter = Counter(finding["code"] for case in cases for finding in (case.get("findings") or []))
    metadata_availability = {
        "orientation_counts": dict(orientation_counter),
        "manufacturer_counts": dict(manufacturer_counter),
    }

    return {
        "cohort_findings": cohort_findings,
        "split_counter": dict(split_counter),
        "status_counter": dict(status_counter),
        "finding_counter": dict(finding_counter.most_common(20)),
        "metadata_availability": metadata_availability,
        "spacing_bounds": {str(axis): bounds for axis, bounds in spacing_bounds.items()},
        "shape_bounds": {str(axis): bounds for axis, bounds in shape_bounds.items()},
        "fov_bounds": {str(axis): bounds for axis, bounds in fov_bounds.items()},
    }


def render_markdown(manifest: Dict[str, Any], cohort_summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> str:
    lines = [
        f"# CBCT QC: {manifest['dataset_name']}",
        "",
        "## Overview",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- Run timestamp: `{manifest['generated_at']}`",
        f"- Audited cases: `{manifest['audited_case_count']}`",
        f"- Sample limit: `{manifest['sample_limit']}`",
        f"- Label schema source: `{manifest['label_schema_source']}`",
        "",
        "## Status Distribution",
        "",
    ]
    for status, count in sorted((cohort_summary.get("status_counter") or {}).items()):
        lines.append(f"- {status}: `{count}`")

    lines.extend([
        "",
        "## Top Findings",
        "",
    ])
    for code, count in list((cohort_summary.get("finding_counter") or {}).items())[:10]:
        lines.append(f"- {code}: `{count}`")

    cohort_findings = cohort_summary.get("cohort_findings") or []
    lines.extend([
        "",
        "## Cohort Risks",
        "",
    ])
    if cohort_findings:
        for finding in cohort_findings[:10]:
            case_ids = ", ".join(finding.get("case_ids", [])[:6])
            lines.append(f"- {finding['type']}: `{case_ids}`")
    else:
        lines.append("- No cohort-level duplicate or leakage findings were confirmed in this run.")

    lines.extend([
        "",
        "## Cases Requiring Attention",
        "",
    ])
    attention_cases = [case for case in cases if case.get("status") in {"reject", "needs_manual_review", "usable_with_warnings"}]
    if attention_cases:
        for case in attention_cases[:20]:
            lines.append(f"- `{case['case_id']}` -> `{case['status']}`: {case.get('status_rationale', 'QC completed.')}")
    else:
        lines.append("- No cases were flagged beyond the usable tier.")

    return "\n".join(lines) + "\n"


def sample_cases(cases: List[Dict[str, Any]], sample_limit: Optional[int], sample_seed: int) -> List[Dict[str, Any]]:
    if sample_limit is None or sample_limit <= 0 or sample_limit >= len(cases):
        return list(cases)
    rng = np.random.default_rng(sample_seed)
    indices = sorted(int(index) for index in rng.choice(len(cases), size=sample_limit, replace=False))
    return [cases[index] for index in indices]


def normalize_case_output(case: Dict[str, Any], dataset_root: Path) -> Dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "normalized_case_id": case["normalized_case_id"],
        "split": case.get("split"),
        "ingest_status": case.get("ingest_status"),
        "label_status": case.get("label_status"),
        "status": case.get("status"),
        "status_rationale": case.get("status_rationale"),
        "image_format": case.get("image_format"),
        "source_image_path": safe_relpath(Path(case["source_image_path"]), dataset_root) if case.get("source_image_path") else None,
        "source_label_path": safe_relpath(Path(case["source_label_path"]), dataset_root) if case.get("source_label_path") else None,
        "canonical_image_name": case.get("canonical_image_name"),
        "canonical_label_name": case.get("canonical_label_name"),
        "findings": case.get("findings") or [],
        "auto_corrections": case.get("auto_corrections") or [],
    }


def build_corrections(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for case in cases:
        for correction in case.get("auto_corrections") or []:
            rows.append({
                "case_id": case["case_id"],
                "normalized_case_id": case["normalized_case_id"],
                **correction,
            })
        if case.get("status") == "reject":
            rows.append({
                "case_id": case["case_id"],
                "normalized_case_id": case["normalized_case_id"],
                "type": "recommended_exclusion",
                "from": case["case_id"],
                "to": None,
                "applied": False,
                "safe_to_apply": False,
                "reason": case.get("status_rationale"),
            })
    return rows


parser = argparse.ArgumentParser(description="Audit CBCT datasets for metadata, geometry, labels, quality, and split risk.")
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--output-root", type=Path, default=None)
parser.add_argument("--report-key", type=str, default=None)
parser.add_argument("--sample-limit", type=int, default=None, help="Audit only N reproducibly sampled cases.")
parser.add_argument("--sample-seed", type=int, default=13)
parser.add_argument("--split-json", type=Path, default=None, help="Optional explicit split assignment JSON.")
parser.add_argument("--max-voxel-sample", type=int, default=DEFAULT_MAX_VOXEL_SAMPLE)
parser.add_argument("--label-policy", choices=("optional", "required", "ignore"), default="optional", help="How missing annotation volumes should be handled.")
args = parser.parse_args()


def main() -> None:
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    cases, discovery = discover_cases(dataset_root)
    cases = sample_cases(cases, args.sample_limit, args.sample_seed)
    label_schema = load_label_schema(dataset_root)

    split_lookup = {}
    if args.split_json:
        split_payload = read_json(args.split_json)
        splits = split_payload.get("splits", split_payload)
        if isinstance(splits, dict):
            for split_name, case_ids in splits.items():
                for case_id in case_ids or []:
                    split_lookup[str(case_id)] = str(split_name)
    for case in cases:
        if case.get("case_id") in split_lookup:
            case["split"] = split_lookup[case["case_id"]]
        apply_case_level_checks(case, label_schema)

    cohort_summary = apply_cohort_level_checks(cases, split_lookup)

    report_key = args.report_key or dataset_root.name
    output_root = (args.output_root or (DEFAULT_REPORT_ROOT / report_key)).resolve()
    ensure_dir(output_root)

    manifest = {
        "dataset_name": dataset_root.name,
        "dataset_root": str(dataset_root),
        "generated_at": utc_now_iso(),
        "report_key": report_key,
        "sample_limit": args.sample_limit,
        "sample_seed": args.sample_seed,
        "label_policy": args.label_policy,
        "audited_case_count": len(cases),
        "discovery": discovery,
        "label_schema_source": label_schema.get("source"),
        "label_schema_values": label_schema.get("allowed_values"),
        "split_json": str(args.split_json.resolve()) if args.split_json else None,
        "output_root": str(output_root),
    }

    case_rows = []
    normalized_rows = []
    for case in cases:
        serializable_case = {
            **case,
            "image_paths": [str(path.resolve()) for path in case.get("image_paths") or []],
            "label_paths": [str(path.resolve()) for path in case.get("label_paths") or []],
        }
        case_rows.append(serializable_case)
        normalized_rows.append(normalize_case_output(case, dataset_root))

    corrections = build_corrections(cases)
    markdown = render_markdown(manifest, cohort_summary, cases)

    write_json(manifest, output_root / "manifest.json")
    write_json(cohort_summary, output_root / "cohort_summary.json")
    (output_root / "cohort_summary.md").write_text(markdown, encoding="utf-8")
    write_jsonl(case_rows, output_root / "cases.jsonl")
    write_jsonl(normalized_rows, output_root / "normalized_cases.jsonl")
    write_jsonl(corrections, output_root / "corrections.jsonl")

    print({
        "dataset_root": str(dataset_root),
        "audited_case_count": len(cases),
        "output_root": str(output_root),
        "status_counter": cohort_summary.get("status_counter"),
        "top_findings": cohort_summary.get("finding_counter"),
    })


if __name__ == "__main__":
    main()
