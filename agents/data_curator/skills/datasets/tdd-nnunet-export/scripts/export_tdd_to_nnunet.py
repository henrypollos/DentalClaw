#!/usr/bin/env python3
"""Export the local TDD dataset into nnUNet-compatible 2D PNG datasets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

import sys

CURRENT_FILE = Path(__file__).resolve()
SKILL_DIR = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[6]
TDD_SKILL_SCRIPTS = CURRENT_FILE.parents[2] / "tdd-curation" / "scripts"
if str(TDD_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TDD_SKILL_SCRIPTS))
CORE_LIB_DIR = CURRENT_FILE.parents[3] / "core" / "_lib"
if str(CORE_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_LIB_DIR))

from tdd_common import NUMERIC_TOOTH_LABELS, image_size, load_cases, polygon_objects_for_case  # noqa: E402
from dataset_qc import run_dataset_qc, write_report_bundle  # noqa: E402


TASKS: Dict[str, Dict[str, object]] = {
    "teeth_binary": {
        "annotation_mode": "binary_mask",
        "mask_key": "teeth_mask",
        "dataset_id": 501,
        "dataset_name": "TDDTeethBinary2D",
        "label_name": "teeth",
        "channel_name": "panoramic_radiograph",
        "description": "TDD panoramic radiographs with binary teeth masks exported as nnUNet 2D PNG.",
    },
    "maxillomandibular_binary": {
        "annotation_mode": "binary_mask",
        "mask_key": "maxillomandibular_mask",
        "dataset_id": 502,
        "dataset_name": "TDDMaxillomandibularBinary2D",
        "label_name": "maxillomandibular",
        "channel_name": "panoramic_radiograph",
        "description": "TDD panoramic radiographs with binary maxillomandibular masks exported as nnUNet 2D PNG.",
    },
    "teeth_32class": {
        "annotation_mode": "polygon_multiclass",
        "dataset_id": 503,
        "dataset_name": "TDDTeeth32Class2D",
        "channel_name": "panoramic_radiograph",
        "description": "TDD panoramic radiographs with 32-class permanent-tooth masks rasterized from polygon annotations.",
    },
}

NUMERIC_TOOTH_LABEL_SET = set(NUMERIC_TOOTH_LABELS)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_roots(output_root: Path) -> Tuple[Path, Path]:
    """
    Accept either the stable nnUNet workspace root:
      artifacts/datasets/nnUNet
    or the raw root itself:
      artifacts/datasets/nnUNet/nnUNet_raw
    and normalize both to:
      (nnunet_root, nnunet_raw_root)
    """
    resolved = output_root.resolve()
    if resolved.name == "nnUNet_raw":
        nnunet_raw_root = ensure_dir(resolved)
        nnunet_root = ensure_dir(resolved.parent)
        return nnunet_root, nnunet_raw_root
    nnunet_root = ensure_dir(resolved)
    nnunet_raw_root = ensure_dir(nnunet_root / "nnUNet_raw")
    return nnunet_root, nnunet_raw_root


def write_json(payload: dict, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_status(path: Path, payload: dict):
    write_json({
        "updated_at": utc_now_iso(),
        **payload,
    }, path)


def sanitize_dataset_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "", name)
    return value or "Dataset"


def find_next_dataset_id(nnunet_raw_root: Path, minimum: int = 501) -> int:
    existing = []
    if nnunet_raw_root.exists():
        for path in nnunet_raw_root.iterdir():
            if not path.is_dir():
                continue
            match = re.match(r"Dataset(\d{3})_", path.name)
            if match:
                existing.append(int(match.group(1)))
    if not existing:
        return minimum
    return max(max(existing) + 1, minimum)


def parse_dataset_id(folder_name: str) -> Optional[int]:
    match = re.match(r"Dataset(\d{3})_", folder_name)
    return int(match.group(1)) if match else None


def normalize_request_payload(args: argparse.Namespace, nnunet_root: Path) -> Dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(nnunet_root.resolve()),
        "preprocessed_root": str(args.preprocessed_root.resolve()),
        "results_root": str(args.results_root.resolve()),
        "task": args.task,
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "threshold": int(args.threshold),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "skip_qc": bool(args.skip_qc),
        "limit": int(args.limit) if args.limit is not None else None,
        "overwrite": bool(args.overwrite),
    }


def request_signature(payload: Dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def request_match_tokens(payload: Dict[str, object]) -> List[str]:
    tokens = [
        "export_tdd_to_nnunet.py",
        "--dataset-root {}".format(payload["dataset_root"]),
        "--output-root {}".format(payload["output_root"]),
        "--task {}".format(payload["task"]),
        "--test-ratio {}".format(payload["test_ratio"]),
        "--seed {}".format(payload["seed"]),
    ]
    if payload.get("dataset_id") is not None:
        tokens.append("--dataset-id {}".format(payload["dataset_id"]))
    if payload.get("dataset_name") is not None:
        tokens.append("--dataset-name {}".format(payload["dataset_name"]))
    if payload.get("limit") is not None:
        tokens.append("--limit {}".format(payload["limit"]))
    if payload.get("skip_qc"):
        tokens.append("--skip-qc")
    if payload.get("overwrite"):
        tokens.append("--overwrite")
    return tokens


def find_running_duplicate(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    tokens = request_match_tokens(payload)
    current_pid = os.getpid()
    parent_pid = os.getppid()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        cmd = parts[1]
        if pid in {current_pid, parent_pid}:
            continue
        if "codex-linux-sandbox" in cmd:
            continue
        if "export_tdd_to_nnunet.py" not in cmd:
            continue
        if "python " not in cmd and "/bin/sh -c" not in cmd:
            continue
        if all(token in cmd for token in tokens):
            return {
                "pid": pid,
                "command": cmd,
            }
    return None


def acquire_request_lock(lock_root: Path, signature: str, payload: Dict[str, object]) -> Tuple[object, Path, Path]:
    ensure_dir(lock_root)
    lock_path = lock_root / "{}.lock".format(signature)
    meta_path = lock_root / "{}.json".format(signature)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(str(meta_path))
    meta_payload = {
        "signature": signature,
        "pid": os.getpid(),
        "started_at": utc_now_iso(),
        "request": payload,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock_handle, lock_path, meta_path


def dataset_structure_ready(dataset_root: Path, expect_holdout: bool) -> bool:
    required = [
        dataset_root / "dataset.json",
        dataset_root / "imagesTr",
        dataset_root / "labelsTr",
    ]
    if expect_holdout:
        required.extend([
            dataset_root / "imagesTs",
            dataset_root / "labelsTs",
        ])
    for path in required:
        if path.suffix == ".json":
            if not path.is_file():
                return False
        else:
            if not path.is_dir():
                return False
    if not any((dataset_root / "imagesTr").iterdir()):
        return False
    if not any((dataset_root / "labelsTr").iterdir()):
        return False
    if expect_holdout:
        if not any((dataset_root / "imagesTs").iterdir()):
            return False
        if not any((dataset_root / "labelsTs").iterdir()):
            return False
    return True


def dataset_split_case_ids(dataset_root: Path) -> Tuple[List[str], List[str]]:
    train_ids = sorted(path.stem.replace("_0000", "") for path in (dataset_root / "imagesTr").glob("*_0000.png"))
    test_ids = sorted(path.stem for path in (dataset_root / "labelsTs").glob("*.png"))
    return train_ids, test_ids


def select_qc_case_ids(train_ids: List[str], test_ids: List[str], qc_limit: Optional[int]) -> Tuple[List[str], List[str], int]:
    if qc_limit is None or qc_limit <= 0:
        return list(train_ids), list(test_ids), len(train_ids) + len(test_ids)
    selected_train = list(train_ids[:qc_limit])
    remaining = max(0, qc_limit - len(selected_train))
    selected_test = list(test_ids[:remaining]) if remaining > 0 else []
    return selected_train, selected_test, qc_limit


def build_qc_cases_for_export(source_cases_by_id: Dict[str, dict], train_ids: List[str], test_ids: List[str], export_task: str) -> List[dict]:
    qc_cases = []
    for case_id in train_ids + test_ids:
        case = source_cases_by_id.get(case_id)
        if case is None:
            continue
        bbox_annotations = []
        for obj in (case.get("bbox_item") or {}).get("Label", {}).get("objects", []):
            bbox_annotations.append({
                "label": str(obj.get("title", "")).strip(),
                "bbox_xyxy": obj.get("bounding box") or [],
            })
        polygon_annotations = []
        for obj in (case.get("polygon_item") or {}).get("Label", {}).get("objects", []):
            polygon_annotations.append({
                "label": str(obj.get("title", "")).strip(),
                "polygons": obj.get("polygons") or [],
            })
        raster_labels = []
        if case.get("teeth_mask") is not None:
            raster_labels.append({"role": "teeth_mask", "path": case["teeth_mask"]})
        if case.get("maxillomandibular_mask") is not None:
            raster_labels.append({"role": "maxillomandibular_mask", "path": case["maxillomandibular_mask"]})
        qc_cases.append({
            "case_id": case_id,
            "images": [{"role": "panoramic_image", "path": case["radiograph"]}],
            "raster_labels": raster_labels,
            "bbox_annotations": bbox_annotations,
            "polygon_annotations": polygon_annotations,
            "metadata": {
                "export_split": "train" if case_id in train_ids else "test",
                "export_task": export_task,
            },
        })
    return qc_cases


def ensure_qc_report_for_dataset(
    dataset_root: Path,
    dataset_folder_name: str,
    dataset_id: Optional[int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_ids, test_ids = dataset_split_case_ids(dataset_root)
    qc_train_ids, qc_test_ids, qc_limit = select_qc_case_ids(train_ids, test_ids, args.qc_limit)
    qc_cases = build_export_qc_cases(dataset_root, qc_train_ids, qc_test_ids, args.task)
    qc_report = run_dataset_qc(
        dataset_root=dataset_root.resolve(),
        dataset_name=dataset_folder_name,
        dataset_mode="nnunet_export",
        cases=qc_cases,
        split_payload={"splits": {"train": qc_train_ids, "val": [], "test": qc_test_ids}},
        metadata={
            "source_dataset_root": str(args.dataset_root.resolve()),
            "export_dataset_root": str(dataset_root),
            "export_task": args.task,
            "nnunet_dataset_id": dataset_id,
            "generated_from_existing_dataset": True,
            "qc_limit": qc_limit,
            "qc_is_sampled": qc_limit > 0,
        },
    )
    qc_report_paths = write_report_bundle(qc_report, report_key=dataset_folder_name)
    return {
        "status": qc_report["summary"]["dataset_status"],
        "report_json": qc_report_paths["report_json"],
        "report_md": qc_report_paths["report_md"],
        "case_count": qc_report["summary"]["case_count"],
        "train_case_count": len(qc_train_ids),
        "test_case_count": len(qc_test_ids),
        "qc_limit": qc_limit,
    }


def find_reusable_dataset(
    nnunet_root: Path,
    nnunet_raw_root: Path,
    args: argparse.Namespace,
    task_cfg: Dict[str, object],
    request_payload: Dict[str, object],
) -> Optional[Dict[str, object]]:
    dataset_name = sanitize_dataset_name(args.dataset_name or str(task_cfg["dataset_name"]))
    expect_holdout = float(args.test_ratio) > 0
    candidates = sorted(nnunet_raw_root.glob("Dataset*_" + dataset_name))
    for dataset_root in candidates:
        if not dataset_root.is_dir():
            continue
        if not dataset_structure_ready(dataset_root, expect_holdout=expect_holdout):
            continue
        if not exported_dataset_is_nnunet_ready(dataset_root, expect_holdout=expect_holdout):
            continue
        folder_name = dataset_root.name
        delivery_json = nnunet_root / "nnUNet_delivery_{}.json".format(folder_name)
        delivery_status = nnunet_root / "nnUNet_delivery_{}.status.json".format(folder_name)
        qc_json = REPO_ROOT / "artifacts" / "results" / "reports" / "datasets_qc" / "{}.qc.json".format(folder_name)
        qc_md = REPO_ROOT / "artifacts" / "results" / "reports" / "datasets_qc" / "{}.qc.md".format(folder_name)
        payload = {}
        if delivery_json.is_file():
            try:
                payload = json.loads(delivery_json.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        if payload:
            if payload.get("task") != args.task:
                continue
            if payload.get("dataset_root_source") != request_payload["dataset_root"]:
                continue
            if int(payload.get("seed", args.seed)) != int(args.seed):
                continue
            if int(payload.get("threshold", args.threshold)) != int(args.threshold):
                continue
        if delivery_status.is_file():
            try:
                status_payload = json.loads(delivery_status.read_text(encoding="utf-8"))
            except Exception:
                status_payload = {}
            if status_payload.get("status") != "completed":
                write_status(delivery_status, {
                    "status": "completed",
                    "stage": "reused_existing_dataset",
                    "task": args.task,
                    "dataset_root_source": str(args.dataset_root.resolve()),
                    "dataset_root": str(dataset_root),
                    "dataset_id": parse_dataset_id(folder_name),
                    "dataset_name": dataset_name,
                    "delivery_report": str(delivery_json) if delivery_json.is_file() else None,
                })
        qc_info = None
        if not args.skip_qc and not qc_json.is_file():
            qc_info = ensure_qc_report_for_dataset(
                dataset_root=dataset_root,
                dataset_folder_name=folder_name,
                dataset_id=parse_dataset_id(folder_name),
                args=args,
            )
        return {
            "dataset_root": str(dataset_root),
            "dataset_folder": folder_name,
            "dataset_id": parse_dataset_id(folder_name),
            "delivery_report": str(delivery_json) if delivery_json.is_file() else None,
            "delivery_status": str(delivery_status) if delivery_status.is_file() else None,
            "qc_report": str(qc_json) if (not args.skip_qc and qc_json.is_file()) else None,
            "qc_report_md": str(qc_md) if (not args.skip_qc and qc_md.is_file()) else None,
            "qc_status": (
                "skipped"
                if args.skip_qc
                else (qc_info["status"] if qc_info is not None else None)
            ),
        }
    return None


def choose_png_mode(image: Image.Image) -> str:
    return "L"


def convert_image_to_png(src: Path, dst: Path) -> dict:
    ensure_dir(dst.parent)
    with Image.open(src) as image:
        converted = image.convert(choose_png_mode(image))
        converted.save(dst)
        return {
            "mode": converted.mode,
            "size": list(converted.size),
        }


def export_label_role(task_name: str) -> str:
    if task_name == "teeth_32class":
        return "teeth_32class_mask"
    if task_name == "maxillomandibular_binary":
        return "maxillomandibular_binary_mask"
    return "teeth_binary_mask"


def build_export_qc_cases(
    dataset_root: Path,
    train_ids: List[str],
    test_ids: List[str],
    task_name: str,
) -> List[dict]:
    label_role = export_label_role(task_name)
    qc_cases: List[dict] = []
    for split_name, case_ids in (("train", train_ids), ("test", test_ids)):
        image_dir = dataset_root / ("imagesTr" if split_name == "train" else "imagesTs")
        label_dir = dataset_root / ("labelsTr" if split_name == "train" else "labelsTs")
        for case_id in case_ids:
            qc_cases.append({
                "case_id": case_id,
                "images": [{"role": "panoramic_image", "path": image_dir / f"{case_id}_0000.png"}],
                "raster_labels": [{"role": label_role, "path": label_dir / f"{case_id}.png"}],
                "metadata": {
                    "export_split": split_name,
                    "export_task": task_name,
                },
            })
    return qc_cases


def exported_dataset_is_nnunet_ready(dataset_root: Path, expect_holdout: bool) -> bool:
    sample_paths = sorted((dataset_root / "imagesTr").glob("*_0000.png"))[:3]
    if expect_holdout:
        sample_paths.extend(sorted((dataset_root / "imagesTs").glob("*_0000.png"))[:2])
    if not sample_paths:
        return False
    try:
        for path in sample_paths:
            with Image.open(path) as image:
                if len(image.getbands()) != 1:
                    return False
        return True
    except Exception:
        return False


def convert_mask_to_label_png(src: Path, dst: Path, threshold: int) -> dict:
    ensure_dir(dst.parent)
    with Image.open(src) as image:
        gray = image.convert("L")
        label = gray.point(lambda x: 1 if x > threshold else 0, mode="L")
        label.save(dst)
        values = sorted(set(label.getdata()))
        histogram = label.histogram()
        foreground = histogram[1] if len(histogram) > 1 else 0
        total = sum(histogram)
        return {
            "mode": label.mode,
            "size": list(label.size),
            "label_values": values,
            "foreground_ratio": (foreground / total) if total else 0.0,
        }


def convert_polygons_to_label_png(case: dict, dst: Path) -> dict:
    ensure_dir(dst.parent)
    width, height = image_size(case["radiograph"])
    label = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(label)
    skipped_labels = {}
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

    label.save(dst)
    values = sorted(set(label.getdata()))
    histogram = label.histogram()
    foreground = sum(histogram[1:]) if len(histogram) > 1 else 0
    total = sum(histogram)
    return {
        "mode": label.mode,
        "size": [width, height],
        "label_values": values,
        "class_values": sorted(case_values),
        "foreground_ratio": (foreground / total) if total else 0.0,
        "skipped_labels": dict(sorted(skipped_labels.items())),
    }


def validate_case(case: dict, task_cfg: dict):
    if case["radiograph"] is None:
        raise ValueError(f"Case {case['case_id']} is missing radiograph.")
    if task_cfg["annotation_mode"] == "polygon_multiclass":
        if case.get("polygon_item") is None:
            raise ValueError(f"Case {case['case_id']} is missing polygon annotations for 32-class export.")
        return
    if case[task_cfg["mask_key"]] is None:
        raise ValueError(f"Case {case['case_id']} is missing mask for task {task_cfg['mask_key']}.")
    if image_size(case["radiograph"]) != image_size(case[task_cfg["mask_key"]]):
        raise ValueError(
            f"Case {case['case_id']} image/mask size mismatch: "
            f"{image_size(case['radiograph'])} vs {image_size(case[task_cfg['mask_key']])}"
        )


def split_cases(case_ids: List[str], test_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if test_ratio <= 0:
        return sorted(case_ids), []
    if test_ratio >= 1:
        raise ValueError("test_ratio must be smaller than 1.")
    ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_count = int(round(len(ids) * test_ratio))
    if test_count <= 0:
        return sorted(ids), []
    test_ids = sorted(ids[:test_count])
    train_ids = sorted(ids[test_count:])
    return train_ids, test_ids


def build_dataset_json(task_cfg: dict, dataset_name: str, num_training: int) -> dict:
    if task_cfg["annotation_mode"] == "polygon_multiclass":
        labels = {"background": 0}
        for tooth_label in NUMERIC_TOOTH_LABELS:
            labels[tooth_label] = int(tooth_label)
        return {
            "name": dataset_name,
            "channel_names": {"0": str(task_cfg["channel_name"])},
            "labels": labels,
            "numTraining": num_training,
            "file_ending": ".png",
            "overwrite_image_reader_writer": "NaturalImage2DIO",
            "description": str(task_cfg["description"]),
            "converted_by": "DentalClaw data_curator tdd-nnunet-export",
        }
    return {
        "name": dataset_name,
        "channel_names": {"0": str(task_cfg["channel_name"])},
        "labels": {
            "background": 0,
            str(task_cfg["label_name"]): 1,
        },
        "numTraining": num_training,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "description": str(task_cfg["description"]),
        "converted_by": "DentalClaw data_curator tdd-nnunet-export",
    }


def main():
    parser = argparse.ArgumentParser(description="Export TDD into nnUNet-compatible 2D PNG datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the local TDD dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Stable nnUNet artifact root (`.../artifacts/datasets/nnUNet`) or `.../nnUNet_raw` for compatibility.",
    )
    parser.add_argument("--preprocessed-root", type=Path, default=REPO_ROOT / "artifacts" / "datasets" / "nnUNet" / "nnUNet_preprocessed", help="nnUNet_preprocessed root used in handoff commands.")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "artifacts" / "models" / "nnUNet" / "nnUNet_results", help="nnUNet_results root used in handoff commands.")
    parser.add_argument("--task", choices=sorted(TASKS), default="teeth_binary")
    parser.add_argument("--dataset-id", type=int, default=None, help="Override nnUNet dataset id.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override nnUNet dataset name suffix.")
    parser.add_argument("--threshold", type=int, default=127, help="Mask threshold for JPEG -> integer PNG label conversion.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Optional holdout ratio to place held-out cases in imagesTs and labelsTs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qc-limit", type=int, default=0, help="Run full QC by default. Use a positive value to limit QC to the first N exported cases.")
    parser.add_argument("--skip-qc", action="store_true", help="Skip dataset QC generation. Use only when the caller explicitly disables QC.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_cfg = TASKS[args.task]
    nnunet_root, nnunet_raw_root = resolve_output_roots(args.output_root)
    nnunet_preprocessed_root = ensure_dir(args.preprocessed_root.resolve())
    nnunet_results_root = ensure_dir(args.results_root.resolve())
    report_root = ensure_dir(nnunet_root)
    request_payload = normalize_request_payload(args, nnunet_root)
    signature = request_signature(request_payload)
    reusable = find_reusable_dataset(nnunet_root, nnunet_raw_root, args, task_cfg, request_payload)
    if reusable is not None:
        print(json.dumps({
            "status": "already_exists",
            "reason": "reusable_dataset_detected",
            "request_signature": signature,
            "request": request_payload,
            **reusable,
        }, indent=2, ensure_ascii=False))
        return
    duplicate = find_running_duplicate(request_payload)
    if duplicate is not None:
        print(json.dumps({
            "status": "already_running",
            "reason": "matching_export_process_detected",
            "request_signature": signature,
            "existing_pid": duplicate["pid"],
            "existing_command": duplicate["command"],
            "request": request_payload,
        }, indent=2, ensure_ascii=False))
        return
    lock_root = report_root / ".export_locks"
    lock_handle = None
    lock_meta_path = None
    try:
        lock_handle, _, lock_meta_path = acquire_request_lock(lock_root, signature, request_payload)
    except RuntimeError as exc:
        meta_path = Path(str(exc))
        meta_payload = {}
        if meta_path.is_file():
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_payload = {}
        print(json.dumps({
            "status": "already_running",
            "reason": "request_lock_held",
            "request_signature": signature,
            "lock_meta_path": str(meta_path),
            "lock_meta": meta_payload,
            "request": request_payload,
        }, indent=2, ensure_ascii=False))
        return

    try:
        dataset_id = (
            args.dataset_id
            if args.dataset_id is not None
            else find_next_dataset_id(nnunet_raw_root, minimum=int(task_cfg["dataset_id"]))
        )
        dataset_name = sanitize_dataset_name(args.dataset_name or str(task_cfg["dataset_name"]))
        dataset_folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
        dataset_root = nnunet_raw_root / dataset_folder_name
        if dataset_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Dataset folder already exists: {dataset_root}")
            shutil.rmtree(dataset_root)
        dataset_root = ensure_dir(dataset_root)
        images_tr = ensure_dir(dataset_root / "imagesTr")
        labels_tr = ensure_dir(dataset_root / "labelsTr")
        status_path = report_root / f"nnUNet_delivery_{dataset_folder_name}.status.json"
        write_status(status_path, {
            "status": "running",
            "stage": "initializing",
            "task": args.task,
            "request_signature": signature,
            "dataset_root_source": str(args.dataset_root.resolve()),
            "dataset_root": str(dataset_root),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
        })

        matched_cases = [case for case in load_cases(args.dataset_root) if case["has_all_assets"]]
        for case in matched_cases:
            validate_case(case, task_cfg)
        if args.limit is not None:
            matched_cases = matched_cases[: args.limit]
        write_status(status_path, {
            "status": "running",
            "stage": "validated",
            "task": args.task,
            "request_signature": signature,
            "dataset_root_source": str(args.dataset_root.resolve()),
            "dataset_root": str(dataset_root),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "matched_case_count": len(matched_cases),
        })

        case_ids = [case["case_id"] for case in matched_cases]
        train_ids, test_ids = split_cases(case_ids, args.test_ratio, args.seed)
        cases_by_id = {case["case_id"]: case for case in matched_cases}

        images_ts = ensure_dir(dataset_root / "imagesTs") if test_ids else None
        labels_ts = ensure_dir(dataset_root / "labelsTs") if test_ids else None

        image_modes = {}
        label_values_union = set()
        foreground_ratios = []
        size_sample = {}
        exported_rows = []
        skipped_labels_aggregate = {}
        empty_label_cases = []

        polygon_annotation_ref = str((args.dataset_root / "Segmentation" / "teeth_polygon.json").resolve())

        for index, case_id in enumerate(train_ids, start=1):
            if index == 1 or index % 50 == 0 or index == len(train_ids):
                write_status(status_path, {
                    "status": "running",
                    "stage": "exporting_train",
                    "task": args.task,
                    "request_signature": signature,
                    "dataset_root_source": str(args.dataset_root.resolve()),
                    "dataset_root": str(dataset_root),
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "train_done": index - 1,
                    "train_total": len(train_ids),
                    "holdout_done": 0,
                    "holdout_total": len(test_ids),
                })
            case = cases_by_id[case_id]
            image_dst = images_tr / f"{case_id}_0000.png"
            label_dst = labels_tr / f"{case_id}.png"
            image_info = convert_image_to_png(case["radiograph"], image_dst)
            if task_cfg["annotation_mode"] == "polygon_multiclass":
                label_info = convert_polygons_to_label_png(case, label_dst)
            else:
                label_info = convert_mask_to_label_png(case[task_cfg["mask_key"]], label_dst, args.threshold)
            if image_info["size"] != label_info["size"]:
                raise ValueError(f"Converted size mismatch for case {case_id}: {image_info['size']} vs {label_info['size']}")
            image_modes.setdefault(image_info["mode"], 0)
            image_modes[image_info["mode"]] += 1
            label_values_union.update(label_info["label_values"])
            foreground_ratios.append(label_info["foreground_ratio"])
            if not [value for value in label_info["label_values"] if value != 0]:
                empty_label_cases.append(case_id)
            for skipped_label, count in label_info.get("skipped_labels", {}).items():
                skipped_labels_aggregate[skipped_label] = skipped_labels_aggregate.get(skipped_label, 0) + count
            size_key = tuple(image_info["size"])
            size_sample[size_key] = size_sample.get(size_key, 0) + 1
            exported_rows.append({
                "case_id": case_id,
                "split": "train",
                "image_src": str(case["radiograph"].resolve()),
                "label_src": str(
                    case[task_cfg["mask_key"]].resolve()
                    if task_cfg["annotation_mode"] == "binary_mask"
                    else f"{polygon_annotation_ref}#{case_id}"
                ),
                "image_dst": str(image_dst),
                "label_dst": str(label_dst),
                "label_values": label_info["label_values"],
            })

        for index, case_id in enumerate(test_ids, start=1):
            if index == 1 or index % 50 == 0 or index == len(test_ids):
                write_status(status_path, {
                    "status": "running",
                    "stage": "exporting_holdout",
                    "task": args.task,
                    "request_signature": signature,
                    "dataset_root_source": str(args.dataset_root.resolve()),
                    "dataset_root": str(dataset_root),
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "train_done": len(train_ids),
                    "train_total": len(train_ids),
                    "holdout_done": index - 1,
                    "holdout_total": len(test_ids),
                })
            case = cases_by_id[case_id]
            if images_ts is None or labels_ts is None:
                raise RuntimeError("imagesTs/labelsTs roots were not initialized for holdout export.")
            image_dst = images_ts / f"{case_id}_0000.png"
            label_dst = labels_ts / f"{case_id}.png"
            image_info = convert_image_to_png(case["radiograph"], image_dst)
            if task_cfg["annotation_mode"] == "polygon_multiclass":
                label_info = convert_polygons_to_label_png(case, label_dst)
            else:
                label_info = convert_mask_to_label_png(case[task_cfg["mask_key"]], label_dst, args.threshold)
            if image_info["size"] != label_info["size"]:
                raise ValueError(f"Converted size mismatch for case {case_id}: {image_info['size']} vs {label_info['size']}")
            image_modes.setdefault(image_info["mode"], 0)
            image_modes[image_info["mode"]] += 1
            label_values_union.update(label_info["label_values"])
            foreground_ratios.append(label_info["foreground_ratio"])
            if not [value for value in label_info["label_values"] if value != 0]:
                empty_label_cases.append(case_id)
            for skipped_label, count in label_info.get("skipped_labels", {}).items():
                skipped_labels_aggregate[skipped_label] = skipped_labels_aggregate.get(skipped_label, 0) + count
            size_key = tuple(image_info["size"])
            size_sample[size_key] = size_sample.get(size_key, 0) + 1
            exported_rows.append({
                "case_id": case_id,
                "split": "holdout",
                "image_src": str(case["radiograph"].resolve()),
                "label_src": str(
                    case[task_cfg["mask_key"]].resolve()
                    if task_cfg["annotation_mode"] == "binary_mask"
                    else f"{polygon_annotation_ref}#{case_id}"
                ),
                "image_dst": str(image_dst),
                "label_dst": str(label_dst),
                "label_values": label_info["label_values"],
            })

        dataset_json = build_dataset_json(task_cfg, dataset_name, len(train_ids))
        write_json(dataset_json, dataset_root / "dataset.json")

        qc_limit = int(args.qc_limit)
        qc_report = None
        qc_report_paths = None
        if not args.skip_qc:
            qc_train_ids, qc_test_ids, qc_limit = select_qc_case_ids(train_ids, test_ids, args.qc_limit)
            qc_cases = build_export_qc_cases(dataset_root, qc_train_ids, qc_test_ids, args.task)
            qc_report = run_dataset_qc(
                dataset_root=dataset_root.resolve(),
                dataset_name=dataset_folder_name,
                dataset_mode="nnunet_export",
                cases=qc_cases,
                split_payload={"splits": {"train": qc_train_ids, "val": [], "test": qc_test_ids}},
                metadata={
                    "source_dataset_root": str(args.dataset_root.resolve()),
                    "export_dataset_root": str(dataset_root),
                    "export_task": args.task,
                    "nnunet_dataset_id": dataset_id,
                    "qc_limit": qc_limit,
                    "qc_is_sampled": qc_limit > 0,
                },
            )
            qc_report_paths = write_report_bundle(qc_report, report_key=dataset_folder_name)

        delivery_report = {
            "skill": "tdd-nnunet-export",
            "task": args.task,
            "request_signature": signature,
            "dataset_root_source": str(args.dataset_root.resolve()),
            "nnunet_root": str(nnunet_root),
            "dataset_folder": dataset_folder_name,
            "dataset_root": str(dataset_root),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "file_ending": ".png",
            "num_training_cases": len(train_ids),
            "num_test_cases": len(test_ids),
            "test_ratio": args.test_ratio,
            "threshold": args.threshold,
            "seed": args.seed,
            "qc_limit": qc_limit,
            "annotation_mode": task_cfg["annotation_mode"],
            "label_values_found": sorted(label_values_union),
            "is_label_space_valid": (
                sorted(label_values_union) == [0, 1]
                if task_cfg["annotation_mode"] == "binary_mask"
                else all(0 <= value <= 32 for value in label_values_union)
            ),
            "expected_label_values": [0, 1] if task_cfg["annotation_mode"] == "binary_mask" else list(range(33)),
            "skipped_labels": dict(sorted(skipped_labels_aggregate.items())),
            "empty_label_cases": empty_label_cases[:50],
            "image_modes": image_modes,
            "env": {
                "nnUNet_raw": str(nnunet_raw_root),
                "nnUNet_preprocessed": str(nnunet_preprocessed_root),
                "nnUNet_results": str(nnunet_results_root),
            },
            "commands": {
                "verify": f"export nnUNet_raw='{nnunet_raw_root}' nnUNet_preprocessed='{nnunet_preprocessed_root}' nnUNet_results='{nnunet_results_root}' && nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
                "train_example": f"export nnUNet_raw='{nnunet_raw_root}' nnUNet_preprocessed='{nnunet_preprocessed_root}' nnUNet_results='{nnunet_results_root}' && nnUNetv2_train {dataset_id} 2d all",
            },
            "size_distribution_sample": [
                {"size": list(size), "count": count}
                for size, count in sorted(size_sample.items(), key=lambda item: (-item[1], item[0]))
            ],
            "foreground_ratio_avg": (sum(foreground_ratios) / len(foreground_ratios)) if foreground_ratios else 0.0,
            "dataset_qc": {
                "status": "skipped" if args.skip_qc else qc_report["summary"]["dataset_status"],
                "report_json": None if args.skip_qc else qc_report_paths["report_json"],
                "report_md": None if args.skip_qc else qc_report_paths["report_md"],
                "case_count": None if args.skip_qc else qc_report["summary"]["case_count"],
                "sampled": False if args.skip_qc else (qc_limit > 0),
                "skip_requested": bool(args.skip_qc),
            },
            "exported_cases": exported_rows,
            "generated_at": utc_now_iso(),
        }
        report_json_path = report_root / f"nnUNet_delivery_{dataset_folder_name}.json"
        write_json(delivery_report, report_json_path)
        write_status(status_path, {
            "status": "completed",
            "stage": "completed",
            "task": args.task,
            "request_signature": signature,
            "dataset_root_source": str(args.dataset_root.resolve()),
            "dataset_root": str(dataset_root),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "delivery_report": str(report_json_path),
        })

        handoff_md = [
            "# nnUNet Delivery",
            "",
            f"- Task: `{args.task}`",
            f"- Dataset folder: `{dataset_folder_name}`",
            f"- nnUNet dataset id: `{dataset_id}`",
            f"- Request signature: `{signature}`",
            f"- nnUNet raw: `{nnunet_raw_root}`",
            f"- nnUNet preprocessed: `{nnunet_preprocessed_root}`",
            f"- nnUNet results: `{nnunet_results_root}`",
            f"- Delivery report: `{report_json_path}`",
            f"- Dataset QC report: `{None if args.skip_qc else qc_report_paths['report_json']}`",
            f"- Dataset QC summary: `{'skipped' if args.skip_qc else qc_report['summary']['dataset_status']}`",
            "",
            "## Example Commands",
            "",
            "```bash",
            f"export nnUNet_raw='{nnunet_raw_root}'",
            f"export nnUNet_preprocessed='{nnunet_preprocessed_root}'",
            f"export nnUNet_results='{nnunet_results_root}'",
            f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
            f"nnUNetv2_train {dataset_id} 2d all",
            "```",
        ]
        report_md_path = report_root / f"nnUNet_delivery_{dataset_folder_name}.md"
        report_md_path.write_text("\n".join(handoff_md) + "\n", encoding="utf-8")

        print(json.dumps(delivery_report, indent=2, ensure_ascii=False))
    finally:
        if lock_meta_path is not None:
            lock_meta_path.unlink(missing_ok=True)
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()


if __name__ == "__main__":
    main()
