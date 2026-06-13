#!/usr/bin/env python3
"""Materialize an nnUNet dataset that keeps only QC-approved cases."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[6]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_output_roots(output_root: Path) -> Tuple[Path, Path]:
    resolved = output_root.resolve()
    if resolved.name == "nnUNet_raw":
        nnunet_raw_root = ensure_dir(resolved)
        nnunet_root = ensure_dir(resolved.parent)
        return nnunet_root, nnunet_raw_root
    nnunet_root = ensure_dir(resolved)
    nnunet_raw_root = ensure_dir(nnunet_root / "nnUNet_raw")
    return nnunet_root, nnunet_raw_root


def sanitize_dataset_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "", name)
    return value or "Dataset"


def parse_dataset_id(folder_name: str) -> Optional[int]:
    match = re.match(r"Dataset(\d{3})_", folder_name)
    return int(match.group(1)) if match else None


def find_next_dataset_id(nnunet_raw_root: Path, minimum: int = 501) -> int:
    existing: List[int] = []
    if nnunet_raw_root.exists():
        for path in nnunet_raw_root.iterdir():
            if not path.is_dir():
                continue
            dataset_id = parse_dataset_id(path.name)
            if dataset_id is not None:
                existing.append(dataset_id)
    if not existing:
        return minimum
    return max(max(existing) + 1, minimum)


def split_dir_names(split_name: str) -> Tuple[str, str]:
    normalized = str(split_name).strip().lower()
    if normalized == "train":
        return "imagesTr", "labelsTr"
    if normalized == "val":
        return "imagesVal", "labelsVal"
    if normalized == "test":
        return "imagesTs", "labelsTs"
    raise ValueError(f"Unsupported split name: {split_name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(path: Path, payload: dict) -> None:
    write_json(path, {"updated_at": utc_now_iso(), **payload})


def split_lookup_from_report(report: dict) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    split_payload = report.get("split_summary") or {}
    split_map = split_payload.get("splits", split_payload)
    if isinstance(split_map, dict):
        for split_name, case_ids in split_map.items():
            for case_id in case_ids or []:
                lookup[str(case_id)] = str(split_name)
    return lookup


def infer_split_lookup_from_dataset(dataset_root: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for split_name, image_dir_name in (("train", "imagesTr"), ("val", "imagesVal"), ("test", "imagesTs")):
        image_dir = dataset_root / image_dir_name
        if not image_dir.is_dir():
            continue
        for path in image_dir.glob("*_*.png"):
            case_id = re.sub(r"_\d{4}$", "", path.stem)
            lookup[case_id] = split_name
    return lookup


def selected_case_records(report: dict, include_statuses: Sequence[str]) -> List[dict]:
    allowed = {value.strip().lower() for value in include_statuses if value.strip()}
    selected = []
    for case in report.get("cases", []):
        status = str(case.get("status", "")).strip().lower()
        if status in allowed:
            selected.append(case)
    return selected


def collect_image_files(image_dir: Path, case_id: str) -> List[Path]:
    return sorted(image_dir.glob(f"{case_id}_*.*"))


def copy_case_assets(source_dataset_root: Path, target_dataset_root: Path, case_id: str, split_name: str) -> Dict[str, List[str]]:
    images_dir_name, labels_dir_name = split_dir_names(split_name)
    source_image_dir = source_dataset_root / images_dir_name
    source_label_dir = source_dataset_root / labels_dir_name
    target_image_dir = ensure_dir(target_dataset_root / images_dir_name)
    target_label_dir = ensure_dir(target_dataset_root / labels_dir_name)

    copied_images: List[str] = []
    copied_labels: List[str] = []
    for image_path in collect_image_files(source_image_dir, case_id):
        target_path = target_image_dir / image_path.name
        shutil.copy2(image_path, target_path)
        copied_images.append(str(target_path))

    for label_path in sorted(source_label_dir.glob(f"{case_id}.*")):
        target_path = target_label_dir / label_path.name
        shutil.copy2(label_path, target_path)
        copied_labels.append(str(target_path))

    if not copied_images:
        raise FileNotFoundError(f"No image files found for case {case_id} in split {split_name}.")
    if not copied_labels:
        raise FileNotFoundError(f"No label files found for case {case_id} in split {split_name}.")
    return {"images": copied_images, "labels": copied_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an nnUNet dataset that keeps only selected QC case statuses.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source nnUNet raw dataset root.")
    parser.add_argument("--qc-report", type=Path, required=True, help="QC JSON report generated for the source dataset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "datasets" / "nnUNet",
        help="Stable nnUNet artifact root (`.../artifacts/datasets/nnUNet`) or `.../nnUNet_raw`.",
    )
    parser.add_argument("--dataset-id", type=int, default=None, help="Override derived nnUNet dataset id.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override derived dataset name suffix.")
    parser.add_argument(
        "--include-status",
        action="append",
        default=None,
        help="QC case status to keep. Repeatable. Defaults to only `ready`.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dataset_root = args.dataset_root.resolve()
    qc_report_path = args.qc_report.resolve()
    if not source_dataset_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_dataset_root}")
    if not qc_report_path.is_file():
        raise FileNotFoundError(f"QC report does not exist: {qc_report_path}")

    nnunet_root, nnunet_raw_root = resolve_output_roots(args.output_root)
    report = load_json(qc_report_path)
    source_dataset_json = load_json(source_dataset_root / "dataset.json")
    include_statuses = args.include_status or ["ready"]
    selected_cases = selected_case_records(report, include_statuses)
    split_lookup = split_lookup_from_report(report) or infer_split_lookup_from_dataset(source_dataset_root)

    source_folder_name = source_dataset_root.name
    source_dataset_id = parse_dataset_id(source_folder_name)
    source_suffix = source_folder_name.split("_", 1)[1] if "_" in source_folder_name else source_folder_name
    derived_dataset_name = sanitize_dataset_name(args.dataset_name or f"{source_suffix}ReadyOnly")
    derived_dataset_id = (
        args.dataset_id
        if args.dataset_id is not None
        else find_next_dataset_id(nnunet_raw_root, minimum=(source_dataset_id + 1) if source_dataset_id is not None else 501)
    )
    derived_folder_name = f"Dataset{derived_dataset_id:03d}_{derived_dataset_name}"
    derived_dataset_root = nnunet_raw_root / derived_folder_name
    status_path = nnunet_root / f"nnUNet_delivery_{derived_folder_name}.status.json"
    report_json_path = nnunet_root / f"nnUNet_delivery_{derived_folder_name}.json"
    report_md_path = nnunet_root / f"nnUNet_delivery_{derived_folder_name}.md"

    if derived_dataset_root.exists():
        if not args.overwrite:
            print(json.dumps({
                "status": "already_exists",
                "dataset_root": str(derived_dataset_root),
                "dataset_id": derived_dataset_id,
                "dataset_name": derived_dataset_name,
                "delivery_report": str(report_json_path) if report_json_path.is_file() else None,
                "delivery_status": str(status_path) if status_path.is_file() else None,
            }, indent=2, ensure_ascii=False))
            return
        shutil.rmtree(derived_dataset_root)

    if not selected_cases:
        raise ValueError("QC filtering selected zero cases.")

    write_status(status_path, {
        "status": "running",
        "stage": "selecting_cases",
        "source_dataset_root": str(source_dataset_root),
        "source_qc_report": str(qc_report_path),
        "dataset_root": str(derived_dataset_root),
        "dataset_id": derived_dataset_id,
        "dataset_name": derived_dataset_name,
    })

    copied_manifest: List[dict] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    case_ids_by_split = {"train": [], "val": [], "test": []}
    derived_dataset_root.mkdir(parents=True, exist_ok=True)

    for case in selected_cases:
        case_id = str(case.get("case_id"))
        split_name = split_lookup.get(case_id)
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Could not determine split for QC-selected case: {case_id}")
        copied = copy_case_assets(source_dataset_root, derived_dataset_root, case_id, split_name)
        split_counts[split_name] += 1
        case_ids_by_split[split_name].append(case_id)
        copied_manifest.append({
            "case_id": case_id,
            "status": case.get("status"),
            "split": split_name,
            "images": copied["images"],
            "labels": copied["labels"],
        })

    if split_counts["train"] <= 0:
        raise ValueError("QC-selected subset does not contain any ready training cases.")

    derived_dataset_json = dict(source_dataset_json)
    derived_dataset_json["name"] = derived_dataset_name
    derived_dataset_json["numTraining"] = split_counts["train"]
    description = str(source_dataset_json.get("description", "")).strip()
    suffix = f" Filtered to QC statuses {sorted({status.lower() for status in include_statuses})}."
    derived_dataset_json["description"] = (description + suffix).strip()
    write_json(derived_dataset_root / "dataset.json", derived_dataset_json)

    subset_report = {
        "skill": "tdd-nnunet-ready-subset",
        "generated_at": utc_now_iso(),
        "status": "completed",
        "source_dataset_root": str(source_dataset_root),
        "source_dataset_id": source_dataset_id,
        "source_qc_report": str(qc_report_path),
        "selected_statuses": sorted({status.lower() for status in include_statuses}),
        "source_dataset_qc_status": report.get("summary", {}).get("dataset_status"),
        "source_case_count": report.get("summary", {}).get("case_count"),
        "source_ready_case_count": report.get("summary", {}).get("ready_case_count"),
        "source_manual_review_case_count": report.get("summary", {}).get("manual_review_case_count"),
        "source_blocked_case_count": report.get("summary", {}).get("blocked_case_count"),
        "dataset_root": str(derived_dataset_root),
        "dataset_folder": derived_folder_name,
        "dataset_id": derived_dataset_id,
        "dataset_name": derived_dataset_name,
        "counts": split_counts,
        "case_ids_by_split": {key: sorted(value) for key, value in case_ids_by_split.items()},
        "selected_case_count": len(copied_manifest),
        "copied_cases": copied_manifest,
    }
    write_json(report_json_path, subset_report)
    write_status(status_path, {
        "status": "completed",
        "stage": "completed",
        "source_dataset_root": str(source_dataset_root),
        "source_qc_report": str(qc_report_path),
        "dataset_root": str(derived_dataset_root),
        "dataset_id": derived_dataset_id,
        "dataset_name": derived_dataset_name,
        "delivery_report": str(report_json_path),
    })

    report_md_lines = [
        "# QC Ready-Only nnUNet Subset",
        "",
        f"- Source dataset: `{source_dataset_root}`",
        f"- Source QC report: `{qc_report_path}`",
        f"- Selected statuses: `{', '.join(sorted({status.lower() for status in include_statuses}))}`",
        f"- Derived dataset folder: `{derived_folder_name}`",
        f"- Derived dataset root: `{derived_dataset_root}`",
        f"- Derived dataset id: `{derived_dataset_id}`",
        f"- Train cases: `{split_counts['train']}`",
        f"- Val cases: `{split_counts['val']}`",
        f"- Test cases: `{split_counts['test']}`",
        f"- Delivery report: `{report_json_path}`",
    ]
    report_md_path.write_text("\n".join(report_md_lines) + "\n", encoding="utf-8")
    print(json.dumps(subset_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
