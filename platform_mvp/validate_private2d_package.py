#!/usr/bin/env python3
"""Validate binary 2D input contents and splits before the training adapter runs."""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONTRACT = THIS_DIR / "private_data_contract.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _norm_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = path.stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def _collect_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and _norm_suffix(path) in extensions:
            files.append(path)
    return sorted(files)


def _first_existing_dir(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists() and path.is_dir():
            return path
    return None


def _find_images(root: Path, image_exts: set[str]) -> tuple[str, list[Path]]:
    image_dir = _first_existing_dir(root, ["images", "imagesTr", "imagesTs", "image", "imgs"])
    if image_dir:
        return image_dir.name, _collect_files(image_dir, image_exts)
    return "flat_root", [p for p in sorted(root.iterdir()) if p.is_file() and _norm_suffix(p) in image_exts]


def _find_masks(root: Path, mask_exts: set[str]) -> tuple[str | None, list[Path]]:
    mask_dir = _first_existing_dir(root, ["masks", "labels", "labelsTr", "labelsTs", "mask", "label"])
    if mask_dir:
        return mask_dir.name, _collect_files(mask_dir, mask_exts)
    return None, []


def validate_package(data_root: Path, contract_path: Path, mode: str) -> dict[str, Any]:
    """Inspect the complete binary 2D package; never infer clinical provenance."""
    data_root = data_root.resolve()
    contract = _read_json(contract_path)
    image_exts = set(contract["accepted_image_extensions"])
    mask_exts = set(contract["accepted_mask_extensions"])
    issues, cases, layouts = [], [], {}
    def issue(code, detail):
        issues.append({"code": code, "detail": detail})
    if not data_root.is_dir():
        issue("missing_data_root", str(data_root))
    pairs = [("train", "imagesTr", "labelsTr"), ("val", "imagesVal", "labelsVal"),
             ("test", "imagesTs", "labelsTs")]
    if not any((data_root / im).is_dir() or (data_root / la).is_dir() for _, im, la in pairs):
        im = _first_existing_dir(data_root, ["images", "image", "imgs"])
        la = _first_existing_dir(data_root, ["masks", "labels", "mask", "label"])
        pairs = [("train", im.name if im else ".", la.name if la else "masks")]
    elif any((data_root / n).is_dir() for n in ["images", "masks"]):
        issue("ambiguous_layout", "Both canonical and imagesTr layouts are present.")
    # The adapter consumes directory splits; do not silently ignore a separate split contract.
    if (data_root / "splits.json").exists():
        issue("unsupported_split_file", "Resolve splits.json into explicit split directories before training.")
    metadata = {}
    if (data_root / "metadata.json").exists():
        try:
            metadata = _read_json(data_root / "metadata.json")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
        except (OSError, ValueError) as exc:
            issue("invalid_metadata", str(exc))
            metadata = {}
    patient_ids = metadata.get("patient_ids", {})
    if not isinstance(patient_ids, dict):
        issue("invalid_patient_mapping", "patient_ids must map split/case_id to patient ID")
        patient_ids = {}
    seen_pixels, seen_patients = defaultdict(list), defaultdict(set)
    totals = {"image_count": 0, "mask_count": 0, "paired_case_count": 0,
              "unmatched_image_count": 0, "unmatched_mask_count": 0}
    split_counts = {}
    raster = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    allowed_labels = set(contract.get("allowed_label_values", [0, 1]))
    all_images, all_masks = [], []
    for split, im_name, la_name in pairs:
        imdir, ladir = data_root / im_name, data_root / la_name
        images = (_collect_files(imdir, image_exts) if im_name != "." else
                  sorted(p for p in data_root.iterdir() if p.is_file() and _norm_suffix(p) in image_exts)) if imdir.is_dir() else []
        masks = _collect_files(ladir, mask_exts) if ladir.is_dir() else []
        # Unknown data files cannot be silently skipped by the gate.
        for folder, accepted in [(imdir, image_exts), (ladir, mask_exts)]:
            if folder.is_dir() and folder != data_root:
                for f in folder.rglob("*"):
                    if f.is_file() and _norm_suffix(f) not in accepted and f.suffix.lower() not in {".json", ".txt", ".xml"}:
                        issue("unsupported_file", str(f.relative_to(data_root)))
        if not images and not masks:
            continue
        layouts[split] = {"images": im_name, "masks": la_name}
        indexed = []
        for files in (images, masks):
            lookup = defaultdict(list)
            for f in files:
                lookup[_case_id(f)].append(f)
            if any(len(v) > 1 for v in lookup.values()):
                issue("duplicate_case_id", split)
            indexed.append({k: v[0] for k, v in lookup.items()})
        ims, mas = indexed
        paired = set(ims) & set(mas)
        missing, orphan = set(ims) - set(mas), set(mas) - set(ims)
        counts = dict(image_count=len(images), mask_count=len(masks), paired_case_count=len(paired),
                      unmatched_image_count=len(missing), unmatched_mask_count=len(orphan))
        split_counts[split] = counts
        for k, v in counts.items(): totals[k] += v
        all_images.extend(images); all_masks.extend(masks)
        # Test images without any labels are allowed for prediction only, never metrics.
        require_labels = mode == "private_train" and (split in {"train", "val"} or bool(masks))
        if require_labels and missing:
            issue("missing_masks" if not masks else "incomplete_pairing", split)
        if orphan:
            issue("orphan_masks", split)
        decoded = {}
        for is_mask, files in [(False, images), (True, masks)]:
            for f in files:
                record = {"split": split, "case_id": _case_id(f), "kind": "mask" if is_mask else "image",
                          "path": str(f.relative_to(data_root))}
                try:
                    record["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
                    if _norm_suffix(f) not in raster:
                        issue("unsupported_2d_format", record["path"])
                        continue
                    with Image.open(f) as img:
                        if getattr(img, "n_frames", 1) != 1:
                            raise ValueError("Multi-frame input is not a single 2D image")
                        arr = np.asarray(img if is_mask else img.convert("L"))
                    if arr.ndim != 2:
                        issue("invalid_mask_dimension" if is_mask else "invalid_image_dimension", record["path"])
                        continue
                    record["shape"] = list(arr.shape)
                    decoded[(is_mask, _case_id(f))] = arr.shape
                    if is_mask:
                        vals = np.unique(arr).tolist()
                        record["label_values"] = vals
                        if not set(vals) <= allowed_labels:
                            issue("invalid_label_values", record["path"])
                    else:
                        fingerprint = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
                        record["pixel_sha256"] = fingerprint
                        seen_pixels[fingerprint].append((split, _case_id(f)))
                        pid = patient_ids.get(split + "/" + _case_id(f))
                        if pid is not None:
                            if not isinstance(pid, (str, int)) or str(pid) == "":
                                issue("invalid_patient_id", record["path"])
                            else:
                                seen_patients[str(pid)].add(split)
                except (OSError, ValueError) as exc:
                    issue("unreadable_file", record["path"] + ": " + str(exc))
                finally:
                    cases.append(record)
        for cid in paired:
            if decoded.get((False, cid)) != decoded.get((True, cid)):
                issue("image_mask_shape_mismatch", split + "/" + cid)
            # The current built-in trainer pairs exact filenames, not normalized stems.
            if mode == "private_train" and ims[cid].name != mas[cid].name:
                issue("backend_filename_mismatch", split + "/" + cid)
    if not totals["image_count"]:
        issue("missing_images", "No readable-layout images found")
    if mode == "private_train" and not split_counts.get("train", {}).get("paired_case_count"):
        issue("missing_training_pairs", "Training requires image-mask pairs in the training split")
    if any(len({s for s, _ in v}) > 1 for v in seen_pixels.values()):
        issue("cross_split_image_duplicate", "Identical decoded pixels occur in different splits")
    if any(len(v) > 1 for v in seen_patients.values()):
        issue("cross_split_patient_duplicate", "A supplied patient ID occurs in different splits")
    image_records = [x for x in cases if x["kind"] == "image"]
    complete_patient_mapping = bool(image_records) and all(
        str(x["split"]) + "/" + str(x["case_id"]) in patient_ids for x in image_records)
    manifest = {"schema_version": 1, "data_root": str(data_root), "dataset_source": "user_supplied",
                "declared_provenance": metadata.get("provenance"), "expected_modality": "2d",
                "annotation_contract": "binary_pixel_mask_0_1", "layouts": layouts, "files": cases,
                "patient_split_check": "checked_supplied_ids" if complete_patient_mapping else "not_established",
                "test_metrics_available": bool(split_counts.get("test", {}).get("paired_case_count"))}
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    ready = not issues
    return {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "contract": _rel(contract_path), "data_root": str(data_root), "mode": mode,
            "image_layout": layouts.get("train", {}).get("images"),
            "mask_layout": layouts.get("train", {}).get("masks"), **totals,
            "sample_images": [_rel(p) for p in all_images[:10]], "sample_masks": [_rel(p) for p in all_masks[:10]],
            "checks": [{"name": "binary_2d_package_contract", "passed": ready, "detail": ", ".join(sorted({x['code'] for x in issues})) or "All inspected checks passed"}],
            "issues": issues, "reason_codes": sorted({x["code"] for x in issues}),
            "manifest": manifest, "manifest_id": manifest_id, "split_counts": split_counts,
            "can_execute_requested_mode": ready,
            "terminal_policy": ("allow_private_train" if mode == "private_train" else "allow_qc_or_inference_only") if ready else "stop_without_training",
            "reason": "Validated binary 2D input contract; patient independence only if supplied IDs checked." if ready else "Input contract failed; see reason_codes."}


def build_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Private2D 输入包预检查报告",
        "",
        "## 1. 检查对象",
        "",
        f"- Data root: `{report['data_root']}`",
        f"- Requested mode: `{report['mode']}`",
        f"- Contract: `{report['contract']}`",
        "",
        "## 2. 数据结构摘要",
        "",
        f"- Image layout: `{report['image_layout']}`",
        f"- Mask layout: `{report['mask_layout']}`",
        f"- Image count: `{report['image_count']}`",
        f"- Mask count: `{report['mask_count']}`",
        f"- Paired case count: `{report['paired_case_count']}`",
        "",
        "## 3. 最小检查",
        "",
        "| Check | Passed | Detail |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {check['detail']} |")

    lines += [
        "",
        "## 4. 平台决策",
        "",
        f"- Can execute requested mode: `{report['can_execute_requested_mode']}`",
        f"- Terminal policy: `{report['terminal_policy']}`",
        f"- Reason: {report['reason']}",
        "",
        "## 5. 汇报口径",
        "",
    ]
    if report["mode"] == "private_train" and not report["can_execute_requested_mode"]:
        lines.append(
            "当前输入包未通过兼容性检查；请按 issues/reason_codes 修正后再训练。"
        )
    elif report["can_execute_requested_mode"]:
        lines.append(
            "当前私有数据包满足请求模式的最小结构要求，可以进入下一步 adapter 执行。"
        )
    else:
        lines.append(
            "当前私有数据包不满足请求模式的最小结构要求，平台应拒绝执行并提示补齐输入。"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a private 2D dental image package.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--mode", choices=["private_train", "inference_or_qc"], default="private_train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root.exists() or not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    report = validate_package(args.data_root.resolve(), args.contract.resolve(), args.mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "private2d_validation_report.json", report)
    (args.out_dir / "private2d_validation_report.md").write_text(
        build_report_md(report),
        encoding="utf-8",
    )
    print("Private2D package validation completed.")
    print(f"Report JSON: {_rel(args.out_dir / 'private2d_validation_report.json')}")
    print(f"Report MD: {_rel(args.out_dir / 'private2d_validation_report.md')}")
    print(f"Decision: {report['terminal_policy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
