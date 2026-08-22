#!/usr/bin/env python3
"""Validate a private 2D dental image package for the platform MVP.

The validator is intentionally lightweight: it checks package structure and
whether supervised training is allowed. It does not inspect pixel-level DICOM
metadata, because the current platform MVP only needs a reliable gate before
training is launched.
"""

from __future__ import annotations

import argparse
import json
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
    contract = _read_json(contract_path)
    image_exts = set(contract["accepted_image_extensions"])
    mask_exts = set(contract["accepted_mask_extensions"])
    image_layout, image_files = _find_images(data_root, image_exts)
    mask_layout, mask_files = _find_masks(data_root, mask_exts)

    image_ids = {_case_id(path) for path in image_files}
    mask_ids = {_case_id(path) for path in mask_files}
    paired_ids = sorted(image_ids & mask_ids)
    unmatched_images = sorted(image_ids - mask_ids)
    unmatched_masks = sorted(mask_ids - image_ids)

    checks = []
    checks.append(
        {
            "name": "image_count_positive",
            "passed": len(image_files) > 0,
            "detail": f"{len(image_files)} image files found.",
        }
    )
    checks.append(
        {
            "name": "label_count_positive",
            "passed": len(mask_files) > 0,
            "detail": f"{len(mask_files)} mask/label files found.",
        }
    )
    checks.append(
        {
            "name": "paired_case_count_positive",
            "passed": len(paired_ids) > 0,
            "detail": f"{len(paired_ids)} paired image-mask cases found.",
        }
    )

    if mode == "private_train":
        can_execute = len(image_files) > 0 and len(mask_files) > 0 and len(paired_ids) > 0
        terminal_policy = "allow_private_train" if can_execute else "stop_without_training"
    else:
        can_execute = len(image_files) > 0
        terminal_policy = "allow_qc_or_inference_only" if can_execute else "reject_missing_images"

    if mode == "private_train" and len(mask_files) == 0:
        reason = "Images were found, but no masks/labels were detected; supervised segmentation training must not start."
    elif can_execute:
        reason = "The package satisfies the minimum structure required for the requested mode."
    else:
        reason = "The package does not satisfy the minimum structure required for the requested mode."

    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "contract": _rel(contract_path),
        "data_root": _rel(data_root),
        "mode": mode,
        "image_layout": image_layout,
        "mask_layout": mask_layout,
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "paired_case_count": len(paired_ids),
        "sample_images": [_rel(path) for path in image_files[:10]],
        "sample_masks": [_rel(path) for path in mask_files[:10]],
        "unmatched_image_count": len(unmatched_images),
        "unmatched_mask_count": len(unmatched_masks),
        "unmatched_images_preview": unmatched_images[:20],
        "unmatched_masks_preview": unmatched_masks[:20],
        "checks": checks,
        "can_execute_requested_mode": can_execute,
        "terminal_policy": terminal_policy,
        "reason": reason,
    }


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
            "当前私有数据包可以进入数据登记/QC阶段，但不能进入监督训练阶段；原因是缺少可配对的 mask/label。平台必须停止训练并给出原因，避免伪造私有数据训练能力。"
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
