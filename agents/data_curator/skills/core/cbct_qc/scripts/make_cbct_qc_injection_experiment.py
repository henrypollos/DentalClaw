#!/usr/bin/env python3
"""Create a sampled CBCT subset and inject controlled QC errors."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def case_ids_from_root(dataset_root: Path) -> List[str]:
    image_root = dataset_root / "imagesTr"
    label_root = dataset_root / "labelsTr"
    image_ids = {path.name.replace("_0000.nii.gz", "") for path in image_root.glob("*.nii.gz")}
    label_ids = {path.name.replace(".nii.gz", "") for path in label_root.glob("*.nii.gz")}
    return sorted(image_ids & label_ids)


def copy_subset(source_root: Path, output_root: Path, case_ids: List[str]) -> None:
    (output_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_root / "labelsTr").mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        shutil.copy2(source_root / "imagesTr" / f"{case_id}_0000.nii.gz", output_root / "imagesTr" / f"{case_id}_0000.nii.gz")
        shutil.copy2(source_root / "labelsTr" / f"{case_id}.nii.gz", output_root / "labelsTr" / f"{case_id}.nii.gz")

    dataset_json = read_json(source_root / "dataset.json")
    dataset_json["name"] = "{}_InjectedQCSubset".format(dataset_json.get("name", output_root.name))
    dataset_json["numTraining"] = len(case_ids)
    dataset_json["numTest"] = 0
    write_json(output_root / "dataset.json", dataset_json)


def load_nifti(path: Path):
    return nib.load(str(path))


def save_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path, *, affine=None, zooms=None, qform=None, sform=None) -> None:
    affine_to_use = np.array(affine if affine is not None else reference_img.affine, dtype=float)
    out = nib.Nifti1Image(data, affine_to_use, header=reference_img.header.copy())
    if zooms is not None:
        out.header.set_zooms(tuple(float(value) for value in zooms))
    if qform is not None:
        out.set_qform(np.array(qform, dtype=float), code=1)
    if sform is not None:
        out.set_sform(np.array(sform, dtype=float), code=1)
    nib.save(out, str(out_path))


def rescaled_affine(affine: np.ndarray, new_spacing: Tuple[float, float, float]) -> np.ndarray:
    result = np.array(affine, dtype=float, copy=True)
    for axis in range(3):
        direction = np.array(result[:3, axis], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = np.zeros(3, dtype=float)
            direction[axis] = 1.0
            norm = 1.0
        result[:3, axis] = direction / norm * float(new_spacing[axis])
    return result


def make_split_payload(case_ids: List[str], rng: random.Random) -> Dict[str, List[str]]:
    shuffled = list(case_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(round(n * 0.7))
    val_end = int(round(n * 0.85))
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def replace_case_id_in_splits(splits: Dict[str, List[str]], old_case_id: str, new_case_id: str) -> None:
    for split_name, case_ids in splits.items():
        updated = []
        for case_id in case_ids:
            if case_id == old_case_id:
                updated.append(new_case_id)
            else:
                updated.append(case_id)
        splits[split_name] = updated


def inject_missing_annotation(output_root: Path, case_id: str) -> Dict[str, object]:
    label_path = output_root / "labelsTr" / f"{case_id}.nii.gz"
    if label_path.exists():
        label_path.unlink()
    return {
        "category": "missing_annotation_volume",
        "case_ids": [case_id],
        "expected_findings_any": ["missing_annotation_volume"],
    }


def inject_shape_mismatch(output_root: Path, case_id: str) -> Dict[str, object]:
    label_path = output_root / "labelsTr" / f"{case_id}.nii.gz"
    label_img = load_nifti(label_path)
    data = np.asarray(label_img.dataobj)
    modified = data[..., :-1]
    save_nifti(modified, label_img, label_path, affine=label_img.affine, zooms=label_img.header.get_zooms()[: modified.ndim])
    return {
        "category": "image_label_shape_mismatch",
        "case_ids": [case_id],
        "expected_findings_any": ["image_label_shape_mismatch"],
    }


def inject_invalid_label_value(output_root: Path, case_id: str, invalid_value: int = 250) -> Dict[str, object]:
    label_path = output_root / "labelsTr" / f"{case_id}.nii.gz"
    label_img = load_nifti(label_path)
    data = np.asarray(label_img.dataobj).copy()
    data[:4, :4, :4] = invalid_value
    save_nifti(data, label_img, label_path, affine=label_img.affine, zooms=label_img.header.get_zooms()[: data.ndim])
    return {
        "category": "invalid_label_value",
        "case_ids": [case_id],
        "expected_findings_any": ["invalid_label_values"],
        "details": {"invalid_value": invalid_value},
    }


def inject_spacing_abnormality(output_root: Path, case_id: str, new_spacing: Tuple[float, float, float]) -> Dict[str, object]:
    image_path = output_root / "imagesTr" / f"{case_id}_0000.nii.gz"
    image_img = load_nifti(image_path)
    data = np.asarray(image_img.dataobj)
    affine = rescaled_affine(image_img.affine, new_spacing)
    save_nifti(
        data,
        image_img,
        image_path,
        affine=affine,
        zooms=new_spacing,
        qform=affine,
        sform=affine,
    )
    return {
        "category": "voxel_spacing_header_abnormality",
        "case_ids": [case_id],
        "expected_findings_any": [
            "invalid_spacing",
            "image_label_spacing_mismatch",
            "spacing_outlier_axis_0",
            "spacing_outlier_axis_1",
            "spacing_outlier_axis_2",
            "metadata_inconsistency",
        ],
        "details": {"new_spacing": list(new_spacing)},
    }


def inject_metadata_inconsistency(output_root: Path, case_id: str) -> Dict[str, object]:
    image_path = output_root / "imagesTr" / f"{case_id}_0000.nii.gz"
    image_img = load_nifti(image_path)
    data = np.asarray(image_img.dataobj)
    affine = np.array(image_img.affine, dtype=float)
    qform = np.array(affine, dtype=float, copy=True)
    qform[:3, 0] *= -1.0
    qform[0, 3] += abs(affine[0, 0]) * (image_img.shape[0] - 1)
    save_nifti(
        data,
        image_img,
        image_path,
        affine=affine,
        zooms=image_img.header.get_zooms()[: data.ndim],
        qform=qform,
        sform=affine,
    )
    return {
        "category": "metadata_inconsistency",
        "case_ids": [case_id],
        "expected_findings_any": ["metadata_inconsistency"],
    }


def inject_duplicate_and_split_leakage(output_root: Path, source_case_id: str, target_case_id: str, splits: Dict[str, List[str]]) -> Dict[str, object]:
    source_image = output_root / "imagesTr" / f"{source_case_id}_0000.nii.gz"
    source_label = output_root / "labelsTr" / f"{source_case_id}.nii.gz"
    target_image = output_root / "imagesTr" / f"{target_case_id}_0000.nii.gz"
    target_label = output_root / "labelsTr" / f"{target_case_id}.nii.gz"
    injected_case_id = f"{source_case_id}_copy"
    injected_image = output_root / "imagesTr" / f"{injected_case_id}_0000.nii.gz"
    injected_label = output_root / "labelsTr" / f"{injected_case_id}.nii.gz"

    if target_image.exists():
        target_image.unlink()
    if target_label.exists():
        target_label.unlink()
    shutil.copy2(source_image, injected_image)
    shutil.copy2(source_label, injected_label)
    replace_case_id_in_splits(splits, target_case_id, injected_case_id)

    for split_name in splits:
        splits[split_name] = [case_id for case_id in splits[split_name] if case_id != source_case_id and case_id != injected_case_id]
    splits["train"].append(source_case_id)
    splits["test"].append(injected_case_id)
    splits["train"] = sorted(set(splits["train"]))
    splits["test"] = sorted(set(splits["test"]))

    return {
        "category": "duplicate_identifier_split_leakage",
        "case_ids": [source_case_id, injected_case_id],
        "expected_findings_all": ["duplicate_case_identifier", "split_leakage"],
        "expected_findings_any": ["exact_duplicate_scan", "suspected_near_duplicate_scan"],
        "details": {"source_case_id": source_case_id, "replaced_case_id": target_case_id, "injected_case_id": injected_case_id},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sampled CBCT subset and inject QC errors.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--cases-per-error", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    rng = random.Random(args.seed)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    all_case_ids = case_ids_from_root(source_root)
    if len(all_case_ids) < args.sample_count:
        raise ValueError(f"Requested {args.sample_count} cases, but only found {len(all_case_ids)}.")

    sampled_case_ids = sorted(rng.sample(all_case_ids, args.sample_count))
    copy_subset(source_root, output_root, sampled_case_ids)

    splits = make_split_payload(sampled_case_ids, rng)
    worklist = list(sampled_case_ids)
    rng.shuffle(worklist)

    needed = args.cases_per_error * 5 + args.cases_per_error * 2
    if len(worklist) < needed:
        raise ValueError("Not enough sampled cases to allocate disjoint injection groups.")

    missing_cases = [worklist.pop() for _ in range(args.cases_per_error)]
    shape_cases = [worklist.pop() for _ in range(args.cases_per_error)]
    invalid_cases = [worklist.pop() for _ in range(args.cases_per_error)]
    spacing_cases = [worklist.pop() for _ in range(args.cases_per_error)]
    metadata_cases = [worklist.pop() for _ in range(args.cases_per_error)]
    duplicate_sources = [worklist.pop() for _ in range(args.cases_per_error)]
    duplicate_targets = [worklist.pop() for _ in range(args.cases_per_error)]

    injections: List[Dict[str, object]] = []

    for case_id in missing_cases:
        injections.append(inject_missing_annotation(output_root, case_id))
    for case_id in shape_cases:
        injections.append(inject_shape_mismatch(output_root, case_id))
    for case_id in invalid_cases:
        injections.append(inject_invalid_label_value(output_root, case_id))
    for case_id in spacing_cases:
        injections.append(inject_spacing_abnormality(output_root, case_id, new_spacing=(1.5, 0.3, 0.3)))
    for case_id in metadata_cases:
        injections.append(inject_metadata_inconsistency(output_root, case_id))
    for source_case_id, target_case_id in zip(duplicate_sources, duplicate_targets):
        injections.append(inject_duplicate_and_split_leakage(output_root, source_case_id, target_case_id, splits))

    split_payload = {"splits": {name: sorted(values) for name, values in splits.items()}}
    write_json(output_root / "splits_injected.json", split_payload)

    category_counter: Dict[str, int] = {}
    for record in injections:
        category = str(record["category"])
        category_counter[category] = category_counter.get(category, 0) + 1

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "sample_count": args.sample_count,
        "cases_per_error": args.cases_per_error,
        "sampled_case_ids": sampled_case_ids,
        "category_counts": category_counter,
        "injections": injections,
        "split_json": str((output_root / "splits_injected.json").resolve()),
    }
    write_json(output_root / "injection_manifest.json", manifest)
    print(json.dumps({"output_root": str(output_root), "category_counts": category_counter}, indent=2))


if __name__ == "__main__":
    main()
