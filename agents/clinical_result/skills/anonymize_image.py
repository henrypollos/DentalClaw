from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import pydicom  # type: ignore
except Exception:
    pydicom = None


def _apply_black_boxes(img: np.ndarray, boxes: Sequence[Tuple[float, float, float, float]]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        xa = max(0, min(w, int(round(w * x1))))
        ya = max(0, min(h, int(round(h * y1))))
        xb = max(0, min(w, int(round(w * x2))))
        yb = max(0, min(h, int(round(h * y2))))
        if xa < xb and ya < yb:
            out[ya:yb, xa:xb] = 0
    return out


def anonymize_png_jpg(
    in_path: str,
    out_path: str,
    *,
    boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> str:
    """
    适用于 PNG/JPG 的匿名化：
    - 统一遮掉常见角落文字
    - 保留图像主体
    """
    src = Path(in_path)
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {in_path}")

    if boxes is None:
        boxes = (
            (0.00, 0.00, 0.24, 0.11),  # 左上角
            (0.76, 0.00, 1.00, 0.11),  # 右上角
            (0.00, 0.90, 1.00, 1.00),  # 底部条
        )

    img = _apply_black_boxes(img, boxes)
    cv2.imwrite(str(dst), img)
    return str(dst)


def anonymize_dicom(in_path: str, out_path: str) -> str:
    if pydicom is None:
        raise RuntimeError("pydicom is not installed, cannot anonymize DICOM.")

    src = Path(in_path)
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ds = pydicom.dcmread(str(src))

    sensitive_tags = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "AccessionNumber",
        "StudyDate",
        "StudyTime",
    ]
    for tag in sensitive_tags:
        try:
            if tag in ds:
                ds.data_element(tag).value = "ANONYMIZED"
        except Exception:
            pass

    try:
        ds.remove_private_tags()
    except Exception:
        pass

    ds.save_as(str(dst))
    return str(dst)


def anonymize_image(in_path: str, out_path: str) -> str:
    suffix = Path(in_path).suffix.lower()
    if suffix == ".dcm":
        return anonymize_dicom(in_path, out_path)
    return anonymize_png_jpg(in_path, out_path)


def anonymize_case_image(case: dict, out_dir: str) -> dict:
    """
    返回一个浅拷贝 case：
    - image_path 改成匿名化副本
    - original_image_path 保留原图路径，写进 metadata
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_path = case.get("image_path")
    if not image_path:
        return dict(case)

    src = Path(image_path)
    stem = src.stem

    if stem.endswith("_0000"):
        base = stem[:-5]  # 去掉 _0000
        anon_name = base + "_anon_0000" + src.suffix
    else:
        anon_name = stem + "_anon" + src.suffix
    anon_path = out_path / anon_name
    anonymized = anonymize_image(str(src), str(anon_path))

    new_case = dict(case)
    new_case["original_image_path"] = str(src)
    new_case["image_path"] = anonymized
    new_case["anonymized_image_path"] = anonymized
    new_case["anonymized"] = True
    return new_case