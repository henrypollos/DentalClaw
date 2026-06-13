# agents/clinical_result/skills/tta_inference.py

from pathlib import Path
import shutil
import cv2
import numpy as np
from scipy import stats

from .model_inference import model_inference


def _save_transformed_image(src_path: str, dst_path: str, transform: str):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src_path)

    if transform == "orig":
        out = img
    elif transform == "hflip":
        out = cv2.flip(img, 1)
    elif transform == "vflip":
        out = cv2.flip(img, 0)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    cv2.imwrite(dst_path, out)


def _invert_mask(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "orig":
        return mask
    if transform == "hflip":
        return np.fliplr(mask)
    if transform == "vflip":
        return np.flipud(mask)
    raise ValueError(f"Unknown transform: {transform}")


def tta_inference(case, model_path, work_dir, transforms=None):
    """
    对单个模型做 TTA：
    - 原图
    - 水平翻转
    - 垂直翻转
    最终用多数投票融合
    """
    transforms = transforms or ["orig", "hflip", "vflip"]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(case["image_path"])
    mask_list = []

    for tf in transforms:
        tf_dir = work_dir / tf
        tf_dir.mkdir(parents=True, exist_ok=True)

        # 1) 生成变换后的图像
        tf_image = tf_dir / image_path.name
        _save_transformed_image(str(image_path), str(tf_image), tf)

        # 2) 构造一个临时 case
        tf_case = dict(case)
        tf_case["image_path"] = str(tf_image)

        # 3) 调用你现有的单模型推理
        result = model_inference(tf_case, model_path, tf_dir)

        # 4) 读出 mask
        mask_dir = Path(result["mask_dir"])
        mask_path = mask_dir / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        # 5) 反变换回原坐标
        mask = _invert_mask(mask, tf)
        mask_list.append(mask)

    # 6) 多数投票融合
    stacked = np.stack(mask_list, axis=0)  # [T, H, W]
    merged = stats.mode(stacked, axis=0, keepdims=False).mode.astype(np.uint8)

    return {
        "mask": merged,
        "meta": {
            "stage": "tta_inference",
            "num_views": len(transforms),
            "transforms": transforms,
        }
    }