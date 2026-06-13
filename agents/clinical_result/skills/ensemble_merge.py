# agents/clinical_result/skills/ensemble_merge.py

import os
import cv2
import numpy as np


def ensemble_merge(mask_dirs, filename):
    """
    对多个模型输出的同名 mask 做多数投票融合。
    适合当前这种“硬标签 PNG mask”输出。
    """
    if not mask_dirs:
        raise ValueError("mask_dirs 不能为空")

    masks = []
    for mask_dir in mask_dirs:
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        masks.append(mask.astype(np.uint8))

    stacked = np.stack(masks, axis=0)  # [N, H, W]

    # 多类别投票：按像素统计每个类别出现次数
    max_label = int(stacked.max())
    votes = np.zeros((max_label + 1,) + stacked.shape[1:], dtype=np.uint16)
    for cls in range(max_label + 1):
        votes[cls] = (stacked == cls).sum(axis=0)

    merged = votes.argmax(axis=0).astype(np.uint8)

    return {
        "mask": merged,
        "meta": {
            "stage": "ensemble_merge",
            "num_models": len(mask_dirs),
            "filename": filename,
        }
    }