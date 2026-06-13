import os
import cv2
import numpy as np


def base_inference(case, model=None):
    """
    不做模型推理，直接读取已有预测结果
    """

    # ⭐ 你的 predictions 目录
    pred_dir = "/data/data2/yiyang/DentalClaw/artifacts/results/verify_run/predictions"

    filename = case["filename"]   # 比如 teeth_0001.png
    pred_path = os.path.join(pred_dir, filename)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction not found: {pred_path}")

    # 读取 mask（灰度图）
    mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to read image: {pred_path}")

    # 归一化（可选）
    mask = mask.astype(np.float32) / 255.0

    return {
        "mask": mask,
        "meta": {
            "source": "precomputed_predictions",
            "file": filename
        }
    }
