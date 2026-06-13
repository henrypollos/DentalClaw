# agents/clinical_result/skills/postprocess.py

import numpy as np
from scipy import ndimage as ndi


def geometric_postprocess(mask, min_size=20):
    """
    多类别几何后处理：
    - 按类别去除小连通域
    - 保留原始标签值
    """
    mask = np.asarray(mask).astype(np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)

    for cls in np.unique(mask):
        if cls == 0:
            continue

        cls_mask = mask == cls
        labeled, num = ndi.label(cls_mask)

        keep = np.zeros_like(cls_mask, dtype=bool)
        for comp_id in range(1, num + 1):
            comp = labeled == comp_id
            if comp.sum() >= min_size:
                keep |= comp

        out[keep] = cls

    return {
        "mask": out,
        "meta": {
            "stage": "geometric_postprocess",
            "min_size": min_size,
        }
    }