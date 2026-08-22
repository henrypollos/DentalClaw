#!/usr/bin/env python3
"""
T6: 给注册表中 6 条已验证方法显式添加方法学适当性规格字段。

新字段（仅作元数据声明，路由决策逻辑不读取，不改变任何决策行为）：
- required_annotation_type : 该方法要求/适配的标注类型（pixel_mask / bbox / case-label / none）
- evaluation_unit          : 评估单位（image / tooth / lesion / case）
- evaluation_metrics       : 该任务类别的预设指标
- required_qc_gates        : 执行前必须通过的 QC 检查
- contraindications        : 禁忌（请求与该方法不适配的情形）

auto_registered 的 planned_adapter 条目不加字段（尚未经过人工验证，不构成"已验证规格"）。

用法: python platform_mvp/add_appropriateness_specs.py
"""
import json
from pathlib import Path

REG = Path(__file__).resolve().parent / "method_registry.json"

SPECS = {
    "tdd_2d_segmentation_infer_report": {
        "required_annotation_type": "pixel_mask",
        "evaluation_unit": "image",
        "evaluation_metrics": ["mean_dice", "iou"],
        "required_qc_gates": ["image_completeness", "annotation_integrity", "split_leakage_check"],
        "contraindications": ["bbox_only_annotations", "case_level_labels", "3d_volumes", "training_requests"],
    },
    "private_2d_segmentation_train": {
        "required_annotation_type": "pixel_mask",
        "evaluation_unit": "image",
        "evaluation_metrics": ["mean_dice", "iou"],
        "required_qc_gates": ["image_completeness", "annotation_integrity", "annotation_completeness", "split_leakage_check"],
        "contraindications": ["bbox_only_annotations", "case_level_labels", "unpaired_images_labels"],
    },
    "toothfairy3_3d_segmentation_infer_or_train": {
        "required_annotation_type": "pixel_mask",
        "evaluation_unit": "image",
        "evaluation_metrics": ["mean_dice", "mean_hd95", "iou"],
        "required_qc_gates": ["volume_completeness", "label_value_validation", "metal_artifact_check", "fov_check", "split_leakage_check"],
        "contraindications": ["bbox_only_annotations", "case_level_labels", "2d_images"],
    },
    "dental_anomaly_detection": {
        "required_annotation_type": "image_level_label_optional",
        "evaluation_unit": "image",
        "evaluation_metrics": ["image_level_flag_summary"],
        "required_qc_gates": ["image_completeness", "duplicate_check"],
        "contraindications": ["pixel_mask_required_tasks", "3d_volumes", "case_level_diagnosis"],
    },
    "dental_2d_super_resolution": {
        "required_annotation_type": "none_required",
        "evaluation_unit": "image",
        "evaluation_metrics": ["psnr", "ssim"],
        "required_qc_gates": ["image_completeness", "resolution_check"],
        "contraindications": ["3d_volumes", "segmentation_requests", "mask_required_tasks"],
    },
    "tta_ensemble_inference": {
        "required_annotation_type": "pixel_mask",
        "evaluation_unit": "image",
        "evaluation_metrics": ["mean_dice", "iou"],
        "required_qc_gates": ["image_completeness", "model_availability", "split_leakage_check"],
        "contraindications": ["training_requests", "3d_volumes", "bbox_only_annotations"],
    },
}


def main() -> None:
    data = json.loads(REG.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in data["methods"]}
    added, skipped = [], []
    for mid, spec in SPECS.items():
        m = by_id.get(mid)
        if m is None:
            skipped.append(mid)
            continue
        # 已存在则不覆盖（幂等）
        if "required_annotation_type" in m:
            skipped.append(mid)
            continue
        m.update(spec)
        added.append(mid)

    tmp = REG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(REG)
    print(f"added specs to {len(added)} methods: {added}")
    if skipped:
        print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
