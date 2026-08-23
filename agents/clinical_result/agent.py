from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from .skills.report import clinical_report_export

ARTIFACT_ROOT = Path("$DENTALCLAW_HOME/artifacts")
ARTIFACT_REPORT_ROOT = ARTIFACT_ROOT / "results" / "reports"
CLINICAL_WORKSPACE_ROOT = ARTIFACT_REPORT_ROOT / "dentalclaw_result_workspace"


def _ensure_under_artifacts(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must live under {ARTIFACT_ROOT}, got {resolved}") from exc
    return resolved


# ==================== 模型路径解析与 folds 检测 ====================
def resolve_model_and_trainer(dataset_name: str = None) -> Path:
    """
    返回默认模型的 trainer 文件夹路径。
    优先从 artifacts 中查找，否则使用 JoD 的固定路径。
    """
    artifacts_root = Path("$DENTALCLAW_HOME/artifacts/models/nnUNet/nnUNet_results")
    default_trainer = Path(
        "$NNUNET_HOME/nnUNet/nnUNet_results/"
        "Dataset106_Teeth32_Labelbox/"
        "nnUNetTrainer__nnUNetPlans__2d"
    )

    if dataset_name and artifacts_root.exists():
        datasets = [p for p in artifacts_root.iterdir() if p.is_dir()]
        for d in datasets:
            if dataset_name in d.name:
                trainer = d / "nnUNetTrainer__nnUNetPlans__2d"
                if trainer.exists():
                    return trainer
        raise ValueError(f"Dataset {dataset_name} not found in artifacts")

    if default_trainer.exists():
        return default_trainer

    raise FileNotFoundError("No valid model found (artifacts + default both missing)")


def detect_available_folds(trainer_folder: Path) -> Tuple[int, ...]:
    """检测 trainer 文件夹下存在的 fold 编号"""
    folds = []
    for p in trainer_folder.iterdir():
        if p.is_dir() and p.name.startswith("fold_"):
            try:
                folds.append(int(p.name.split("_")[1]))
            except Exception:
                continue
    if not folds:
        raise RuntimeError(f"No folds found in {trainer_folder}")
    return tuple(sorted(folds))


# ==================== 辅助函数 ====================
def _parse_folds(raw: Any) -> Tuple[int, ...]:
    """解析 folds 配置，支持 tuple/list/string/int"""
    if raw is None:
        return (0,)
    if isinstance(raw, (tuple, list)):
        folds = []
        for x in raw:
            if str(x).strip().lower() == "all":
                continue
            folds.append(int(x))
        return tuple(folds) if folds else (0,)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return (0,)
        if raw.lower() == "all":
            return (0, 1, 2, 3, 4)
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return (int(raw),)


def _case_base_name(image_path: str) -> str:
    stem = Path(image_path).stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def _build_predictor(
    trainer_folder: str,
    use_folds: Sequence[int],
    checkpoint_name: str,
    use_tta: bool,
) -> nnUNetPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        trainer_folder,
        use_folds=tuple(use_folds),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def _predict_single_case(
    image_path: str,
    trainer_folder: str,
    output_dir: str,
    use_folds: Sequence[int] = (0,),
    checkpoint_name: str = "checkpoint_final.pth",
    use_tta: bool = True,
) -> Tuple[np.ndarray, str]:
    predictor = _build_predictor(
        trainer_folder=trainer_folder,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
        use_tta=use_tta,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).name
    if not image_name.endswith("_0000.png"):
        raise ValueError(f"nnUNet input must be *_0000.png, got: {image_name}")

    predictor.predict_from_files(
        [[str(image_path)]],
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

    candidates = list(out_dir.glob("*.png"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No nnUNet output found in {out_dir}")

    out_mask = candidates[0]
    mask = cv2.imread(str(out_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {out_mask}")

    return mask, str(out_mask)


def _majority_vote(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("masks cannot be empty")
    stacked = np.stack(masks, axis=0)
    voted = stats.mode(stacked, axis=0, keepdims=False).mode
    return voted.astype(np.uint8)


def _predict_ensemble(
    image_path: str,
    trainer_folders: Sequence[str],
    output_dir: str,
    use_folds: Sequence[int],
    checkpoint_name: str,
    use_tta: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks: List[np.ndarray] = []
    model_outputs: List[Dict[str, Any]] = []
    mask_paths: List[str] = []

    for idx, trainer_folder in enumerate(trainer_folders, start=1):
        model_dir = out_dir / f"model_{idx}"
        mask, mask_path = _predict_single_case(
            image_path=image_path,
            trainer_folder=trainer_folder,
            output_dir=str(model_dir),
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
            use_tta=use_tta,
        )
        masks.append(mask)
        mask_paths.append(mask_path)
        model_outputs.append(
            {
                "model_name": Path(trainer_folder).name,
                "foreground_pixels": int((mask > 0).sum()),
                "labels_detected": [int(v) for v in np.unique(mask) if int(v) != 0],
                "notes": f"nnUNet prediction from {Path(trainer_folder).name}",
                "mask_path": mask_path,
            }
        )

    merged = _majority_vote(masks)
    merged_path = out_dir / f"{_case_base_name(image_path)}_ensemble.png"
    cv2.imwrite(str(merged_path), merged)

    return merged, model_outputs, mask_paths


# ==================== ClinicalResultAgent ====================
class ClinicalResultAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def run(self, case: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
        if out_dir is None:
            case_id = case.get("id", "unknown")
            out_dir = ARTIFACT_REPORT_ROOT / case_id
        else:
            out_dir = Path(out_dir)
        out_dir = _ensure_under_artifacts(Path(out_dir), "clinical result output directory")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 合并配置：优先使用 case 中的 config，其次 self.config，最后自动检测
        # 这样 OpenClaw 技能可以通过 case["config"] 传参
        config = self.config.copy()
        if "config" in case and isinstance(case["config"], dict):
            config.update(case["config"])

        # 保存原始 label_path（匿名化前）
        original_label_path = case.get("label_path")

        # 匿名化
        from .skills.anonymize_image import anonymize_case_image
        anonymize = bool(config.get("anonymize", True))
        working_case = anonymize_case_image(
            case,
            str(Path(out_dir) / "anonymized")
        ) if anonymize else dict(case)
        case = working_case
        gt_path = original_label_path

        # ---------- 模型路径解析 ----------
        trainer_folders = [str(Path(p)) for p in config.get("model_paths", [])]
        if not trainer_folders:
            # 没有指定模型 → 使用默认模型
            default_trainer = resolve_model_and_trainer()
            trainer_folders = [str(default_trainer)]
            print(f"[Agent] No model_paths provided, using default: {default_trainer}")

        # folds 解析：优先使用 config，否则自动检测
        nnunet_folds = _parse_folds(config.get("nnunet_folds", None))
        if nnunet_folds == (0,) and len(trainer_folders) == 1:
            try:
                detected = detect_available_folds(Path(trainer_folders[0]))
                if detected:
                    nnunet_folds = detected
                    print(f"[Agent] Auto-detected folds: {nnunet_folds}")
            except Exception as e:
                print(f"[Agent] Could not detect folds, using default (0): {e}")

        use_tta = bool(config.get("use_tta", True))
        checkpoint_name = str(config.get("checkpoint_name", "checkpoint_final.pth"))

        workspace_report_dir = _ensure_under_artifacts(
            CLINICAL_WORKSPACE_ROOT / case.get("id", "unknown"),
            "clinical result workspace directory",
        )
        workspace_report_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(case["image_path"], cv2.IMREAD_UNCHANGED)
        image_h, image_w = (image.shape[:2] if image is not None else (0, 0))
        image_area = int(image_h * image_w) if image_h and image_w else 0

        # ---------- nnUNet 推理 ----------
        if len(trainer_folders) > 1:
            final_mask, model_outputs, _ = _predict_ensemble(
                image_path=case["image_path"],
                trainer_folders=trainer_folders,
                output_dir=str(out_dir / "nnunet_ensemble"),
                use_folds=nnunet_folds,
                checkpoint_name=checkpoint_name,
                use_tta=use_tta,
            )
            ensemble_strategy = "multi_trainer_majority_vote"
            use_ensemble_flag = True
        else:
            final_mask, mask_path = _predict_single_case(
                image_path=case["image_path"],
                trainer_folder=trainer_folders[0],
                output_dir=str(out_dir / "nnunet_single"),
                use_folds=nnunet_folds,
                checkpoint_name=checkpoint_name,
                use_tta=use_tta,
            )
            model_outputs = [
                {
                    "model_name": Path(trainer_folders[0]).name,
                    "foreground_pixels": int((final_mask > 0).sum()),
                    "labels_detected": [int(v) for v in np.unique(final_mask) if int(v) != 0],
                    "notes": (
                        f"nnUNet internal fold ensemble over folds {nnunet_folds}"
                        if len(nnunet_folds) > 1
                        else "nnUNet single-fold inference"
                    ),
                    "mask_path": mask_path,
                }
            ]
            ensemble_strategy = "internal_fold_ensemble" if len(nnunet_folds) > 1 else "single_model"
            use_ensemble_flag = len(nnunet_folds) > 1

        # 后处理
        from .skills.postprocess import geometric_postprocess
        post = geometric_postprocess(final_mask, min_size=int(self.config.get("min_size", 20)))
        final_mask = post["mask"]

        # ---------- 构建 workflow_meta ----------
        workflow_meta = {
            "pipeline_name": "dentalclaw_result_pipeline",
            "modality": "Panoramic Dental Radiograph",
            "analysis_type": "AI-assisted segmentation and review",
            "dataset_context_name": "Dental panoramic radiograph",
            "run_time_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_path": case["image_path"],
            "out_dir": str(out_dir),
            "use_tta": use_tta,
            "use_ensemble": use_ensemble_flag,
            "ensemble_strategy": ensemble_strategy,
            "nnunet_folds": list(nnunet_folds),
            "checkpoint_name": checkpoint_name,
            "model_names": [Path(p).name for p in trainer_folders],
            "input_size": f"{image_w}x{image_h}" if image_w and image_h else "n/a",
            "image_area": image_area,
            "full_panoramic_context": True,
            "input_scale_mismatch_risk": True,
            "recommendations": [
                "Review segmentation boundaries for over-segmentation or omission.",
                "Compare the result against the raw image.",
                "Prioritize manual review when the predicted foreground is extensive.",
            ],
        }
        workflow_meta["anonymized_input"] = bool(case.get("anonymized", False))
        workflow_meta["original_image_path"] = case.get("original_image_path")
        workflow_meta["fold_ensemble"] = len(nnunet_folds) > 1

        # ---------- governance tags ----------
        governance_tags = []
        if int((final_mask > 0).sum()) >= 0.8 * final_mask.size:
            governance_tags.extend(["full_pano", "possible_oversegmentation", "manual_review_required"])
        if use_tta:
            governance_tags.append("tta_enabled")
        if use_ensemble_flag:
            governance_tags.append("ensemble_inference")
        if gt_path:
            governance_tags.append("reference_mask_provided")
        else:
            governance_tags.append("no_reference_mask")

        # ---------- case_notes ----------
        case_notes = [
            "The case was processed as a full panoramic radiograph rather than a cropped tooth patch.",
            "The trained model was optimized for patch-level inputs, so full-image inference may be scale-sensitive.",
        ]
        if gt_path is None:
            case_notes.append("No reference mask was provided; quantitative performance is reported from inference outputs only.")
        else:
            # 移除可能已经存在的误导 note（安全起见）
            case_notes = [note for note in case_notes if "No reference mask was provided" not in note]

        if int((final_mask > 0).sum()) >= 0.8 * final_mask.size:
            case_notes.append("The predicted foreground is extensive and may indicate over-segmentation on the full panorama.")

        dataset_context = [
            "Full panoramic dental X-ray context",
            "nnUNetv2 2D full-resolution inference",
            "Manual review recommended for potential scale mismatch",
        ]

        # ---------- 生成报告 ----------
        report_dir = out_dir / "report"
        print(f"[Agent] Using reference mask: {gt_path}")
        report = clinical_report_export(
            case=case,
            mask=final_mask,
            out_dir=str(report_dir),
            workspace_dir=str(workspace_report_dir),
            workflow_meta=workflow_meta,
            model_outputs=model_outputs,
            governance_tags=governance_tags,
            case_notes=case_notes,
            dataset_context=dataset_context,
            reference_mask_path=gt_path,
        )

        return {
            "text": report["report"],
            "content": report["report"],
            "report_html": report.get("report_html"),
            "html_path": report.get("html_path"),
            "overlay_path": report.get("overlay_path"),
            "summary": report["summary"],
            "review_list": report["review_list"],
            "out_dir": str(out_dir),
            "report_dir": str(report_dir),
            "evaluation_summary": report.get("evaluation_summary"),
            "files": {
                "report_md": str(report_dir / "report.md"),
                "report_html": report.get("html_path"),
                "summary_json": str(report_dir / "summary.json"),
                "review_list_json": str(report_dir / "review_list.json"),
                "overlay_png": report.get("overlay_path"),
            },
        }
