#!/usr/bin/env python3
"""DentalClaw 平台规划器。

目标：
1. 先把用户一句话解析成结构化任务意图；
2. 根据任务意图选择预选模型；
3. 生成一个最小可执行的 workflow 模板。

当前版本只做 MVP 级别的规则化规划，后续可以继续扩展到更复杂的模型/环境调度。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = REPO_ROOT / "benchmark_trace" / "model_registry.json"


def load_model_registry(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """加载预选模型清单。"""
    registry_path = path or MODEL_REGISTRY_PATH
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # 默认内置一组 MVP 模型，便于先跑通骨架。
    return {
        "nnunet_v2": {
            "id": "nnunet_v2",
            "name": "nnU-Net v2",
            "task_types": ["segmentation"],
            "modalities": ["2d", "3d"],
            "supports": ["train", "inference"],
            "framework": "nnunet",
            "default_template": "segmentation/train",
        },
        "swin_unetr_v2": {
            "id": "swin_unetr_v2",
            "name": "Swin UNETR V2",
            "task_types": ["segmentation"],
            "modalities": ["3d"],
            "supports": ["train", "inference"],
            "framework": "monai",
            "default_template": "segmentation/train",
        },
        "yolov11": {
            "id": "yolov11",
            "name": "YOLOv11",
            "task_types": ["detection"],
            "modalities": ["2d"],
            "supports": ["train", "inference"],
            "framework": "ultralytics",
            "default_template": "detection/train",
        },
        "convnext_v2": {
            "id": "convnext_v2",
            "name": "ConvNeXt V2",
            "task_types": ["classification"],
            "modalities": ["2d"],
            "supports": ["train", "inference"],
            "framework": "torchvision",
            "default_template": "classification/train",
        },
    }


def _detect_task_type(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ["分割", "segmentation", "segment", "mask"]):
        return "segmentation"
    if any(k in lowered for k in ["检测", "detection", "detect", "bbox", "目标"]):
        return "detection"
    if any(k in lowered for k in ["分类", "classification", "classify", "类别"]):
        return "classification"
    return "unsupported"


def _detect_mode(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ["训练", "train", "fine-tune", "微调"]):
        return "train"
    if any(k in lowered for k in ["推理", "inference", "infer", "预测", "eval", "评估"]):
        return "inference"
    return "train"


def _detect_modality(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ["3d", "三维", "ct", "cbct", "体积"]):
        return "3d"
    if any(k in lowered for k in ["2d", "二维", "图像", "影像"]):
        return "2d"
    return "2d"


def _select_model(task_type: str, modality: str, mode: str, registry: Dict[str, Dict[str, Any]]) -> str | None:
    candidates: List[Dict[str, Any]] = []
    for model in registry.values():
        if task_type not in model.get("task_types", []):
            continue
        if modality not in model.get("modalities", []):
            continue
        if mode not in model.get("supports", []):
            continue
        candidates.append(model)

    if not candidates:
        return None

    # 规则优先级：segmentation -> 3d -> swint unetr, otherwise default by task
    if task_type == "segmentation" and modality == "3d":
        for model in candidates:
            if model["id"] == "swin_unetr_v2":
                return model["id"]
    if task_type == "segmentation":
        for model in candidates:
            if model["id"] == "nnunet_v2":
                return model["id"]
    if task_type == "detection":
        for model in candidates:
            if model["id"] == "yolov11":
                return model["id"]
    if task_type == "classification":
        for model in candidates:
            if model["id"] == "convnext_v2":
                return model["id"]
    return candidates[0]["id"]


def _build_training_command(task_type: str, modality: str, model_id: str) -> str:
    """生成一个 dry-run 训练命令。"""
    framework = {
        "swin_unetr_v2": "monai",
        "nnunet_v2": "nnunet",
        "yolov11": "ultralytics",
        "convnext_v2": "torchvision",
    }.get(model_id, "generic")

    workspace = "artifacts/training_runs/<run_id>"
    if task_type == "segmentation" and modality == "3d":
        return (
            f"python agents/experimentation/skills/tooth_autotrain_{framework}/"
            f"scripts/run_training.py --dataset-spec <dataset_spec.json> "
            f"--task-spec <task_spec.json> --workspace {workspace} --dry-run"
        )

    return (
        f"python agents/experimentation/skills/tooth_autotrain_{framework}/"
        f"scripts/run_training.py --dataset-spec <dataset_spec.json> "
        f"--task-spec <task_spec.json> --workspace {workspace} --dry-run"
    )


def build_execution_plan(user_text: str) -> Dict[str, Any]:
    """根据自然语言输入，生成一个结构化执行计划。"""
    task_type = _detect_task_type(user_text)
    mode = _detect_mode(user_text)
    modality = _detect_modality(user_text)

    if task_type == "unsupported":
        return {
            "supported": False,
            "task_type": task_type,
            "mode": mode,
            "modality": modality,
            "workflow_template": None,
            "selected_model_id": None,
            "reason": "当前 MVP 仅支持分割、检测、分类任务。",
        }

    registry = load_model_registry()
    selected_model_id = _select_model(task_type, modality, mode, registry)
    if not selected_model_id:
        return {
            "supported": False,
            "task_type": task_type,
            "mode": mode,
            "modality": modality,
            "workflow_template": None,
            "selected_model_id": None,
            "reason": "当前模型注册表中没有匹配的候选模型。",
        }

    workflow_template = f"{task_type}/{mode}"
    execution_steps = [
        "prepare_dataset",
        "run_training",
        "evaluate_model",
    ]
    if mode == "inference":
        execution_steps = ["prepare_dataset", "run_inference", "collect_results"]

    plan = {
        "supported": True,
        "task_type": task_type,
        "mode": mode,
        "modality": modality,
        "workflow_template": workflow_template,
        "selected_model_id": selected_model_id,
        "selected_model_name": registry[selected_model_id]["name"],
        "execution_steps": execution_steps,
        "reason": "已命中 MVP 任务模板和预选模型。",
    }

    if mode == "train":
        plan["training_command"] = _build_training_command(task_type, modality, selected_model_id)
        plan["training_command_note"] = "本版本仅生成训练命令，不实际执行训练。"
    else:
        plan["inference_command"] = (
            f"python agents/experimentation/skills/tooth_autoinfer_{registry[selected_model_id]['framework']}/"
            f"scripts/run_inference.py --model-path <checkpoint> --dataset-spec <dataset_spec.json> --dry-run"
        )

    return plan
