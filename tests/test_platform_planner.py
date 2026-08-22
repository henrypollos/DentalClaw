import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_trace.platform_planner import build_execution_plan, load_model_registry


def test_build_execution_plan_for_segmentation_training():
    plan = build_execution_plan("帮我用私有数据训练一个 3D 牙齿分割模型")
    assert plan["supported"] is True
    assert plan["task_type"] == "segmentation"
    assert plan["mode"] == "train"
    assert plan["modality"] == "3d"
    assert plan["workflow_template"] == "segmentation/train"
    assert plan["selected_model_id"] == "swin_unetr_v2"


def test_build_execution_plan_rejects_unsupported_task():
    plan = build_execution_plan("帮我做一个视频动作识别任务")
    assert plan["supported"] is False
    assert plan["task_type"] == "unsupported"


def test_build_execution_plan_includes_training_command_and_steps():
    plan = build_execution_plan("请帮我训练一个 3D 牙齿分割模型")
    assert plan["supported"] is True
    assert "training_command" in plan
    assert "dry-run" in plan["training_command"].lower()
    assert "run_training" in plan["execution_steps"]
    assert "evaluate_model" in plan["execution_steps"]


def test_model_registry_contains_preselected_models():
    registry = load_model_registry()
    assert "nnunet_v2" in registry
    assert "yolov11" in registry
    assert "convnext_v2" in registry
