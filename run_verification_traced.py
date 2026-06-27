from __future__ import annotations

import functools
import inspect
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_trace.trace_recorder import (
    end_run,
    record_orchestrator_action,
    start_run,
    traced_tool_call_sync,
)
from run_verification import (
    build_dataset_spec,
    build_task_spec,
)
from schemas.specs import BudgetSpec
from skills.auto_train_infer_skill import AutoTrainInferenceSkill
from skills.tooth_segmentation_skill import ToothSegmentationSkill


def to_trace_value(value: Any) -> Any:
    """
    将DatasetSpec、TaskSpec等对象转换为可读的JSON结构。
    """
    if is_dataclass(value):
        return {
            key: to_trace_value(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_trace_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_trace_value(item)
            for item in value
        ]

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return repr(value)


def instrument_method(
    instance: Any,
    method_name: str,
    agent_name: str,
    decision_summary: str,
    next_action: str,
) -> None:
    """
    为指定实例方法增加工具调用追踪。
    """
    if not hasattr(instance, method_name):
        print(
            f"[WARN] Method not found: "
            f"{type(instance).__name__}.{method_name}"
        )
        return

    original_method = getattr(instance, method_name)
    signature = inspect.signature(original_method)

    @functools.wraps(original_method)
    def wrapped(*args, **kwargs):
        try:
            bound = signature.bind_partial(
                *args,
                **kwargs,
            )
            bound.apply_defaults()

            arguments = {
                key: to_trace_value(value)
                for key, value in bound.arguments.items()
            }

        except Exception:
            arguments = {
                "args": to_trace_value(args),
                "kwargs": to_trace_value(kwargs),
            }

        return traced_tool_call_sync(
            agent=agent_name,
            tool_name=method_name,
            arguments=arguments,
            invoke=lambda: original_method(
                *args,
                **kwargs,
            ),
            decision_summary=decision_summary,
            next_action=next_action,
        )

    setattr(instance, method_name, wrapped)


def build_traced_task_skill() -> ToothSegmentationSkill:
    task_skill = ToothSegmentationSkill()

    tool_definitions = [
        (
            "preprocess_dataset",
            "DataCurationAgent",
            "根据任务需求对数据集进行预处理并生成标准化数据。",
            "analyze_dataset",
        ),
        (
            "analyze_dataset",
            "DataCurationAgent",
            "分析图像维度、标注结构、类别和数据规模。",
            "bootstrap_existing_experiments",
        ),
        (
            "bootstrap_existing_experiments",
            "ExperimentationAgent",
            "检查当前工作空间中是否存在可复用的历史实验。",
            "generate_initial_experiments",
        ),
        (
            "generate_initial_experiments",
            "ExperimentationAgent",
            "根据数据集、任务和计算预算生成初始实验配置。",
            "run_training",
        ),
        (
            "run_training",
            "ExperimentationAgent",
            "使用指定实验配置启动模型训练。",
            "select_best_model",
        ),
        (
            "suggest_next_experiments",
            "ExperimentationAgent",
            "根据已有实验结果生成下一轮实验配置。",
            "run_training",
        ),
        (
            "select_best_model",
            "ExperimentationAgent",
            "根据主评估指标选择最优模型。",
            "run_inference",
        ),
        (
            "run_inference",
            "ExperimentationAgent",
            "使用最优模型执行推理并生成预测结果。",
            "generate_report",
        ),
        (
            "generate_report",
            "ReportingAgent",
            "聚合实验结果并生成结构化报告。",
            "workflow_complete",
        ),
    ]

    for (
        method_name,
        agent_name,
        decision_summary,
        next_action,
    ) in tool_definitions:
        instrument_method(
            task_skill,
            method_name,
            agent_name,
            decision_summary,
            next_action,
        )

    return task_skill


def main() -> None:
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    intent_id = "SEG_TDD_TRACE_VERIFY_001"

    prompt = (
        "请使用TDD全景片数据集训练32类牙齿分割模型，"
        "完成数据预处理、训练、推理、评估和报告生成。"
    )

    workspace = (
        REPO_ROOT
        / "workspace"
        / "trace_verify"
        / stamp
    )

    dataset_spec = build_dataset_spec(
        REPO_ROOT
    )
    task_spec = build_task_spec()

    # 第一次测试只执行一个trial和一个epoch，
    # 避免追踪测试运行时间过长。
    budget_spec = BudgetSpec(
        max_trials=1,
        max_epochs_per_trial=1,
        max_parallel=1,
    )

    run_id = start_run(
        intent_id=intent_id,
        prompt=prompt,
        dataset="TDD",
        task_type="tooth_segmentation",
        session_id=f"offline_{stamp}",
    )

    print(f"[TRACE] run_id: {run_id}")
    print(f"[TRACE] workspace: {workspace}")

    try:
        record_orchestrator_action(
            agent="CoordinatorAgent",
            action="workflow_start",
            reason=(
                "用户请求执行TDD全景片32类牙齿分割，"
                "开始构建标准化训练工作流。"
            ),
            target={
                "dataset": "TDD",
                "task_type": "tooth_segmentation",
            },
        )

        task_skill = build_traced_task_skill()

        orchestrator = AutoTrainInferenceSkill(
            task_skill
        )

        result = traced_tool_call_sync(
            agent="CoordinatorAgent",
            tool_name="auto_train_inference_workflow",
            arguments={
                "dataset_spec": to_trace_value(
                    dataset_spec
                ),
                "task_spec": to_trace_value(
                    task_spec
                ),
                "budget_spec": to_trace_value(
                    budget_spec
                ),
                "workspace": str(workspace),
            },
            invoke=lambda: orchestrator.run(
                dataset_spec=dataset_spec,
                task_spec=task_spec,
                budget_spec=budget_spec,
                workspace=str(workspace),
            ),
            decision_summary=(
                "协调数据处理、模型训练、推理、"
                "评估和报告生成。"
            ),
            next_action="workflow_complete",
        )

        summary = {
            "run_id": run_id,
            "workspace": str(workspace),
            "status": "completed",
            "best_model_export_path": (
                result.get(
                    "best_model_export_path"
                )
            ),
            "best_model": result.get(
                "best_model"
            ),
            "memory_path": result.get(
                "memory_path"
            ),
            "memory_records": result.get(
                "memory_records"
            ),
            "report_dir": str(
                workspace / "report"
            ),
            "inference_dir": str(
                workspace / "best_inference"
            ),
        }

        workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            workspace
            / "trace_verify_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        end_run(
            status="completed",
            final_response=(
                "TDD分割追踪验证工作流执行完成。"
            ),
            workflow_config=summary,
        )

        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    except Exception as exc:
        end_run(
            status="failed",
            final_response=(
                "TDD分割追踪验证工作流执行失败。"
            ),
            workflow_config={
                "workspace": str(workspace),
            },
            error=str(exc),
        )

        print(
            f"[ERROR] Workflow failed: {exc}"
        )
        raise


if __name__ == "__main__":
    main()
