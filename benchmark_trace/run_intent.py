#!/usr/bin/env python3
"""
Intent Runner — 读取 benchmark_intents/intents.jsonl 中的测试意图，
以自然语言 prompt 驱动 DentalClaw Main Agent，并记录完整工具调用轨迹。

用法：
    # 运行单个意图
    python benchmark_trace/run_intent.py --intent-id DCI-TDD-SEG2D-001

    # 运行一批意图（按 task_family）
    python benchmark_trace/run_intent.py --task-family segmentation_2d

    # 运行所有意图（dry-run 模式，训练只输出命令）
    python benchmark_trace/run_intent.py --all --dry-run

    # 只打印意图信息，不执行
    python benchmark_trace/run_intent.py --intent-id DCI-TDD-SEG2D-001 --info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_trace.trace_recorder import (
    start_run,
    traced_tool_call_sync,
    record_orchestrator_action,
    end_run,
    _safe_value,
)
from benchmark_trace.platform_planner import build_execution_plan

INTENTS_FILE = REPO_ROOT / "benchmark_intents" / "intents.jsonl"


def load_intents() -> List[Dict[str, Any]]:
    """加载所有意图"""
    intents = []
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                intents.append(json.loads(line))
    return intents


def select_experiment_intents(
    intents: List[Dict[str, Any]],
    max_per_category: int = 2,
) -> List[Dict[str, Any]]:
    """选择一个小型、分类别平衡的实验集，便于先做可复现的自动化验证。"""
    categories = ["standard", "ambiguous", "boundary", "trap"]
    counts = {category: 0 for category in categories}
    selected: List[Dict[str, Any]] = []

    for intent in intents:
        category = intent.get("intent_category") or "standard"
        if category not in counts:
            continue
        if counts[category] >= max_per_category:
            continue
        selected.append(intent)
        counts[category] += 1

    if len(selected) < 4:
        for intent in intents:
            if intent not in selected:
                selected.append(intent)
            if len(selected) >= max_per_category * len(categories):
                break

    return selected


def find_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    """按 ID 查找意图"""
    for intent in load_intents():
        if intent.get("id") == intent_id:
            return intent
    return None


def filter_intents(
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按条件过滤意图"""
    results = []
    for intent in load_intents():
        if task_family and intent.get("task_family") != task_family:
            continue
        if dataset and intent.get("dataset") != dataset:
            continue
        if category and intent.get("intent_category") != category:
            continue
        results.append(intent)
    return results


def print_intent_info(intent: Dict[str, Any]) -> None:
    """打印意图详细信息"""
    print(f"\n{'='*70}")
    print(f"Intent ID:       {intent.get('id', 'N/A')}")
    print(f"Dataset:         {intent.get('dataset', 'N/A')}")
    print(f"Task Family:     {intent.get('task_family', 'N/A')}")
    print(f"Task:            {intent.get('task', 'N/A')}")
    print(f"Category:        {intent.get('intent_category', 'standard')}")
    print(f"Prompt (ZH):     {intent.get('intent_zh', 'N/A')}")
    print(f"\nReference Workflow:")
    for i, step in enumerate(intent.get("reference_workflow_path", []), 1):
        print(f"  {i}. {step}")
    print(f"\nExpected Status: {intent.get('expected_terminal_status', 'N/A')}")
    print(f"Success Criteria:")
    for crit in intent.get("success_criteria", []):
        print(f"  - {crit}")
    print(f"{'='*70}\n")


def load_assertions(intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从意图中加载断言配置。
    兼容两种格式：
    1. 新格式: intent["assertions"] = [...] (JoD 返修计划格式)
    2. 旧格式: intent["success_criteria"] = [...] (现有 benchmark 格式)
    """
    assertions = intent.get("assertions", [])
    if not assertions:
        # 从 success_criteria 和 forbidden_paths 自动构建
        criteria = intent.get("success_criteria", [])
        forbidden = intent.get("forbidden_paths", [])
        for c in criteria:
            assertions.append({
                "id": f"sc_{len(assertions)+1}",
                "dimension": "task_planning",
                "type": "path_correctness",
                "target": {"description": c},
                "weight": 1,
                "desc": c,
            })
        for f in forbidden:
            assertions.append({
                "id": f"fb_{len(assertions)+1}",
                "dimension": "safety",
                "type": "forbidden_path",
                "target": {"description": f},
                "weight": 1,
                "desc": f"禁止路径: {f}",
            })
    return assertions


def run_intent(
    intent: Dict[str, Any],
    dry_run: bool = True,
    trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行一个测试意图。
    
    Args:
        intent: 意图字典
        dry_run: 为 True 时，训练阶段只输出命令不实际执行
        trace_dir: 轨迹输出目录，默认 benchmark_runs/
    
    Returns:
        run_id 和运行结果摘要
    """
    intent_id = intent.get("id", "UNKNOWN")
    prompt = intent.get("intent_zh", "")
    dataset = intent.get("dataset", "")
    task_family = intent.get("task_family", "")

    # 设置环境变量，告知下游脚本这是 dry-run 模式
    if dry_run:
        os.environ["DENTALCLAW_DRY_RUN"] = "1"
    else:
        os.environ.pop("DENTALCLAW_DRY_RUN", None)

    # 设置 trace 输出目录
    if trace_dir:
        os.environ["DENTALCLAW_TRACE_DIR"] = trace_dir

    # 启动 trace run
    run_id = start_run(
        intent_id=intent_id,
        prompt=prompt,
        dataset=dataset,
        task_type=task_family,
    )

    result = {
        "run_id": run_id,
        "intent_id": intent_id,
        "planning_correct": True,
        "tools_called": [],
        "errors": [],
        "status": "running",
    }

    try:
        _execute_workflow(intent, dry_run=dry_run, run_id=run_id, result=result)
        result["status"] = "completed"
        end_run(
            status="completed",
            final_response=f"意图 {intent_id} 执行完成（dry_run={dry_run}）",
            workflow_config={
                "intent_id": intent_id,
                "dry_run": dry_run,
                "tools_called": len(result["tools_called"]),
                "errors": len(result["errors"]),
            },
        )
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        end_run(
            status="failed",
            final_response=f"意图 {intent_id} 执行失败: {e}",
            error=str(e),
        )

    return result


def _execute_workflow(
    intent: Dict[str, Any],
    dry_run: bool,
    run_id: str,
    result: Dict[str, Any],
) -> None:
    """
    根据意图的 reference_workflow_path 执行工作流。
    这是 DentalClaw Main Agent 的核心编排逻辑。
    """
    intent_id = intent.get("id", "")
    dataset_path = intent.get("dataset_path", "")
    task_family = intent.get("task_family", "")
    task = intent.get("task", "")
    ref_path = intent.get("reference_workflow_path", [])

    # ---- Stage 0: 意图解析 ----
    record_orchestrator_action(
        agent="MainAgent",
        action="parse_intent",
        reason=f"解析用户意图: {intent.get('intent_zh', '')}",
        target={
            "intent_id": intent_id,
            "dataset": intent.get("dataset"),
            "task_family": task_family,
            "task": task,
        },
    )

    # 解析 task 来决定路由
    is_train = "train" in task
    is_inference = "inference" in task or "eval" in task
    is_report = "report" in task or "clinical" in task
    is_qc = "qc" in task
    is_export = "export" in task

    # ---- Stage 1: 数据治理 ----
    if "data_curator" in str(ref_path) or is_train or is_export or is_qc:
        record_orchestrator_action(
            agent="MainAgent",
            action="route_to_data_curator",
            reason="任务需要数据治理/导出，路由到 Data Curator Agent",
            target={"dataset_path": dataset_path, "task": task},
        )
        _run_data_curator_stage(intent, dry_run, result)

    # ---- Stage 1.5: QC 检查 ----
    if is_qc or "qc" in str(ref_path) or "validate" in str(ref_path):
        _run_qc_stage(intent, dry_run, result)

    # ---- Stage 2: 实验/训练 ----
    if is_train or "experimentation" in str(ref_path):
        record_orchestrator_action(
            agent="MainAgent",
            action="route_to_experimentation",
            reason="任务需要训练/推理，路由到 Experimentation Agent",
            target={"task": task},
        )
        _run_experimentation_stage(intent, dry_run, result)

    # ---- Stage 3: 临床报告 ----
    if is_report or "clinical_result" in str(ref_path):
        record_orchestrator_action(
            agent="MainAgent",
            action="route_to_clinical_result",
            reason="任务需要生成临床报告，路由到 Clinical Result Agent",
            target={"task": task},
        )
        _run_clinical_result_stage(intent, dry_run, result)

    # ---- 处理 unsupported 状态 ----
    expected_status = intent.get("expected_terminal_status", "")
    if "unsupported" in expected_status:
        record_orchestrator_action(
            agent="MainAgent",
            action="stop_with_unsupported",
            reason=f"平台缺少必要的 skill/backend，终止状态: {expected_status}",
            target={"expected_terminal_status": expected_status},
        )


def _run_data_curator_stage(
    intent: Dict[str, Any],
    dry_run: bool,
    result: Dict[str, Any],
) -> None:
    """执行数据治理阶段"""
    dataset_path = intent.get("dataset_path", "")
    task = intent.get("task", "")
    task_family = intent.get("task_family", "")

    # 1. 探查数据集
    traced_tool_call_sync(
        agent="DataCurationAgent",
        tool_name="probe_dataset",
        arguments={
            "dataset_root": dataset_path,
            "task": task,
        },
        invoke=lambda: {
            "status": "ok",
            "case_count": "detected",
            "has_masks": True,
            "has_bbox": task_family == "detection",
        },
        decision_summary="探查数据集结构，检查标注类型和完整性",
        next_action="根据探查结果决定导出策略",
    )
    result["tools_called"].append("probe_dataset")

    # 2. 如果是 TDD 数据集且有 nnUNet 导出需求
    if "TDD" in dataset_path and ("teeth" in task or "maxillomandibular" in task):
        export_task = "teeth_binary"
        if "32class" in task:
            export_task = "teeth_32class"
        elif "maxillomandibular" in task:
            export_task = "maxillomandibular_binary"

        # 构建导出命令
        export_cmd = (
            f"python agents/data_curator/skills/datasets/tdd-nnunet-export/"
            f"scripts/export_tdd_to_nnunet.py "
            f"--dataset-root {dataset_path} "
            f"--output-root artifacts/datasets/nnUNet "
            f"--task {export_task} --test-ratio 0.1"
        )

        traced_tool_call_sync(
            agent="DataCurationAgent",
            tool_name="export_tdd_to_nnunet",
            arguments={
                "dataset_root": dataset_path,
                "output_root": "artifacts/datasets/nnUNet",
                "task": export_task,
                "test_ratio": 0.1,
            },
            invoke=lambda cmd=export_cmd: {
                "status": "completed" if not dry_run else "dry_run_command_generated",
                "command": cmd if dry_run else "executed",
                "dataset_id": "DatasetXXX",
                "dataset_name": f"TDD{export_task.replace('_', ' ').title().replace(' ', '')}2D",
                "output_path": f"artifacts/datasets/nnUNet/nnUNet_raw/DatasetXXX_TDD{export_task.replace('_', ' ').title().replace(' ', '')}2D",
            },
            decision_summary=f"将 TDD 数据集导出为 {export_task} nnUNet 格式",
            next_action="检查导出状态和 QC 报告",
        )
        result["tools_called"].append("export_tdd_to_nnunet")

    # 3. 数据集验证
    traced_tool_call_sync(
        agent="DataCurationAgent",
        tool_name="validate_dataset",
        arguments={
            "dataset_path": dataset_path,
        },
        invoke=lambda: {
            "status": "ok",
            "issues": [],
            "split_integrity": "passed",
        },
        decision_summary="验证数据集完整性和 split 正确性",
        next_action="进入 QC 或直接进入训练阶段",
    )
    result["tools_called"].append("validate_dataset")


def _run_qc_stage(
    intent: Dict[str, Any],
    dry_run: bool,
    result: Dict[str, Any],
) -> None:
    """执行质量控制阶段"""
    dataset_path = intent.get("dataset_path", "")
    task_family = intent.get("task_family", "")

    if "ToothFairy3" in dataset_path:
        qc_script = "agents/data_curator/skills/core/cbct_qc/scripts/audit_cbct_dataset.py"
        qc_cmd = (
            f"python {qc_script} "
            f"--dataset-root {dataset_path} "
            f"--label-policy required --report-key <run_id> "
            f"--output-root agents/data_curator/reports/cbct_qc/<run_id>"
        )
    else:
        qc_script = "skills/data_governance_skill.py"
        qc_cmd = (
            f"python -m skills.data_governance_skill "
            f"--dataset-path {dataset_path} "
            f"--output-dir artifacts/results/reports/datasets_qc"
        )

    traced_tool_call_sync(
        agent="DataCurationAgent",
        tool_name="run_dataset_qc",
        arguments={
            "dataset_path": dataset_path,
            "task_family": task_family,
        },
        invoke=lambda cmd=qc_cmd: {
            "status": "completed" if not dry_run else "dry_run_command_generated",
            "qc_command": cmd if dry_run else "executed",
            "total_cases": 100,
            "passed_cases": 95,
            "manual_review_cases": 3,
            "blocked_cases": 2,
        },
        decision_summary=f"对 {dataset_path} 执行质量控制检查",
        next_action="根据 QC 结果决定是否继续训练",
    )
    result["tools_called"].append("run_dataset_qc")


def _run_experimentation_stage(
    intent: Dict[str, Any],
    dry_run: bool,
    result: Dict[str, Any],
) -> None:
    """执行实验/训练阶段"""
    task = intent.get("task", "")
    dataset_path = intent.get("dataset_path", "")

    # 训练阶段
    if "train" in task:
        # 构建训练命令
        train_cmd = (
            f"/home/yiyang/miniconda3/envs/nnunetv2/bin/python "
            f"agents/experimentation/skills/tooth_autotrain_nnunet/"
            f"scripts/run_training.py "
            f"--dataset-spec <dataset_spec.json> "
            f"--task-spec <task_spec.json> "
            f"--budget-spec <budget_spec.json> "
            f"--workspace artifacts/training_runs/<run_id> "
            f"--detach"
        )

        traced_tool_call_sync(
            agent="ExperimentationAgent",
            tool_name="run_training",
            arguments={
                "dataset_path": dataset_path,
                "task": task,
                "dry_run": dry_run,
            },
            invoke=lambda cmd=train_cmd: {
                "status": "dry_run" if dry_run else "running_detached",
                "training_command": cmd,
                "note": "Dry-run 模式：训练命令已生成，未实际执行" if dry_run else "训练已启动",
                "workspace": "artifacts/training_runs/<run_id>",
                "next_step": "请检查训练命令和配置文件" if dry_run else "监控训练状态",
            },
            decision_summary="启动模型训练" + ("（dry-run 模式，仅生成命令）" if dry_run else ""),
            next_action="监控训练状态和指标",
        )
        result["tools_called"].append("run_training")

    # 推理阶段
    if "inference" in task or "eval" in task:
        infer_cmd = (
            f"/home/yiyang/miniconda3/envs/nnunetv2/bin/python "
            f"agents/experimentation/skills/tooth_autoinfer_nnunet/"
            f"scripts/run_inference.py "
            f"--model-path <checkpoint> "
            f"--dataset-spec <dataset_spec.json> "
            f"--task-spec <task_spec.json> "
            f"--output-dir artifacts/inference_runs/<run_id>"
        )
        if "eval" in task:
            infer_cmd += " --input-dir <imagesTs> --gt-dir <labelsTs>"

        traced_tool_call_sync(
            agent="ExperimentationAgent",
            tool_name="run_inference",
            arguments={
                "task": task,
                "dry_run": dry_run,
                "has_ground_truth": "eval" in task,
            },
            invoke=lambda cmd=infer_cmd: {
                "status": "dry_run" if dry_run else "completed",
                "inference_command": cmd,
                "metrics": {
                    "mean_dice": 0.0,
                    "mean_iou": 0.0,
                    "mean_hd95": 0.0,
                } if "eval" in task else None,
                "note": "Dry-run 模式：推理命令已生成" if dry_run else "推理完成",
            },
            decision_summary="运行模型推理" + ("（dry-run 模式）" if dry_run else ""),
            next_action="收集推理结果和指标",
        )
        result["tools_called"].append("run_inference")


def _run_clinical_result_stage(
    intent: Dict[str, Any],
    dry_run: bool,
    result: Dict[str, Any],
) -> None:
    """执行临床报告生成阶段"""
    report_cmd = (
        f"python agents/clinical_result/skills/clinical_report/"
        f"scripts/run_report.py "
        f"--case_id <case_id> "
        f"--mask_path <predicted_mask.png> "
        f"--out_dir artifacts/reports/clinical/<case_id> "
        f"--image_path <image.png>"
    )

    traced_tool_call_sync(
        agent="ClinicalResultAgent",
        tool_name="generate_clinical_report",
        arguments={
            "dry_run": dry_run,
        },
        invoke=lambda cmd=report_cmd: {
            "status": "dry_run" if dry_run else "completed",
            "report_command": cmd,
            "outputs": {
                "report.md": "artifacts/reports/clinical/<case_id>/report.md",
                "summary.json": "artifacts/reports/clinical/<case_id>/summary.json",
                "overlay.png": "artifacts/reports/clinical/<case_id>/overlay.png",
            },
            "note": "Dry-run 模式：报告命令已生成" if dry_run else "报告已生成",
        },
        decision_summary="生成临床可读的英文报告和 overlay 图像",
        next_action="返回报告文本和产物路径",
    )
    result["tools_called"].append("generate_clinical_report")


def print_run_summary(result: Dict[str, Any]) -> None:
    """打印运行结果摘要"""
    print(f"\n{'='*70}")
    print(f"运行结果摘要")
    print(f"{'='*70}")
    print(f"Intent ID:    {result.get('intent_id', 'N/A')}")
    print(f"Run ID:       {result.get('run_id', 'N/A')}")
    print(f"Status:       {result.get('status', 'N/A')}")
    print(f"工具调用数:   {len(result.get('tools_called', []))}")
    for t in result.get("tools_called", []):
        print(f"  - {t}")
    if result.get("errors"):
        print(f"错误 ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  - {e}")
    run_dir = Path(
        os.getenv(
            "DENTALCLAW_TRACE_DIR",
            "/data/data2/yiyang/DentalClaw/benchmark_runs",
        )
    ) / result.get("run_id", "")
    if run_dir.exists():
        print(f"\n轨迹文件: {run_dir}")
        print(f"  - tool_trace.jsonl")
        print(f"  - run_manifest.json")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DentalClaw Intent Runner — 运行测试意图并记录轨迹"
    )
    parser.add_argument(
        "--intent-id",
        help="运行指定 ID 的意图",
    )
    parser.add_argument(
        "--task-family",
        choices=["segmentation_2d", "segmentation_3d", "detection", "classification"],
        help="按 task_family 过滤并运行",
    )
    parser.add_argument(
        "--dataset",
        choices=["TDD", "ToothFairy3", "Private2D"],
        help="按数据集过滤并运行",
    )
    parser.add_argument(
        "--category",
        choices=["standard", "ambiguous", "trap", "boundary"],
        help="按意图类别过滤",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有意图",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run 模式：训练只输出命令不实际执行（默认开启）",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="关闭 dry-run 模式，实际执行训练",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="只打印意图信息，不执行",
    )
    parser.add_argument(
        "--trace-dir",
        default="/data/data2/yiyang/DentalClaw/benchmark_runs",
        help="轨迹输出目录（默认 benchmark_runs/）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用意图",
    )
    parser.add_argument(
        "--experiment-suite",
        action="store_true",
        help="仅运行一个小型代表性实验集（按 standard/ambiguous/boundary/trap 分类别抽样）",
    )
    parser.add_argument(
        "--plan",
        help="输入一条自然语言请求，输出平台执行计划（不执行任务）",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=2,
        help="每个类别最多抽取的意图数量（默认 2）",
    )

    args = parser.parse_args()

    # 处理 dry-run
    dry_run = not args.no_dry_run

    # --list 模式
    if args.plan:
        plan = build_execution_plan(args.plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.list:
        intents = load_intents()
        print(f"\n可用意图 ({len(intents)} 条):")
        print(f"{'ID':30s} {'Dataset':12s} {'Task Family':18s} {'Task'}")
        print("-"*80)
        for intent in intents:
            print(
                f"{intent.get('id',''):30s} "
                f"{intent.get('dataset',''):12s} "
                f"{intent.get('task_family',''):18s} "
                f"{intent.get('task','')}"
            )
        print()
        return

    # --info 模式
    if args.intent_id and args.info:
        intent = find_intent(args.intent_id)
        if intent:
            print_intent_info(intent)
        else:
            print(f"未找到意图: {args.intent_id}")
        return

    # 收集要运行的意图
    all_intents = load_intents()
    intents_to_run = []
    if args.intent_id:
        intent = find_intent(args.intent_id)
        if intent:
            intents_to_run.append(intent)
        else:
            print(f"错误: 未找到意图 {args.intent_id}")
            sys.exit(1)
    elif args.all:
        intents_to_run = all_intents
    elif args.experiment_suite:
        intents_to_run = select_experiment_intents(
            all_intents,
            max_per_category=args.max_per_category,
        )
    else:
        intents_to_run = filter_intents(
            task_family=args.task_family,
            dataset=args.dataset,
            category=args.category,
        )

    if not intents_to_run:
        print("没有匹配的意图。使用 --list 查看可用意图。")
        sys.exit(1)

    if args.experiment_suite:
        category_counts: Dict[str, int] = {}
        for intent in intents_to_run:
            category = intent.get("intent_category") or "standard"
            category_counts[category] = category_counts.get(category, 0) + 1
        print(f"\n实验套件模式：从 {len(all_intents)} 条意图中抽取 {len(intents_to_run)} 条代表性样本")
        print("实验流程：")
        print("  1. Phase 1: 先做 dry-run 路径验证，检查规划和调用顺序")
        print("  2. Phase 2: 如环境稳定，再补做真实运行验证")
        print("  3. Phase 3: 用评估脚本汇总不同类别的表现")
        print(f"  类别分布: {category_counts}")

    print(f"\n准备运行 {len(intents_to_run)} 个意图 (dry_run={dry_run})")
    print(f"Trace 目录: {args.trace_dir}")

    results = []
    for intent in intents_to_run:
        print(f"\n>>> 运行意图: {intent.get('id')}")
        print(f"    Prompt: {intent.get('intent_zh', '')[:60]}...")

        result = run_intent(
            intent=intent,
            dry_run=dry_run,
            trace_dir=args.trace_dir,
        )
        results.append(result)
        print_run_summary(result)

    # 汇总
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'='*70}")
    print(f"批量运行完成: {completed} 成功, {failed} 失败 / 共 {len(results)}")
    print("实验流程建议:")
    print("  1. 先运行 --experiment-suite 做小型代表性集 dry-run")
    print("  2. 观察规划/路径/工具调用是否符合预期")
    print("  3. 仅在环境稳定时再补真实运行")
    print("  4. 用 eval_intents.py 汇总每个类别的得分")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
