#!/usr/bin/env python3
"""
Intent Eval — 评估 DentalClaw 实际执行轨迹与参考工作流的匹配程度。

用法：
    # 评估单个 run
    python benchmark_trace/eval_intents.py --run-dir benchmark_runs/DCI-TDD-SEG2D-001_20260624_*

    # 评估所有 runs
    python benchmark_trace/eval_intents.py --all

    # 输出详细报告
    python benchmark_trace/eval_intents.py --run-dir <dir> --verbose

    # 生成汇总 CSV
    python benchmark_trace/eval_intents.py --all --csv eval_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_trace.trace_recorder import TRACE_ROOT

INTENTS_FILE = REPO_ROOT / "benchmark_intents" / "intents.jsonl"


# ==============================================================================
# 评估维度
# ==============================================================================

DIMENSION_WEIGHTS = {
    "task_planning": 0.25,      # 任务规划正确性
    "tool_correctness": 0.20,   # 工具调用正确性
    "trajectory_efficiency": 0.10,  # 轨迹效率
    "qc_detection": 0.15,       # 质控检测能力
    "ambiguity_handling": 0.10, # 歧义处理
    "boundary_identification": 0.10,  # 能力边界识别
    "artifact_completeness": 0.10,    # 产物完整性
}


def load_intents() -> Dict[str, Dict[str, Any]]:
    """加载意图，按 id 索引"""
    intents = {}
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                intent = json.loads(line)
                intents[intent["id"]] = intent
    return intents


def find_run_dirs() -> List[Path]:
    """扫描 benchmark_runs 下所有 run 目录"""
    runs = []
    for p in TRACE_ROOT.iterdir():
        if p.is_dir() and (p / "tool_trace.jsonl").exists():
            runs.append(p)
    return sorted(runs, key=lambda p: p.name)


def load_trace(path: Path) -> List[Dict[str, Any]]:
    """加载 tool_trace.jsonl"""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_manifest(path: Path) -> Dict[str, Any]:
    """加载 run_manifest.json"""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tool_sequence(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从事件中提取工具调用序列（去重 start/end）"""
    tools = []
    seen_calls = set()
    for event in events:
        if event.get("event") == "tool_call_start":
            call_id = event.get("call_id")
            if call_id not in seen_calls:
                seen_calls.add(call_id)
                tools.append({
                    "call_id": call_id,
                    "agent": event.get("agent"),
                    "tool_name": event.get("tool_name"),
                    "arguments": event.get("arguments", {}),
                    "decision_summary": event.get("decision_summary"),
                    "next_action": event.get("next_action"),
                    "sequence": event.get("sequence"),
                })
    return tools


def extract_orchestrator_actions(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """提取编排层动作"""
    actions = []
    for event in events:
        if event.get("event") == "orchestrator_action":
            actions.append(event)
    return actions


def extract_errors(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """提取错误事件"""
    errors = []
    for event in events:
        if event.get("event") == "tool_call_error":
            errors.append(event)
    return errors


# Agent 名称映射表（reference → actual）
AGENT_NAME_MAP = {
    "main": "MainAgent",
    "data_curator": "DataCurationAgent",
    "experimentation": "ExperimentationAgent",
    "clinical_result": "ClinicalResultAgent",
    "research": "ResearchAgent",
}


def _normalize_agent(name: str) -> str:
    """统一 agent 名称，支持模糊匹配"""
    # 直接匹配
    if name in AGENT_NAME_MAP:
        return AGENT_NAME_MAP[name]
    # 反向匹配
    for ref, actual in AGENT_NAME_MAP.items():
        if actual == name:
            return actual
        if ref.lower() in name.lower() or name.lower() in ref.lower():
            return actual
    return name


def _normalize_action(action: str) -> str:
    """统一动作名称，提取关键动词"""
    # 移除参数括号
    action = re.sub(r"\(.*?\)", "", action)
    # 提取最后一个 . 后的部分（针对 script_path.action 格式）
    if "." in action:
        action = action.split(".")[-1]
    return action.strip()


def parse_reference_path(ref_path: List[str]) -> List[Dict[str, Any]]:
    """
    解析 reference_workflow_path 为结构化节点列表。
    
    节点格式示例:
        "main.parse_intent(dataset=TDD,task=segmentation_2d,mode=train)"
        "data_curator.tdd-nnunet-export.scripts/export_tdd_to_nnunet.py --task teeth_binary"
    """
    nodes = []
    for step in ref_path:
        # 解析 agent 和动作
        agent_match = re.match(r"^(\w+)\.", step)
        agent = agent_match.group(1) if agent_match else "unknown"

        # 提取动作：agent 后的第一个词（函数名或脚本名）
        action_part = step[len(agent)+1:] if agent_match else step
        action_match = re.match(r"([\w./-]+)", action_part)
        action = action_match.group(1) if action_match else action_part

        # 提取参数
        params = {}
        param_match = re.search(r"\((.*?)\)", step)
        if param_match:
            for pair in param_match.group(1).split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k.strip()] = v.strip()

        nodes.append({
            "raw": step,
            "agent": agent,
            "normalized_agent": _normalize_agent(agent),
            "action": action,
            "normalized_action": _normalize_action(action),
            "params": params,
            "is_script": ".py" in step,
            "script_path": re.search(r"(\S+\.py)", step).group(1) if ".py" in step else None,
        })
    return nodes


# ==============================================================================
# 评估指标
# ==============================================================================

def eval_task_planning(
    actual_tools: List[Dict[str, Any]],
    ref_nodes: List[Dict[str, Any]],
    orchestrator_actions: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估任务规划正确性。
    
    指标:
    - 关键节点正确率: 实际路径与参考路径的匹配程度
    - 路径通过率: 实际路径是否覆盖了所有必需节点
    """
    # 实际 agent 和工具名
    actual_agents = set()
    for t in actual_tools:
        agent = t.get("agent", "")
        actual_agents.add(agent)
    for a in orchestrator_actions:
        agent = a.get("agent", "")
        if agent:
            actual_agents.add(agent)

    actual_tool_names = set(t.get("tool_name", "") for t in actual_tools)

    # 参考节点（使用归一化名称）
    ref_agents = set(n["normalized_agent"] for n in ref_nodes)
    ref_actions = set(n["normalized_action"] for n in ref_nodes)

    # Agent 路由正确性（模糊匹配）
    correct_agents = set()
    for ref_a in ref_agents:
        ref_lower = ref_a.lower()
        for act_a in actual_agents:
            act_lower = act_a.lower()
            if ref_lower in act_lower or act_lower in ref_lower:
                correct_agents.add(ref_a)
                break

    agent_precision = len(correct_agents) / len(actual_agents) if actual_agents else 0
    agent_recall = len(correct_agents) / len(ref_agents) if ref_agents else 0
    agent_f1 = 2 * agent_precision * agent_recall / (agent_precision + agent_recall) if (agent_precision + agent_recall) > 0 else 0

    # 关键动作覆盖 — 使用更灵活的匹配
    covered_actions = 0
    for node in ref_nodes:
        n_action = node["normalized_action"].lower()
        n_agent = node["normalized_agent"].lower()
        matched = False

        # 检查工具调用
        for tool in actual_tools:
            tool_name = tool.get("tool_name", "").lower()
            tool_agent = tool.get("agent", "").lower()
            summary = tool.get("decision_summary", "").lower()

            # Agent 匹配 + 工具名/动作匹配
            agent_ok = n_agent in tool_agent or tool_agent in n_agent
            if agent_ok and (n_action in tool_name or tool_name in n_action):
                matched = True
                break
            if agent_ok and n_action in summary:
                matched = True
                break

        # 检查编排动作
        if not matched:
            for act in orchestrator_actions:
                act_agent = act.get("agent", "").lower()
                act_action = act.get("action", "").lower()
                act_summary = act.get("decision_summary", "").lower()
                agent_ok = n_agent in act_agent or act_agent in n_agent
                if agent_ok and (n_action in act_action or n_action in act_summary):
                    matched = True
                    break

        if matched:
            covered_actions += 1

    action_coverage = covered_actions / len(ref_nodes) if ref_nodes else 0

    missing_ref_agents = set()
    for ref_a in ref_agents:
        found = any(
            ref_a.lower() in act_a.lower() or act_a.lower() in ref_a.lower()
            for act_a in actual_agents
        )
        if not found:
            missing_ref_agents.add(ref_a)

    details = {
        "agent_precision": round(agent_precision, 3),
        "agent_recall": round(agent_recall, 3),
        "agent_f1": round(agent_f1, 3),
        "action_coverage": round(action_coverage, 3),
        "covered_actions": covered_actions,
        "total_actions": len(ref_nodes),
        "correct_agents": list(correct_agents),
        "missing_agents": list(missing_ref_agents),
    }

    score = 0.4 * agent_f1 + 0.6 * action_coverage
    return round(score, 3), details


def eval_tool_correctness(
    actual_tools: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估工具调用正确性。
    
    指标:
    - 工具选择正确率
    - 参数完整性
    - 错误率
    """
    total_calls = len(actual_tools)
    error_count = len(errors)

    # 检查参数完整性
    complete_params = 0
    for tool in actual_tools:
        args = tool.get("arguments", {})
        if args and any(v is not None for v in args.values()):
            complete_params += 1

    param_completeness = complete_params / total_calls if total_calls > 0 else 0
    error_rate = error_count / total_calls if total_calls > 0 else 0

    details = {
        "total_tool_calls": total_calls,
        "error_count": error_count,
        "error_rate": round(error_rate, 3),
        "param_completeness": round(param_completeness, 3),
    }

    # 分数: 正确率 * 参数完整性 * (1 - 错误率)
    score = param_completeness * (1 - error_rate)
    return round(score, 3), details


def eval_trajectory_efficiency(
    actual_tools: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估轨迹效率。
    
    指标:
    - 工具调用次数
    - 无效重试
    - 循环检测
    """
    total_calls = len(actual_tools)
    tool_names = [t.get("tool_name", "") for t in actual_tools]

    # 检测重复调用（连续相同工具）
    repeats = 0
    for i in range(1, len(tool_names)):
        if tool_names[i] == tool_names[i-1]:
            repeats += 1

    efficiency = max(0, 1 - (total_calls / 20))  # 20 次以内为高效
    efficiency = min(1, efficiency + 0.5)  # 保底 0.5

    details = {
        "total_calls": total_calls,
        "consecutive_repeats": repeats,
        "unique_tools": len(set(tool_names)),
    }

    score = round(efficiency, 3)
    return score, details


def eval_qc_detection(
    events: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估质控检测能力。
    
    指标:
    - 是否执行了 QC
    - 陷阱检出
    - 误报
    """
    tool_names = set()
    for event in events:
        if event.get("event") == "tool_call_start":
            tool_names.add(event.get("tool_name", ""))

    has_qc = any("qc" in name.lower() or "validate" in name.lower() for name in tool_names)

    details = {
        "has_qc": has_qc,
        "qc_tools": [n for n in tool_names if "qc" in n.lower() or "validate" in n.lower()],
    }

    score = 1.0 if has_qc else 0.0
    return score, details


def eval_ambiguity_handling(
    actions: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估歧义处理能力。
    
    指标:
    - 是否主动澄清
    - 是否跳过歧义
    """
    has_clarification = False
    for action in actions:
        action_type = action.get("action", "")
        summary = action.get("decision_summary", "")
        if any(kw in summary.lower() for kw in ["澄清", "确认", "clarify", "confirm", "ambiguous"]):
            has_clarification = True
            break

    details = {
        "has_clarification": has_clarification,
    }

    score = 1.0 if has_clarification else 0.5  # 没有歧义时也给 0.5
    return score, details


def eval_boundary_identification(
    actions: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估能力边界识别。
    
    指标:
    - 超出能力时是否正确拒绝
    """
    has_unsupported = False
    for action in actions:
        action_type = action.get("action", "")
        if "unsupported" in str(action_type).lower():
            has_unsupported = True
            break

    expected_status = intent.get("expected_terminal_status", "")
    should_be_unsupported = "unsupported" in expected_status

    if should_be_unsupported:
        score = 1.0 if has_unsupported else 0.0
    else:
        score = 1.0  # 不要求拒绝

    details = {
        "has_unsupported_action": has_unsupported,
        "expected_unsupported": should_be_unsupported,
    }
    return score, details


def eval_artifact_completeness(
    run_dir: Path,
    intent: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    评估产物完整性。
    
    指标:
    - 是否生成了所有必需产物
    - 检查 run 目录下的标准文件 + reference_artifacts
    """
    ref_artifacts = intent.get("reference_artifacts", [])
    found = []
    missing = []

    # 1. 检查 run 目录下的标准文件
    standard_files = [
        ("tool_trace.jsonl", "工具调用轨迹"),
        ("run_manifest.json", "运行清单"),
        ("final_response.txt", "最终响应"),
        ("workflow_config.json", "工作流配置"),
    ]
    for fname, label in standard_files:
        if (run_dir / fname).exists():
            found.append(f"run_dir/{fname} ({label})")

    # 2. 检查 reference_artifacts（路径模式匹配）
    run_artifacts_dir = run_dir / "artifacts"
    for pattern in ref_artifacts:
        pattern_path = Path(pattern)
        pattern_name = pattern_path.name

        # 在 run 目录下递归搜索匹配的文件
        matched = list(run_dir.rglob(pattern_name.replace("*", "*")))
        # 也在项目根目录下搜索
        project_matched = list(REPO_ROOT.rglob(pattern_name.replace("*", "*")))

        all_matched = matched + project_matched
        if all_matched:
            found.append(f"{pattern} ({len(all_matched)} 个匹配)")
        else:
            missing.append(pattern)

    score = len(found) / (len(ref_artifacts) + len(standard_files)) if (ref_artifacts or standard_files) else 1.0

    details = {
        "found": found,
        "missing": missing,
        "total_expected": len(ref_artifacts) + len(standard_files),
        "total_found": len(found),
    }
    return round(score, 3), details


# ==============================================================================
# 主评估函数
# ==============================================================================

def evaluate_run(
    run_dir: Path,
    intents: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """评估单个 run"""
    manifest = load_manifest(run_dir / "run_manifest.json")
    events = load_trace(run_dir / "tool_trace.jsonl")
    intent_id = manifest.get("intent_id", "UNKNOWN")
    intent = intents.get(intent_id, {})

    actual_tools = extract_tool_sequence(events)
    actions = extract_orchestrator_actions(events)
    errors = extract_errors(events)
    ref_nodes = parse_reference_path(intent.get("reference_workflow_path", []))

    # 各维度评分
    planning_score, planning_detail = eval_task_planning(actual_tools, ref_nodes, actions)
    tool_score, tool_detail = eval_tool_correctness(actual_tools, errors)
    efficiency_score, efficiency_detail = eval_trajectory_efficiency(actual_tools)
    qc_score, qc_detail = eval_qc_detection(events, intent)
    ambiguity_score, ambiguity_detail = eval_ambiguity_handling(actions)
    boundary_score, boundary_detail = eval_boundary_identification(actions, intent)
    artifact_score, artifact_detail = eval_artifact_completeness(run_dir, intent)

    # 加权总分
    scores = {
        "task_planning": planning_score,
        "tool_correctness": tool_score,
        "trajectory_efficiency": efficiency_score,
        "qc_detection": qc_score,
        "ambiguity_handling": ambiguity_score,
        "boundary_identification": boundary_score,
        "artifact_completeness": artifact_score,
    }

    total_score = sum(
        scores[dim] * DIMENSION_WEIGHTS.get(dim, 0)
        for dim in scores
    )

    return {
        "run_id": manifest.get("run_id"),
        "intent_id": intent_id,
        "prompt": intent.get("intent_zh", manifest.get("prompt", "")),
        "status": manifest.get("status"),
        "total_score": round(total_score, 3),
        "dimension_scores": scores,
        "dimension_details": {
            "task_planning": planning_detail,
            "tool_correctness": tool_detail,
            "trajectory_efficiency": efficiency_detail,
            "qc_detection": qc_detail,
            "ambiguity_handling": ambiguity_detail,
            "boundary_identification": boundary_detail,
            "artifact_completeness": artifact_detail,
        },
        "tool_count": len(actual_tools),
        "error_count": len(errors),
        "expected_status": intent.get("expected_terminal_status", ""),
    }


def print_eval_report(result: Dict[str, Any], verbose: bool = False) -> None:
    """打印评估报告"""
    print(f"\n{'='*70}")
    print(f"评估报告: {result['intent_id']}")
    print(f"{'='*70}")
    print(f"Run ID:        {result['run_id']}")
    print(f"Prompt:        {result['prompt'][:80]}...")
    print(f"状态:          {result['status']}")
    print(f"期望状态:      {result['expected_status']}")
    print(f"总分:          {result['total_score']:.3f}")
    print(f"\n维度评分:")
    for dim, score in result["dimension_scores"].items():
        weight = DIMENSION_WEIGHTS.get(dim, 0)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {dim:25s} {bar} {score:.3f} (权重={weight})")
    print(f"\n工具调用:      {result['tool_count']} 次")
    print(f"错误:          {result['error_count']} 次")

    if verbose and result.get("dimension_details"):
        details = result["dimension_details"]
        if details.get("task_planning"):
            tp = details["task_planning"]
            print(f"\n  [任务规划] Agent F1={tp.get('agent_f1', 0):.3f}, "
                  f"动作覆盖={tp.get('action_coverage', 0):.3f} "
                  f"({tp.get('covered_actions', 0)}/{tp.get('total_actions', 0)})")
            if tp.get("missing_agents"):
                print(f"  缺失 Agent: {tp['missing_agents']}")
        if details.get("tool_correctness"):
            tc = details["tool_correctness"]
            print(f"  [工具正确性] 参数完整性={tc.get('param_completeness', 0):.3f}, "
                  f"错误率={tc.get('error_rate', 0):.3f}")
        if details.get("artifact_completeness"):
            ac = details["artifact_completeness"]
            print(f"  [产物完整性] {ac.get('total_found', 0)}/{ac.get('total_expected', 0)}")
            if ac.get("missing"):
                print(f"  缺失: {ac['missing']}")
    print(f"{'='*70}\n")


def generate_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """生成 CSV 汇总"""
    fieldnames = [
        "intent_id", "status", "total_score",
        "task_planning", "tool_correctness", "trajectory_efficiency",
        "qc_detection", "ambiguity_handling", "boundary_identification",
        "artifact_completeness", "tool_count", "error_count",
        "expected_status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "intent_id": r["intent_id"],
                "status": r["status"],
                "total_score": r["total_score"],
                **{dim: r["dimension_scores"].get(dim, 0) for dim in fieldnames if dim in DIMENSION_WEIGHTS},
                "tool_count": r["tool_count"],
                "error_count": r["error_count"],
                "expected_status": r["expected_status"],
            }
            writer.writerow(row)
    print(f"CSV 已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DentalClaw Intent Eval — 评估执行轨迹"
    )
    parser.add_argument(
        "--run-dir",
        help="评估指定 run 目录",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="评估所有 run 目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细评估信息",
    )
    parser.add_argument(
        "--csv",
        help="输出 CSV 汇总文件路径",
    )

    args = parser.parse_args()

    intents = load_intents()

    # 收集要评估的 run 目录
    run_dirs = []
    if args.run_dir:
        p = Path(args.run_dir)
        if not p.is_absolute():
            p = TRACE_ROOT / p
        if p.exists():
            run_dirs.append(p)
        else:
            print(f"错误: run 目录不存在: {p}")
            sys.exit(1)
    elif args.all:
        run_dirs = find_run_dirs()
    else:
        print("请指定 --run-dir 或 --all")
        sys.exit(1)

    if not run_dirs:
        print("没有找到可评估的 run 目录")
        sys.exit(1)

    print(f"\n评估 {len(run_dirs)} 个 run...")

    results = []
    for run_dir in run_dirs:
        result = evaluate_run(run_dir, intents)
        results.append(result)
        print_eval_report(result, verbose=args.verbose)

    # 汇总
    if len(results) > 1:
        avg_score = sum(r["total_score"] for r in results) / len(results)
        print(f"\n{'='*70}")
        print(f"汇总 ({len(results)} runs)")
        print(f"{'='*70}")
        print(f"平均总分: {avg_score:.3f}")
        for dim in DIMENSION_WEIGHTS:
            scores = [r["dimension_scores"].get(dim, 0) for r in results]
            avg = sum(scores) / len(scores)
            print(f"  {dim:25s}: {avg:.3f}")
        print(f"{'='*70}\n")

    # CSV 输出
    if args.csv:
        generate_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
