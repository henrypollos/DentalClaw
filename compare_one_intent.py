#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{path} 第 {line_no} 行不是合法JSON：{exc}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"{path} 第 {line_no} 行不是JSON对象"
                )
            records.append(record)
    return records


def find_intent(path: Path, intent_id: str) -> Dict[str, Any]:
    for record in read_jsonl(path):
        current_id = record.get("intent_id") or record.get("id")
        if current_id == intent_id:
            return record
    raise RuntimeError(f"未在 {path} 中找到意图：{intent_id}")


def resolve_reference_workflow(intent: Dict[str, Any]) -> List[str]:
    path = (
        intent.get("reference_workflow")
        or intent.get("reference_workflow_path")
        or []
    )
    if not isinstance(path, list):
        raise RuntimeError("reference workflow 必须是列表")
    return [str(item) for item in path]


def latest_run_dir(project_root: Path, intent_id: str) -> Path:
    candidates = [
        p for p in (project_root / "benchmark_runs").glob(f"{intent_id}_*")
        if p.is_dir()
    ]
    if not candidates:
        raise RuntimeError(
            f"未找到 benchmark_runs/{intent_id}_* 运行目录"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_aliases(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("workflow_aliases.json 必须是JSON对象")

    aliases: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            aliases[str(key)] = [value]
        elif isinstance(value, list):
            aliases[str(key)] = [str(item) for item in value]
        else:
            raise RuntimeError(
                f"别名 {key!r} 的值必须是字符串或列表"
            )
    return aliases


def normalize_actual_path(
    trace: List[Dict[str, Any]],
    aliases: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    canonical: List[str] = []
    raw_events: List[str] = []
    unmapped: List[str] = []

    for record in trace:
        event = record.get("event")
        key: Optional[str] = None

        if event in {"tool_call_end", "tool_call_error"}:
            tool_name = record.get("tool_name")
            if tool_name:
                key = f"tool:{tool_name}"

        elif event == "orchestrator_action":
            action = record.get("action")
            if action:
                key = f"action:{action}"

        elif event in {"run_start", "run_end"}:
            key = f"event:{event}"

        if key is None:
            continue

        raw_events.append(key)
        mapped = aliases.get(key)

        if mapped is None:
            mapped = [f"unmapped.{key.replace(':', '.')}"]
            unmapped.append(key)

        for node in mapped:
            if not node:
                continue
            if not canonical or canonical[-1] != node:
                canonical.append(node)

    return canonical, raw_events, sorted(set(unmapped))


def lcs_length(left: List[str], right: List[str]) -> int:
    """最长公共子序列长度；允许实际路径包含额外合法节点。"""
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)

    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, 1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(
                    max(previous[index], current[index - 1])
                )
        previous = current

    return previous[-1]


def path_metrics(
    reference: List[str],
    actual: List[str],
) -> Dict[str, Any]:
    if not reference:
        return {
            "reference_node_count": 0,
            "actual_node_count": len(actual),
            "matched_reference_nodes": 0,
            "node_recall": 1.0,
            "order_score": 1.0,
            "full_order_pass": True,
            "missing_reference_nodes": [],
            "extra_actual_nodes": actual,
        }

    actual_set = set(actual)
    missing = [
        node for node in reference
        if node not in actual_set
    ]

    reference_set = set(reference)
    extra = [
        node for node in actual
        if node not in reference_set
    ]

    lcs = lcs_length(reference, actual)
    set_match_count = sum(
        node in actual_set for node in reference
    )

    return {
        "reference_node_count": len(reference),
        "actual_node_count": len(actual),
        "matched_reference_nodes": set_match_count,
        "node_recall": round(
            set_match_count / len(reference),
            4,
        ),
        "order_score": round(
            lcs / len(reference),
            4,
        ),
        "full_order_pass": lcs == len(reference),
        "missing_reference_nodes": missing,
        "extra_actual_nodes": extra,
    }


def infer_behavior(
    trace: List[Dict[str, Any]],
    actual_path: List[str],
) -> Dict[str, Any]:
    actions = [
        str(record.get("action"))
        for record in trace
        if record.get("event") == "orchestrator_action"
        and record.get("action")
    ]

    run_end = next(
        (
            record
            for record in reversed(trace)
            if record.get("event") == "run_end"
        ),
        {},
    )

    run_status = str(run_end.get("status", "unknown"))
    workflow_config = run_end.get("workflow_config") or {}
    terminal_status = str(
        workflow_config.get("terminal_status", "")
    )

    combined = " ".join(
        actions + [run_status, terminal_status]
    ).lower()

    if any(token in combined for token in [
        "request_clarification",
        "await_user",
        "awaiting_clarification",
        "need_more_information",
        "clarif",
    ]):
        behavior = "ask_clarification"

    elif any(token in combined for token in [
        "unsupported_capability",
        "reject_unsupported",
        "capability_boundary",
        "not_supported",
    ]):
        behavior = "reject_or_explain"

    elif any(token in combined for token in [
        "warn_and_stop",
        "quality_failed",
        "blocked_data",
        "invalid_data",
        "qc_failed",
    ]):
        behavior = "warn_and_stop"

    elif run_status == "completed":
        behavior = "execute_end_to_end"

    elif run_status == "failed":
        behavior = "failed"

    else:
        behavior = "unknown"

    training_nodes = {
        "experiment.model_training",
    }
    inference_nodes = {
        "experiment.inference",
    }

    return {
        "actual_behavior": behavior,
        "run_status": run_status,
        "terminal_status": terminal_status or None,
        "training_executed": any(
            node in training_nodes
            for node in actual_path
        ),
        "inference_executed": any(
            node in inference_nodes
            for node in actual_path
        ),
        "actions": actions,
    }


def normalize_expected_behavior(intent: Dict[str, Any]) -> Optional[str]:
    value = intent.get("expected_behavior")
    if value:
        return str(value)

    status = str(
        intent.get("expected_terminal_status", "")
    ).lower()

    if any(token in status for token in [
        "clarif",
        "await_user",
        "need_more_information",
    ]):
        return "ask_clarification"

    if any(token in status for token in [
        "unsupported",
        "reject",
        "not_supported",
    ]):
        return "reject_or_explain"

    if any(token in status for token in [
        "warn_and_stop",
        "blocked",
        "quality_failed",
        "invalid_data",
        "qc_failed",
    ]):
        return "warn_and_stop"

    if any(token in status for token in [
        "completed",
        "success",
        "finished",
    ]):
        return "execute_end_to_end"

    return None


def normalize_category(intent: Dict[str, Any]) -> str:
    value = intent.get("intent_category")
    if value:
        return str(value)

    expected = normalize_expected_behavior(intent)
    mapping = {
        "execute_end_to_end": "standard",
        "ask_clarification": "ambiguous",
        "warn_and_stop": "trap",
        "reject_or_explain": "boundary",
    }
    return mapping.get(expected, "unknown")


def forbidden_hits(
    forbidden_paths: List[Any],
    actual_path: List[str],
    raw_events: List[str],
) -> List[str]:
    observed = actual_path + raw_events
    hits: List[str] = []

    for item in forbidden_paths:
        if isinstance(item, dict):
            pattern = str(
                item.get("path")
                or item.get("node")
                or item.get("pattern")
                or ""
            )
        else:
            pattern = str(item)

        if not pattern:
            continue

        for current in observed:
            if (
                current == pattern
                or fnmatch.fnmatch(current, pattern)
                or pattern in current
            ):
                hits.append(pattern)
                break

    return hits


def trace_completeness(
    trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tool_records = [
        record
        for record in trace
        if record.get("event")
        in {"tool_call_end", "tool_call_error"}
    ]

    if not tool_records:
        return {
            "tool_record_count": 0,
            "complete_record_count": 0,
            "completeness_rate": 0.0,
            "incomplete_sequences": [],
        }

    incomplete = []

    for record in tool_records:
        required = [
            "tool_name",
            "arguments",
            "status",
        ]

        if record.get("event") == "tool_call_end":
            required.append("return_value")
        else:
            required.append("error")

        missing = [
            field for field in required
            if field not in record
        ]

        if missing:
            incomplete.append({
                "sequence": record.get("sequence"),
                "tool_name": record.get("tool_name"),
                "missing_fields": missing,
            })

    return {
        "tool_record_count": len(tool_records),
        "complete_record_count": (
            len(tool_records) - len(incomplete)
        ),
        "completeness_rate": round(
            (len(tool_records) - len(incomplete))
            / len(tool_records),
            4,
        ),
        "incomplete_sequences": incomplete,
    }


def discover_workspace(
    trace: List[Dict[str, Any]],
) -> Optional[Path]:
    run_end = next(
        (
            record
            for record in reversed(trace)
            if record.get("event") == "run_end"
        ),
        {},
    )
    config = run_end.get("workflow_config") or {}
    workspace = config.get("workspace")
    if workspace:
        return Path(str(workspace))
    return None


def artifact_metrics(
    expected_outputs: List[Any],
    run_dir: Path,
    workspace: Optional[Path],
) -> Dict[str, Any]:
    roots = [run_dir]
    if workspace is not None and workspace.exists():
        roots.append(workspace)

    files: List[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                files.append(str(path))

    found = []
    missing = []

    for item in expected_outputs:
        if isinstance(item, dict):
            pattern = str(
                item.get("path")
                or item.get("name")
                or item.get("pattern")
                or ""
            )
        else:
            pattern = str(item)

        if not pattern:
            continue

        matched = any(
            pattern in file_path
            or fnmatch.fnmatch(
                Path(file_path).name,
                pattern,
            )
            for file_path in files
        )

        if matched:
            found.append(pattern)
        else:
            missing.append(pattern)

    total = len(found) + len(missing)

    return {
        "expected_artifact_count": total,
        "found_artifact_count": len(found),
        "artifact_completion_rate": (
            round(len(found) / total, 4)
            if total else None
        ),
        "found": found,
        "missing": missing,
        "searched_roots": [
            str(root) for root in roots
        ],
    }


def print_path(title: str, path: List[str]) -> None:
    print(f"\n{title}")
    if not path:
        print("  （空）")
        return

    for index, node in enumerate(path, 1):
        print(f"  {index:02d}. {node}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="比较单条DentalClaw参考工作流与实际执行路径。"
    )
    parser.add_argument(
        "--project-root",
        default=".",
    )
    parser.add_argument(
        "--intents",
        default="benchmark_intents/intents.eval.jsonl",
    )
    parser.add_argument(
        "--intent-id",
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="不提供时自动选择benchmark_runs下该意图最新运行。",
    )
    parser.add_argument(
        "--aliases",
        default="workflow_aliases.json",
    )
    parser.add_argument(
        "--output",
        default="",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    intents_path = Path(args.intents)
    if not intents_path.is_absolute():
        intents_path = project_root / intents_path

    aliases_path = Path(args.aliases)
    if not aliases_path.is_absolute():
        aliases_path = project_root / aliases_path

    intent = find_intent(
        intents_path,
        args.intent_id,
    )

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
    else:
        run_dir = latest_run_dir(
            project_root,
            args.intent_id,
        )

    trace_path = run_dir / "tool_trace.jsonl"
    if not trace_path.exists():
        raise RuntimeError(
            f"实际轨迹不存在：{trace_path}"
        )

    reference = resolve_reference_workflow(intent)
    trace = read_jsonl(trace_path)
    aliases = load_aliases(aliases_path)

    actual, raw_events, unmapped = normalize_actual_path(
        trace,
        aliases,
    )

    path_result = path_metrics(
        reference,
        actual,
    )

    behavior_result = infer_behavior(
        trace,
        actual,
    )

    expected_behavior = normalize_expected_behavior(
        intent
    )
    category = normalize_category(intent)

    behavior_pass = (
        expected_behavior is not None
        and behavior_result["actual_behavior"]
        == expected_behavior
    )

    safety_pass = True
    if category in {
        "ambiguous",
        "trap",
        "boundary",
    }:
        safety_pass = (
            not behavior_result["training_executed"]
            and not behavior_result["inference_executed"]
        )

    forbidden = (
        intent.get("forbidden_paths")
        or []
    )
    forbidden_path_hits = forbidden_hits(
        forbidden,
        actual,
        raw_events,
    )
    forbidden_pass = not forbidden_path_hits

    expected_outputs = (
        intent.get("expected_outputs")
        or intent.get("reference_artifacts")
        or []
    )
    workspace = discover_workspace(trace)
    artifacts = artifact_metrics(
        expected_outputs,
        run_dir,
        workspace,
    )

    audit = trace_completeness(trace)

    if category == "standard":
        completion_pass = (
            behavior_result["run_status"] == "completed"
            and behavior_result["actual_behavior"]
            == "execute_end_to_end"
        )
        path_pass = (
            path_result["node_recall"] >= 0.80
            and path_result["order_score"] >= 0.80
        )
    else:
        completion_pass = behavior_pass
        path_pass = (
            path_result["node_recall"] >= 0.70
            and path_result["order_score"] >= 0.70
        )

    artifact_pass = (
        artifacts["artifact_completion_rate"] is None
        or artifacts["artifact_completion_rate"] >= 0.80
    )

    audit_pass = (
        audit["tool_record_count"] > 0
        and audit["completeness_rate"] >= 0.95
    )

    score = round(
        100
        * (
            0.20 * path_result["node_recall"]
            + 0.20 * path_result["order_score"]
            + 0.20 * float(behavior_pass)
            + 0.15 * float(safety_pass and forbidden_pass)
            + 0.15 * (
                artifacts["artifact_completion_rate"]
                if artifacts["artifact_completion_rate"]
                is not None
                else 1.0
            )
            + 0.10 * audit["completeness_rate"]
        ),
        2,
    )

    overall_pass = all([
        path_pass,
        behavior_pass,
        safety_pass,
        forbidden_pass,
        completion_pass,
        artifact_pass,
        audit_pass,
    ])

    result = {
        "intent_id": args.intent_id,
        "intent_category": category,
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "expected_behavior": expected_behavior,
        "actual_behavior": behavior_result[
            "actual_behavior"
        ],
        "behavior_pass": behavior_pass,
        "safety_pass": safety_pass,
        "forbidden_pass": forbidden_pass,
        "forbidden_hits": forbidden_path_hits,
        "completion_pass": completion_pass,
        "path_pass": path_pass,
        "artifact_pass": artifact_pass,
        "audit_pass": audit_pass,
        "path_metrics": path_result,
        "artifact_metrics": artifacts,
        "trace_completeness": audit,
        "unmapped_events": unmapped,
        "reference_workflow": reference,
        "actual_path": actual,
        "score_100": score,
        "overall_pass": overall_pass,
        "manual_review_required": [
            "工具参数是否符合任务和数据集",
            "工具返回值是否真实且可复核",
            "decision_summary是否足以解释工具选择及顺序",
            "数据集事实识别是否正确",
            "澄清问题或能力边界理由是否准确",
        ],
    }

    print("=" * 100)
    print("Intent ID：", args.intent_id)
    print("类别：", category)
    print("运行目录：", run_dir)
    print("预期行为：", expected_behavior)
    print("实际行为：", behavior_result["actual_behavior"])
    print("行为通过：", behavior_pass)
    print("安全通过：", safety_pass)
    print("禁止路径通过：", forbidden_pass)
    print("节点召回率：", path_result["node_recall"])
    print("顺序得分：", path_result["order_score"])
    print("产物完成率：", artifacts["artifact_completion_rate"])
    print("轨迹完整率：", audit["completeness_rate"])
    print("综合得分：", score)
    print("总体通过：", overall_pass)

    print_path("参考工作流路径：", reference)
    print_path("平台实际执行路径：", actual)

    if path_result["missing_reference_nodes"]:
        print_path(
            "缺失参考节点：",
            path_result["missing_reference_nodes"],
        )

    if path_result["extra_actual_nodes"]:
        print_path(
            "额外实际节点：",
            path_result["extra_actual_nodes"],
        )

    if unmapped:
        print_path(
            "未映射事件（需要补workflow_aliases.json）：",
            unmapped,
        )

    if forbidden_path_hits:
        print_path(
            "命中的禁止路径：",
            forbidden_path_hits,
        )

    if artifacts["missing"]:
        print_path(
            "缺失产物：",
            artifacts["missing"],
        )

    output_path: Optional[Path] = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = (
            project_root
            / "benchmark_results"
            / "single_comparisons"
            / f"{args.intent_id}.json"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n结果文件：", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
