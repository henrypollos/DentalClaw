#!/usr/bin/env python3
"""
OpenClaw Runner — 通过 `openclaw agent` CLI 真正调用 LLM 执行测试意图。

用法：
    python benchmark_trace/openclaw_runner.py --intent-id DCI-TDD-SEG2D-001
    python benchmark_trace/openclaw_runner.py --task-family segmentation_2d
    python benchmark_trace/openclaw_runner.py --all
    python benchmark_trace/openclaw_runner.py --all --model gpt-5.4-mini

与 run_intent.py 的区别：
    run_intent.py → 硬编码模拟（假 trace）
    openclaw_runner.py → `openclaw agent --json` → 真实 LLM 决策（真 trace）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_trace.trace_recorder import (
    start_run,
    end_run,
    record_orchestrator_action,
    emit_event,
    _safe_value,
    TRACE_ROOT,
)

# 意图源文件
INTENT_SOURCE = REPO_ROOT / "benchmark_intents" / "intents.jsonl"


def load_intents(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or INTENT_SOURCE
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    for intent in load_intents():
        if intent.get("id") == intent_id:
            return intent
    return None


def filter_intents(
    task_family: Optional[str] = None,
    dataset: Optional[str] = None,
) -> List[Dict[str, Any]]:
    results = []
    for intent in load_intents():
        if task_family and intent.get("task_family") != task_family:
            continue
        if dataset and intent.get("dataset") != dataset:
            continue
        results.append(intent)
    return results


def call_openclaw_agent(
    prompt: str,
    model: str = "qwen3-coder-plus",
    timeout: int = 600,
    dry_run: bool = True,
    thinking: str = "medium",
) -> Dict[str, Any]:
    """
    调用 `openclaw agent` CLI，返回 JSON 结构。

    dry_run=True: 在 prompt 前加 "DRY-RUN: " 前缀，告知 Agent 只生成命令不实际执行训练。
    """
    if dry_run and not prompt.startswith("DRY-RUN:"):
        full_prompt = (
            "DRY-RUN MODE: Do NOT execute training commands. "
            "Only generate the training command for review. "
            f"User request: {prompt}"
        )
    else:
        full_prompt = prompt

    cmd = [
        "openclaw",
        "agent",
        "--agent", "main",
        "--message", full_prompt,
        "--json",
        "--timeout", str(timeout),
        "--thinking", thinking,
    ]

    # 不通过 openclaw agent --model 指定模型（让它用 agent 配置里的默认模型）
    # 如需覆盖，加: ["--local"] + env vars

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr[:2000] or "openclaw agent returned non-zero",
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }

        # openclaw agent --json 输出 JSON
        try:
            parsed = json.loads(result.stdout)
            return {"status": "completed", **parsed}
        except json.JSONDecodeError:
            return {
                "status": "completed_raw",
                "raw_output": result.stdout[:5000],
                "stderr": result.stderr[:1000],
            }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"openclaw agent timed out after {timeout}s"}
    except FileNotFoundError:
        return {"status": "error", "error": "openclaw CLI not found. Is OpenClaw installed?"}


def convert_openclaw_output_to_trace(
    openclaw_result: Dict[str, Any],
    run_id: str,
    intent: Dict[str, Any],
) -> None:
    """
    将 openclaw agent 的 JSON 输出转换为 trace_recorder 格式的事件，
    写入 tool_trace.jsonl。
    """
    # 记录编排决策
    record_orchestrator_action(
        agent="MainAgent",
        action="openclaw_agent_call",
        reason=f"通过 OpenClaw Agent 执行意图: {intent.get('intent_zh', '')[:80]}",
        target={
            "intent_id": intent.get("id", intent.get("intent_id", "")),
            "prompt": intent.get("intent_zh", ""),
            "dry_run": True,
        },
    )

    # 记录 openclaw 返回
    status = openclaw_result.get("status", "unknown")
    if status == "completed":
        emit_event(
            "openclaw_response",
            run_id=run_id,
            intent_id=intent.get("id", ""),
            status="success",
            raw_result=json.dumps(
                _safe_value(openclaw_result),
                ensure_ascii=False,
                default=str,
            )[:5000],
        )
    else:
        emit_event(
            "openclaw_response",
            run_id=run_id,
            intent_id=intent.get("id", ""),
            status="error",
            error=openclaw_result.get("error", "unknown"),
            raw_result=json.dumps(
                _safe_value(openclaw_result),
                ensure_ascii=False,
                default=str,
            )[:5000],
        )


def run_intent_with_openclaw(
    intent: Dict[str, Any],
    model: str = "qwen3-coder-plus",
    dry_run: bool = True,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    用 openclaw agent 执行一条意图。
    """
    intent_id = intent.get("id") or intent.get("intent_id", "UNKNOWN")
    prompt = intent.get("intent_zh", "")
    dataset = intent.get("dataset", "")
    task_family = intent.get("task_family", "")

    # 设置环境变量
    if dry_run:
        os.environ["DENTALCLAW_DRY_RUN"] = "1"
    else:
        os.environ.pop("DENTALCLAW_DRY_RUN", None)

    # 启动 trace
    run_id = start_run(
        intent_id=intent_id,
        prompt=prompt,
        dataset=dataset,
        task_type=task_family,
    )

    result = {
        "run_id": run_id,
        "intent_id": intent_id,
        "status": "running",
        "openclaw_status": None,
        "error": None,
    }

    try:
        # 调用 OpenClaw
        openclaw_result = call_openclaw_agent(
            prompt=prompt,
            model=model,
            timeout=timeout,
            dry_run=dry_run,
        )

        result["openclaw_status"] = openclaw_result.get("status")

        # 转换为 trace 事件
        convert_openclaw_output_to_trace(openclaw_result, run_id, intent)

        # 结束 trace
        if openclaw_result.get("status") in ("completed",):
            result["status"] = "completed"
            end_run(
                status="completed",
                final_response=openclaw_result.get("raw_output", "")[:1000]
                or json.dumps(openclaw_result, ensure_ascii=False, default=str)[:1000],
                workflow_config={
                    "intent_id": intent_id,
                    "dry_run": dry_run,
                    "model": model,
                    "openclaw_status": openclaw_result.get("status"),
                },
            )
        else:
            result["status"] = "failed"
            error_msg = openclaw_result.get("error", "unknown")
            result["error"] = error_msg
            end_run(
                status="failed",
                final_response=f"OpenClaw agent 调用失败: {error_msg}",
                error=error_msg,
            )

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        end_run(
            status="failed",
            final_response=f"运行异常: {e}",
            error=str(e),
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Runner — 通过 openclaw agent CLI 真 LLM 执行测试意图"
    )
    parser.add_argument("--intent-id", help="运行指定 ID 的意图")
    parser.add_argument(
        "--task-family",
        choices=["segmentation_2d", "segmentation_3d", "detection", "classification"],
        help="按 task_family 过滤",
    )
    parser.add_argument("--dataset", choices=["TDD", "ToothFairy3", "Private2D"], help="按数据集过滤")
    parser.add_argument("--all", action="store_true", help="运行所有意图")
    parser.add_argument("--model", default="qwen3-coder-plus", help="LLM 模型名称")
    parser.add_argument("--dry-run", action="store_true", default=True, help="DRY-RUN 模式（默认开启）")
    parser.add_argument("--no-dry-run", action="store_true", help="关闭 DRY-RUN，真正执行训练")
    parser.add_argument("--timeout", type=int, default=600, help="每条意图超时秒数")
    parser.add_argument("--list", action="store_true", help="列出所有意图")
    parser.add_argument("--intents-file", default=str(INTENT_SOURCE), help="意图 JSONL 路径")

    args = parser.parse_args()

    dry_run = not args.no_dry_run

    # --list
    if args.list:
        intents = load_intents(Path(args.intents_file) if args.intents_file != str(INTENT_SOURCE) else None)
        print(f"\n可用意图 ({len(intents)} 条):")
        for i in intents:
            iid = i.get("id") or i.get("intent_id", "")
            ds = i.get("dataset", "")
            tf = i.get("task_family", "")
            zh = (i.get("intent_zh") or "")[:60]
            print(f"  {iid:30s} {ds:12s} {tf:18s} {zh}")
        print()
        return

    # 收集意图
    intents_to_run = []
    if args.intent_id:
        intent = find_intent(args.intent_id)
        if intent:
            intents_to_run.append(intent)
        else:
            print(f"未找到意图: {args.intent_id}")
            sys.exit(1)
    elif args.all:
        intents_to_run = load_intents(Path(args.intents_file) if args.intents_file != str(INTENT_SOURCE) else None)
    else:
        intents_to_run = filter_intents(task_family=args.task_family, dataset=args.dataset)

    if not intents_to_run:
        print("无匹配意图。用 --list 查看。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"OpenClaw Runner: {len(intents_to_run)} 条意图")
    print(f"Model: {args.model}   Dry-run: {dry_run}   Timeout: {args.timeout}s")
    print(f"{'='*60}")

    results = []
    for idx, intent in enumerate(intents_to_run, 1):
        iid = intent.get("id") or intent.get("intent_id", "UNKNOWN")
        prompt = (intent.get("intent_zh") or "")[:80]
        print(f"\n[{idx}/{len(intents_to_run)}] {iid}")
        print(f"  Prompt: {prompt}...")

        start = time.time()
        result = run_intent_with_openclaw(
            intent=intent,
            model=args.model,
            dry_run=dry_run,
            timeout=args.timeout,
        )
        elapsed = time.time() - start

        status = "✅" if result["status"] == "completed" else "❌"
        print(f"  {status} Status: {result['status']} ({elapsed:.1f}s)")
        if result.get("error"):
            print(f"  Error: {result['error'][:120]}")
        results.append(result)

    # 汇总
    ok = sum(1 for r in results if r["status"] == "completed")
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"完成: {ok} 成功 / {fail} 失败")
    print(f"Trace 目录: {TRACE_ROOT}")
    print(f"下一步: python benchmark_trace/eval_intents.py --all")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
