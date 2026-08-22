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
import shlex
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
from benchmark_trace.run_intent import select_experiment_intents

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


def discover_available_models() -> List[str]:
    """查询 OpenClaw 当前已配置的可用模型列表。"""
    try:
        result = subprocess.run(
            "openclaw models list",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENCLAW_NO_COLOR": "1"},
        )
        output = result.stdout or result.stderr or ""
        names = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("Model") or line.startswith("[plugins]"):
                continue
            # 收集所有模型行 (modelstudio/*, deepseek/*, zai/*, etc.)
            parts = line.split()
            if parts and "/" in parts[0]:
                names.append(parts[0])
        return names
    except Exception:
        return []


def pick_model_candidates(model: Optional[str]) -> List[str]:
    """优先使用显式指定模型；否则按当前可用模型列表依次尝试（DeepSeek 优先于欠费的 ModelStudio）。"""
    if model:
        return [model]
    available = discover_available_models()
    # DeepSeek 和 GLM 优先（ModelStudio 欠费时仍可用）
    preferred = [
        "deepseek/deepseek-v4-pro",
        "zai/glm-5",
        "modelstudio/qwen3.5-plus",
        "modelstudio/qwen3.5-flash",
        "modelstudio/qwen3-coder-plus",
        "modelstudio/qwen3.6-plus",
    ]
    ordered = [candidate for candidate in preferred if candidate in available]
    ordered.extend(candidate for candidate in available if candidate not in ordered)
    return ordered or ["deepseek/deepseek-v4-pro"]


def build_platform_mvp_prompt(intent: str) -> str:
    """构造一个强约束的平台 MVP prompt，要求 OpenClaw 返回结构化 JSON。"""
    return (
        "You are orchestrating the DentalClaw platform MVP. "
        "Invoke platform_mvp/run_platform_mvp.py with the parsed JSON payload. "
        "Return a JSON object with exactly these keys: "
        '{"intent": "<user request>", "execute": true, "case_id": "100", '
        '"reuse_inference": true, "reuse_fullflow_run": null, '
        '"run_dir": "artifacts/platform_mvp_runs/openclaw_platform_mvp"}. '
        "Do not include markdown, commentary, or extra keys. "
        f"User request: {intent}"
    )


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列（颜色码等），防止污染 JSON 解析和意图文本。"""
    import re as _re
    return _re.sub(r"\x1b\[[0-9;]*m", "", text)


def _clean_stdout(raw: str) -> str:
    """先剥离 ANSI 颜色码，再按行过滤 [plugins] 日志，最后尝试定位 JSON 开头。"""
    clean = _strip_ansi(raw)
    # 按行过滤 [plugins] 前缀
    lines = [
        line for line in clean.split("\n")
        if not line.strip().startswith("[plugins]")
        and not line.strip().startswith("[model-fallback")
        and line.strip() != ""
    ]
    text = "\n".join(lines).strip()
    # 如果在文本中间嵌入了 JSON（以 { 开头），优先从第一个 { 开始截取
    brace_idx = text.find("{")
    if brace_idx > 0:
        text = text[brace_idx:]
    return text


def normalize_platform_mvp_payload(payload: Any, fallback_intent: Optional[str] = None) -> Dict[str, Any]:
    """把 OpenClaw 返回值规范化为平台 MVP 的执行参数。"""
    if isinstance(payload, str):
        # 用力剥离可能的 ANSI 码和日志前缀后再试 JSON
        clean = _strip_ansi(payload)
        # 尝试从第一个 { 开始解析
        brace_idx = clean.find("{")
        if brace_idx >= 0:
            clean = clean[brace_idx:]
        try:
            payload = json.loads(clean)
        except json.JSONDecodeError:
            # 解析失败时绝不把原始噪声当 intent；回退到调用方传入的 fallback
            payload = {"intent": fallback_intent or payload.strip()[:200]}

    # 展开 OpenClaw 的三层嵌套: { ..., result: { payloads: [{text: "{...}"}] } }
    if isinstance(payload, dict):
        inner = payload.get("result")
        if isinstance(inner, dict):
            payload = inner
        payloads_list = payload.get("payloads")
        if isinstance(payloads_list, list) and payloads_list:
            first = payloads_list[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                try:
                    payload = json.loads(first["text"])
                except json.JSONDecodeError:
                    payload = {"intent": first["text"]}
        if isinstance(payload, dict) and "payload" in payload and isinstance(payload["payload"], dict):
            payload = payload["payload"]

    if not isinstance(payload, dict):
        payload = {"intent": fallback_intent or ""}

    intent_value = payload.get("intent") or payload.get("prompt") or fallback_intent or ""
    execute_value = payload.get("execute", True)
    if isinstance(execute_value, str):
        execute_value = execute_value.lower() in {"1", "true", "yes", "y"}

    reuse_inference = payload.get("reuse_inference", True)
    if isinstance(reuse_inference, str):
        reuse_inference = reuse_inference.lower() in {"1", "true", "yes", "y"}

    # 确保每次运行使用独立目录，避免 30 条意图互相覆盖
    run_dir = str(payload.get("run_dir") or "artifacts/platform_mvp_runs/openclaw_platform_mvp")

    return {
        "intent": str(intent_value).strip(),
        "execute": bool(execute_value),
        "case_id": str(payload.get("case_id") or "100"),
        "reuse_inference": bool(reuse_inference),
        "reuse_fullflow_run": payload.get("reuse_fullflow_run"),
        "run_dir": run_dir,
    }


def set_default_model(model_name: str) -> Dict[str, Any]:
    """通过 OpenClaw CLI 将默认模型切换为指定模型。"""
    try:
        result = subprocess.run(
            f"openclaw models set {shlex.quote(model_name)}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENCLAW_NO_COLOR": "1"},
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[:1000],
            "stderr": result.stderr[:1000],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def call_openclaw_agent(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = 600,
    dry_run: bool = True,
    thinking: str = "medium",
    mode: str = "default",
) -> Dict[str, Any]:
    """
    调用 `openclaw agent` CLI，返回 JSON 结构。

    dry_run=True: 在 prompt 前加 "DRY-RUN: " 前缀，告知 Agent 只生成命令不实际执行训练。
    """
    if mode == "platform_mvp":
        full_prompt = build_platform_mvp_prompt(prompt)
    elif dry_run and not prompt.startswith("DRY-RUN:"):
        full_prompt = (
            "DRY-RUN MODE: Do NOT execute training commands. "
            "Only generate the training command for review. "
            f"User request: {prompt}"
        )
    else:
        full_prompt = prompt

    model_candidates = pick_model_candidates(model)
    available_models = discover_available_models()
    last_error = None

    # Fast-path: 短超时时只试第一个候选模型
    if timeout <= 10:
        model_candidates = model_candidates[:1]

    for model_name in model_candidates:
        set_default_model(model_name)

        # 转义 prompt 中的特殊字符，防止 shell 注入
        escaped_prompt = full_prompt.replace("'", "'\\''")

        # 使用 shell 模式，stderr 重定向到 /dev/null 避免 plugin 警告污染 stdout
        cmd = (
            f"openclaw agent --agent main "
            f"--message '{escaped_prompt}' "
            f"--json --timeout {timeout} --thinking {thinking}"
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                cwd=str(REPO_ROOT),
                env={**os.environ, "OPENCLAW_NO_COLOR": "1"},
            )

            # 清理 stdout 中的 ANSI 颜色码和 [plugins] 行
            clean_stdout = _clean_stdout(result.stdout)

            # 尝试从 stderr 中提取更明确的错误信息
            clean_stderr = "\n".join(
                line for line in result.stderr.split("\n")
                if not line.startswith("[plugins]") and not line.startswith("[model-fallback")
            ).strip()

            if result.returncode != 0:
                last_error = {
                    "status": "error",
                    "error": clean_stderr[:2000] or f"openclaw agent exit code={result.returncode}",
                    "stdout": clean_stdout[:2000],
                    "stderr": clean_stderr[:2000],
                    "selected_model": model_name,
                    "available_models": available_models,
                }
                continue

            # 尝试解析 JSON
            if clean_stdout:
                try:
                    parsed = json.loads(clean_stdout)
                    # OpenClaw JSON 格式: {"runId":..., "status":"ok", "result":{...}}
                    openclaw_status = parsed.get("status", "unknown")
                    if openclaw_status == "ok":
                        return {"status": "completed", **parsed, "selected_model": model_name}
                    last_error = {
                        "status": "agent_error",
                        "error": f"openclaw agent returned status={openclaw_status}",
                        "result": parsed,
                        "selected_model": model_name,
                        "available_models": available_models,
                    }
                    continue
                except json.JSONDecodeError:
                    pass

            return {
                "status": "completed_raw",
                "raw_output": clean_stdout[:5000],
                "stderr": clean_stderr[:1000],
                "selected_model": model_name,
                "available_models": available_models,
            }

        except subprocess.TimeoutExpired:
            last_error = {
                "status": "timeout",
                "error": f"openclaw agent timed out after {timeout}s",
                "selected_model": model_name,
                "available_models": available_models,
            }
            continue
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "openclaw CLI not found. Is OpenClaw installed?",
                "selected_model": model,
                "available_models": [],
            }

    return last_error or {
        "status": "error",
        "error": "openclaw agent failed for all tried models",
        "selected_model": model_candidates[-1],
        "available_models": available_models,
    }


def execute_platform_mvp_via_openclaw(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = 600,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """调用 OpenClaw 生成平台 MVP 参数，再执行现有平台入口。"""
    openclaw_result = call_openclaw_agent(
        prompt=prompt,
        model=model,
        timeout=timeout,
        dry_run=dry_run,
        mode="platform_mvp",
    )

    inner = openclaw_result.get("result") if isinstance(openclaw_result.get("result"), dict) else None
    payload = normalize_platform_mvp_payload(
        inner or openclaw_result.get("raw_output") or {},
        prompt,
    )

    # 为每次运行生成独立目录，避免 30 条意图互相覆盖
    import uuid
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = Path(payload["run_dir"])
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir = run_dir.parent / (run_dir.name + "_" + suffix)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 记录一次平台 MVP delegate 工具调用
    call_id = f"platform_mvp_{suffix}"
    emit_event(
        "tool_call_start",
        call_id=call_id,
        agent="MainAgent",
        tool_name="platform_mvp.delegate",
        arguments={"intent": payload["intent"], "execute": payload["execute"]},
        decision_summary=f"OpenClaw → platform_mvp delegation for: {prompt[:80]}",
    )

    command = [
        sys.executable,
        str(REPO_ROOT / "platform_mvp/run_platform_mvp.py"),
        "--intent",
        payload["intent"],
        "--run-dir",
        str(run_dir),
        "--case-id",
        payload["case_id"],
    ]
    if payload["execute"]:
        command.append("--execute")
    if payload["reuse_inference"]:
        command.append("--reuse-inference")
    if payload.get("reuse_fullflow_run"):
        command.extend(["--reuse-fullflow-run", str(payload["reuse_fullflow_run"])])

    stdout_path = run_dir / "logs" / "openclaw_platform_mvp_stdout.log"
    stderr_path = run_dir / "logs" / "openclaw_platform_mvp_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start_ts = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    elapsed_ms = round((time.time() - start_ts) * 1000, 2)

    # 读取平台 plan 和 execution result 作为返回值
    plan_path = run_dir / "platform_plan.json"
    execution_path = run_dir / "execution_result.json"
    plan_data = None
    exec_data = None
    try:
        if plan_path.exists():
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        if execution_path.exists():
            exec_data = json.loads(execution_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass

    execution_result = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "stdout": str(stdout_path.relative_to(REPO_ROOT)) if stdout_path.is_absolute() else str(stdout_path),
        "stderr": str(stderr_path.relative_to(REPO_ROOT)) if stderr_path.is_absolute() else str(stderr_path),
        "platform_payload": payload,
        "plan_path": str(plan_path.relative_to(REPO_ROOT)) if plan_path and plan_path.exists() else None,
        "execution_path": str(execution_path.relative_to(REPO_ROOT)) if execution_path and execution_path.exists() else None,
        "platform_supported": plan_data.get("supported") if plan_data else None,
        "platform_executable": plan_data.get("executable") if plan_data else None,
        "platform_selected_method": (plan_data.get("selected_method") or {}).get("id") if plan_data else None,
    }

    emit_event(
        "tool_call_end",
        call_id=call_id,
        agent="MainAgent",
        tool_name="platform_mvp.delegate",
        arguments={"intent": payload["intent"]},
        return_value=_safe_value(execution_result),
        duration_ms=elapsed_ms,
        status=execution_result["status"],
        decision_summary=f"platform_supported={execution_result.get('platform_supported')}, "
                        f"executable={execution_result.get('platform_executable')}, "
                        f"method={execution_result.get('platform_selected_method')}",
    )

    if completed.returncode != 0:
        return {**execution_result, "openclaw_result": openclaw_result}

    return {**execution_result, "openclaw_result": openclaw_result}


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
    platform_mvp: bool = False,
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
        if platform_mvp:
            platform_result = execute_platform_mvp_via_openclaw(
                prompt=prompt,
                model=model,
                timeout=timeout,
                dry_run=dry_run,
            )
            result["platform_mvp_status"] = platform_result.get("status")
            result["platform_payload"] = platform_result.get("platform_payload")
            result["platform_run_dir"] = str(Path(platform_result.get("stdout", "")).parent.parent)
            result["platform_stdout"] = platform_result.get("stdout")
            result["platform_stderr"] = platform_result.get("stderr")
            result["openclaw_status"] = platform_result.get("openclaw_result", {}).get("status")

            # 根据平台实际决策确定语义状态，而不只是"管道没崩"
            platform_supported = platform_result.get("platform_supported")
            platform_executable = platform_result.get("platform_executable")

            if platform_result.get("status") != "completed":
                result["status"] = "failed"
                result["error"] = platform_result.get("openclaw_result", {}).get("error", "platform_mvp_failed")
                semantic_status = "failed"
            elif platform_supported is True and platform_executable is True:
                result["status"] = "executed"
                semantic_status = "executed"
            elif platform_supported is True and platform_executable is False:
                result["status"] = "planned_adapter_identified"
                semantic_status = "planned_adapter_identified"
            elif platform_supported is False:
                result["status"] = "correctly_rejected"
                semantic_status = "correctly_rejected"
            else:
                result["status"] = "completed"
                semantic_status = "completed"

            end_run(
                status=semantic_status,
                final_response=json.dumps(platform_result, ensure_ascii=False, default=str)[:2000],
                workflow_config={
                    "intent_id": intent_id,
                    "dry_run": dry_run,
                    "model": model,
                    "platform_mvp": True,
                    "platform_payload": platform_result.get("platform_payload"),
                    "platform_supported": platform_result.get("platform_supported"),
                    "platform_executable": platform_result.get("platform_executable"),
                    "platform_selected_method": platform_result.get("platform_selected_method"),
                },
            )
            return result

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
    parser.add_argument("--model", default=None, help="LLM 模型名称；留空时自动从当前可用模型中选择")
    parser.add_argument("--dry-run", action="store_true", default=True, help="DRY-RUN 模式（默认开启）")
    parser.add_argument("--no-dry-run", action="store_true", help="关闭 DRY-RUN，真正执行训练")
    parser.add_argument("--timeout", type=int, default=600, help="每条意图超时秒数")
    parser.add_argument("--platform-mvp", action="store_true", help="让 OpenClaw 生成平台 MVP 参数并执行现有平台入口")
    parser.add_argument("--list", action="store_true", help="列出所有意图")
    parser.add_argument("--intents-file", default=str(INTENT_SOURCE), help="意图 JSONL 路径")
    parser.add_argument(
        "--experiment-suite",
        action="store_true",
        help="仅运行一个小型代表性实验集（按类别抽样）",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=2,
        help="每个类别最多抽取的意图数量",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只运行前 N 条意图（0 表示不限制）",
    )

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
    all_intents = load_intents(Path(args.intents_file) if args.intents_file != str(INTENT_SOURCE) else None)
    if args.intent_id:
        intent = next((i for i in all_intents if i.get("id") == args.intent_id or i.get("intent_id") == args.intent_id), None)
        if intent:
            intents_to_run.append(intent)
        else:
            print(f"未找到意图: {args.intent_id}")
            sys.exit(1)
    elif args.all:
        intents_to_run = all_intents
    elif args.experiment_suite:
        intents_to_run = select_experiment_intents(all_intents, max_per_category=args.max_per_category)
    else:
        intents_to_run = filter_intents(task_family=args.task_family, dataset=args.dataset)

    if not intents_to_run:
        print("无匹配意图。用 --list 查看。")
        sys.exit(1)

    if args.limit > 0:
        intents_to_run = intents_to_run[: args.limit]

    print(f"\n{'='*60}")
    print(f"OpenClaw Runner: {len(intents_to_run)} 条意图")
    print(f"Model: {args.model}   Dry-run: {dry_run}   Timeout: {args.timeout}s")
    if args.experiment_suite:
        print("实验模式: 先用小型代表性实验集做 dry-run 路径验证，再补真实运行验证")
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
            platform_mvp=args.platform_mvp,
        )
        elapsed = time.time() - start

        ok_statuses = {"completed", "executed", "correctly_rejected", "planned_adapter_identified"}
        status_icon = "✅" if result["status"] in ok_statuses else "❌"
        print(f"  {status_icon} Status: {result['status']} ({elapsed:.1f}s)")
        if result.get("error"):
            print(f"  Error: {result['error'][:200]}")
        if result.get("selected_model"):
            print(f"  Model: {result['selected_model']}")
        if result.get("available_models"):
            print(f"  Available models: {', '.join(result['available_models'][:5])}")
        results.append(result)

    # 汇总
    ok_statuses = {"completed", "executed", "correctly_rejected", "planned_adapter_identified"}
    ok = sum(1 for r in results if r["status"] in ok_statuses)
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"完成: {ok} 成功 / {fail} 失败")
    print(f"Trace 目录: {TRACE_ROOT}")
    print(f"下一步: python benchmark_trace/eval_intents.py --all")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
