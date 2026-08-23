from __future__ import annotations

import hashlib
import inspect
import json
import os
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Awaitable, Callable, Optional


TRACE_ROOT = Path(
    os.getenv(
        "DENTALCLAW_TRACE_DIR",
        "$DENTALCLAW_HOME/benchmark_runs",
    )
)

_TRACE_CONTEXT: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "dentalclaw_trace_context",
    default=None,
)

_WRITE_LOCK = Lock()



def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_value(value: Any, max_text_length: int = 3000) -> Any:
    """
    将工具参数和返回值转换为可写入JSON的形式。
    大数组、张量和复杂对象只记录摘要，避免日志文件过大。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > max_text_length:
            return value[:max_text_length] + "...<truncated>"
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): _safe_value(v, max_text_length)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) > 100:
            return {
                "type": type(value).__name__,
                "length": len(values),
                "preview": [
                    _safe_value(v, max_text_length)
                    for v in values[:20]
                ],
            }
        return [_safe_value(v, max_text_length) for v in values]

    # numpy数组、torch张量等
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {
            "type": type(value).__name__,
            "shape": list(shape),
            "dtype": str(dtype),
        }

    text = repr(value)
    if len(text) > max_text_length:
        text = text[:max_text_length] + "...<truncated>"

    return {
        "type": type(value).__name__,
        "repr": text,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(
        record,
        ensure_ascii=False,
        default=str,
    )

    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def _get_context() -> dict[str, Any]:
    context = _TRACE_CONTEXT.get()
    if context is None:
        raise RuntimeError(
            "Trace context has not been initialized. "
            "Call start_run() before invoking tools."
        )
    return context


def emit_event(event: str, **payload: Any) -> None:
    context = _get_context()
    context["sequence"] += 1

    record = {
        "event": event,
        "run_id": context["run_id"],
        "intent_id": context["intent_id"],
        "sequence": context["sequence"],
        "timestamp": _now_iso(),
        **_safe_value(payload),
    }

    _append_jsonl(context["trace_file"], record)


def start_run(
    intent_id: str,
    prompt: str,
    dataset: Optional[str] = None,
    task_type: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    run_id = f"{intent_id}_{timestamp}_{suffix}"

    run_dir = TRACE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tool_returns").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    context = {
        "run_id": run_id,
        "intent_id": intent_id,
        "run_dir": run_dir,
        "trace_file": run_dir / "tool_trace.jsonl",
        "sequence": 0,
    }
    _TRACE_CONTEXT.set(context)

    manifest = {
        "run_id": run_id,
        "intent_id": intent_id,
        "session_id": session_id,
        "prompt": prompt,
        "dataset": dataset,
        "task_type": task_type,
        "started_at": _now_iso(),
        "status": "running",
    }

    with (run_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    emit_event(
        "run_start",
        prompt=prompt,
        dataset=dataset,
        task_type=task_type,
        session_id=session_id,
    )

    return run_id


def save_full_return(call_id: str, result: Any) -> tuple[str, str]:
    """
    单独保存工具完整返回值，并计算哈希。
    日志主文件只保存摘要和文件路径。
    """
    context = _get_context()
    output_path = (
        context["run_dir"]
        / "tool_returns"
        / f"{call_id}.json"
    )

    serializable = _safe_value(result, max_text_length=10000)

    content = json.dumps(
        serializable,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    output_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return str(output_path), digest


async def traced_tool_call(
    *,
    agent: str,
    tool_name: str,
    arguments: dict[str, Any],
    invoke: Callable[[], Any],
    decision_summary: Optional[str] = None,
    next_action: Optional[str] = None,
) -> Any:
    """
    所有工具统一通过此函数调用。
    支持同步函数和异步函数。
    """
    context = _get_context()
    call_id = f"call_{context['sequence'] + 1:03d}_{uuid.uuid4().hex[:6]}"
    start_time = perf_counter()
    timestamp_start = _now_iso()

    emit_event(
        "tool_call_start",
        call_id=call_id,
        agent=agent,
        tool_name=tool_name,
        arguments=arguments,
        decision_summary=decision_summary,
        next_action=next_action,
    )

    try:
        result = invoke()

        if inspect.isawaitable(result):
            result = await result

        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        result_path, result_sha256 = save_full_return(call_id, result)

        emit_event(
            "tool_call_end",
            call_id=call_id,
            timestamp_start=timestamp_start,
            timestamp_end=_now_iso(),
            duration_ms=duration_ms,
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            return_value=result,
            return_file=result_path,
            return_sha256=result_sha256,
            status="success",
            decision_summary=decision_summary,
            next_action=next_action,
            error=None,
        )

        return result

    except Exception as exc:
        duration_ms = round((perf_counter() - start_time) * 1000, 2)

        emit_event(
            "tool_call_error",
            call_id=call_id,
            timestamp_start=timestamp_start,
            timestamp_end=_now_iso(),
            duration_ms=duration_ms,
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            return_value=None,
            status="error",
            decision_summary=decision_summary,
            next_action=next_action,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

        raise


def record_orchestrator_action(
    *,
    agent: str,
    action: str,
    reason: str,
    target: Optional[dict[str, Any]] = None,
) -> None:
    """
    用于记录非工具行为，例如追问、拒绝、跳过和人工确认。
    """
    emit_event(
        "orchestrator_action",
        agent=agent,
        action=action,
        decision_summary=reason,
        target=target or {},
    )


def end_run(
    *,
    status: str,
    final_response: Optional[str] = None,
    workflow_config: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    context = _get_context()
    run_dir: Path = context["run_dir"]

    if final_response is not None:
        (run_dir / "final_response.txt").write_text(
            final_response,
            encoding="utf-8",
        )

    if workflow_config is not None:
        with (run_dir / "workflow_config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                _safe_value(workflow_config),
                file,
                ensure_ascii=False,
                indent=2,
            )

    emit_event(
        "run_end",
        status=status,
        final_response=final_response,
        workflow_config=workflow_config,
        error=error,
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "ended_at": _now_iso(),
            "status": status,
            "error": error,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def traced_tool_call_sync(
    *,
    agent: str,
    tool_name: str,
    arguments: dict[str, Any],
    invoke: Callable[[], Any],
    decision_summary: Optional[str] = None,
    next_action: Optional[str] = None,
) -> Any:
    """
    同步工具调用追踪器。

    记录：
    1. 调用了什么工具；
    2. 传入了什么参数；
    3. 返回了什么；
    4. 是否成功；
    5. 执行时间；
    6. 异常内容。
    """
    context = _get_context()

    call_id = (
        f"call_{context['sequence'] + 1:03d}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    start_timer = perf_counter()
    timestamp_start = _now_iso()

    emit_event(
        "tool_call_start",
        call_id=call_id,
        agent=agent,
        tool_name=tool_name,
        arguments=arguments,
        decision_summary=decision_summary,
        next_action=next_action,
    )

    try:
        result = invoke()

        if inspect.isawaitable(result):
            raise TypeError(
                f"{tool_name} returned an awaitable in a synchronous "
                "trace call. Use traced_tool_call instead."
            )

        duration_ms = round(
            (perf_counter() - start_timer) * 1000,
            2,
        )

        result_path, result_sha256 = save_full_return(
            call_id,
            result,
        )

        emit_event(
            "tool_call_end",
            call_id=call_id,
            timestamp_start=timestamp_start,
            timestamp_end=_now_iso(),
            duration_ms=duration_ms,
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            return_value=result,
            return_file=result_path,
            return_sha256=result_sha256,
            status="success",
            decision_summary=decision_summary,
            next_action=next_action,
            error=None,
        )

        return result

    except Exception as exc:
        duration_ms = round(
            (perf_counter() - start_timer) * 1000,
            2,
        )

        emit_event(
            "tool_call_error",
            call_id=call_id,
            timestamp_start=timestamp_start,
            timestamp_end=_now_iso(),
            duration_ms=duration_ms,
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            return_value=None,
            status="error",
            decision_summary=decision_summary,
            next_action=next_action,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

        raise
