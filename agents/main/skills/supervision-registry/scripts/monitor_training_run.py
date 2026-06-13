#!/usr/bin/env python3
"""Summarize the live state of a DentalClaw training workspace."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def tail_lines(path: Path, limit: int = 20) -> List[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def latest_training_log(workspace: Path) -> Optional[Path]:
    candidates = sorted(workspace.rglob("training_log_*.txt"))
    return candidates[-1] if candidates else None


def summarize_workspace(workspace: Path) -> Dict[str, Any]:
    run_status = maybe_read_json(workspace / "run_status.json") or {}
    launcher_status = maybe_read_json(workspace / "launcher_status.json") or {}
    history = read_json(workspace / "history.json") if (workspace / "history.json").is_file() else []
    search_events = read_events(workspace / "search_events.jsonl")
    search_strategy = maybe_read_json(workspace / "search_strategy.json") or {}
    run_summary = maybe_read_json(workspace / "run_summary.json") or {}
    train_log_path = latest_training_log(workspace)
    latest_log_lines = tail_lines(train_log_path, limit=12) if train_log_path else []

    completed_trials = [item for item in history if item.get("status") == "completed"]
    failed_trials = [item for item in history if item.get("status") == "failed"]
    latest_event = search_events[-1] if search_events else None
    current_experiment = run_status.get("current_experiment") or {}
    best_record = None
    if completed_trials:
        best_record = max(
            completed_trials,
            key=lambda item: (
                float(((item.get("metrics") or {}).get("mean_dice") or 0.0)),
                float(((item.get("metrics") or {}).get("mean_iou") or 0.0)),
            ),
        )

    status = {
        "generated_at": utc_now_iso(),
        "workspace": str(workspace),
        "launcher_status": launcher_status.get("status"),
        "controller_pid": launcher_status.get("pid"),
        "run_status": run_status.get("status"),
        "stage": run_status.get("stage"),
        "task_id": run_status.get("task_id"),
        "dataset_root": run_status.get("dataset_root"),
        "max_trials": run_status.get("max_trials"),
        "completed_trials": run_status.get("completed_trials"),
        "history_count": len(history),
        "completed_trial_count": len(completed_trials),
        "failed_trial_count": len(failed_trials),
        "current_experiment": current_experiment,
        "latest_event": latest_event,
        "planned_trials_preview": (search_strategy.get("planned_experiments") or [])[:3],
        "best_so_far": {
            "exp_id": best_record.get("exp_id"),
            "trial_name": (best_record.get("config") or {}).get("trial_name"),
            "mean_dice": (best_record.get("metrics") or {}).get("mean_dice"),
            "mean_iou": (best_record.get("metrics") or {}).get("mean_iou"),
            "selection_reason": (best_record.get("config") or {}).get("selection_reason"),
        } if best_record else None,
        "run_summary_path": str(workspace / "run_summary.json") if (workspace / "run_summary.json").is_file() else None,
        "main_handoff_path": str(workspace / "main_handoff.md") if (workspace / "main_handoff.md").is_file() else None,
        "training_log_path": str(train_log_path) if train_log_path else None,
        "latest_training_log_lines": latest_log_lines,
    }

    if status["run_status"] == "completed":
        status["supervisor_recommendation"] = "Training workflow completed. Main should read main_handoff.md, report best results, and if needed start a new run with revised specs."
    elif status["run_status"] == "failed":
        status["supervisor_recommendation"] = "Training workflow failed. Main should inspect the latest event, controller logs, and current experiment artifacts before retrying."
    elif status["completed_trial_count"] < int(status.get("max_trials") or 0):
        status["supervisor_recommendation"] = "Workflow is still active or incomplete. Main should keep monitoring launcher_status.json, run_status.json, search_events.jsonl, and the current training log."
    else:
        status["supervisor_recommendation"] = "Workflow state is ambiguous. Main should inspect controller logs and search artifacts."
    return status


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a DentalClaw training workspace for supervision.")
    parser.add_argument("--workspace", required=True, help="Training workspace to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    payload = summarize_workspace(workspace)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
