#!/usr/bin/env python3
"""DentalClaw multi-agent trace protocol.

Defines the standard trace event format for all agents in the platform.
Frontend panels consume these traces to render per-agent workflow views.

Agent topology (4 logical agents):
  planner       — intent parsing, web search, method selection, decision
  data_curator  — dataset validation, QC, preprocessing
  experimenter  — training, hyperparameter tuning, model selection
  clinician     — inference, TTA/ensemble, clinical report generation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Agent identity ─────────────────────────────────────────────────────

KNOWN_AGENTS = {
    "planner": {
        "id": "planner",
        "name": "Planner",
        "description": "Intent parsing, web search, method registry lookup, agent decision",
        "workspace": "agents/main",
        "steps": [
            "intent.parse",
            "web.search",
            "registry.method_lookup",
            "agent.decide",
            "platform.request_clarification",
            "agent.external_proposal",
            "platform.reject_or_explain",
        ],
    },
    "data_curator": {
        "id": "data_curator",
        "name": "Data Curator",
        "description": "Dataset validation, QC auditing, preprocessing, nnUNet export",
        "workspace": "agents/data_curator",
        "steps": [
            "dataset.qc_read",
            "dataset.validate_package",
            "dataset.export_to_nnunet",
            "dataset.cbct_qc",
            "dataset.prepare_3d_specs",
        ],
    },
    "experimenter": {
        "id": "experimenter",
        "name": "Experimenter",
        "description": "Training, hyperparameter search, model checkpoint selection",
        "workspace": "agents/experimentation",
        "steps": [
            "model.checkpoint_select",
            "experiment.training",
            "experiment.best_model_select",
            "experiment.inference",
            "experiment.tta_ensemble_inference",
        ],
    },
    "clinician": {
        "id": "clinician",
        "name": "Clinician",
        "description": "Inference, TTA/ensemble, clinical report, overlay generation",
        "workspace": "agents/clinical_result",
        "steps": [
            "clinical_report.generate",
            "platform.collect_evidence",
        ],
    },
}


# ── Trace event ────────────────────────────────────────────────────────

@dataclass
class AgentTraceEvent:
    """A single step executed by an agent."""

    agent_id: str                  # e.g. "planner"
    step_name: str                 # e.g. "intent.parse"
    status: str                    # pending | running | completed | failed | skipped

    started_at: str = ""           # ISO 8601
    completed_at: str = ""         # ISO 8601
    duration_ms: float = 0.0

    input_summary: str = ""        # human-readable one-liner
    output_summary: str = ""       # human-readable one-liner
    decision: str = ""             # agent's rationale / decision note
    detail: dict[str, Any] = field(default_factory=dict)  # arbitrary JSON

    @classmethod
    def pending(cls, agent_id: str, step_name: str, input_summary: str = "") -> "AgentTraceEvent":
        return cls(
            agent_id=agent_id,
            step_name=step_name,
            status="pending",
            input_summary=input_summary,
        )

    def start(self) -> "AgentTraceEvent":
        self.status = "running"
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return self

    def complete(self, output_summary: str = "", decision: str = "", detail: dict | None = None) -> "AgentTraceEvent":
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        if self.started_at:
            try:
                start_dt = datetime.fromisoformat(self.started_at)
                end_dt = datetime.fromisoformat(self.completed_at)
                self.duration_ms = (end_dt - start_dt).total_seconds() * 1000
            except Exception:
                pass
        self.output_summary = output_summary
        self.decision = decision
        if detail:
            self.detail = detail
        return self

    def fail(self, error: str) -> "AgentTraceEvent":
        self.status = "failed"
        self.completed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.output_summary = error
        return self

    def skip(self, reason: str = "") -> "AgentTraceEvent":
        self.status = "skipped"
        self.output_summary = reason
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Trace recorder ─────────────────────────────────────────────────────

class TraceRecorder:
    """Records a sequence of agent trace events for one platform run."""

    def __init__(self, run_id: str, output_dir: Path):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[AgentTraceEvent] = []
        self._started_at = datetime.now(timezone.utc)

    def add(self, event: AgentTraceEvent) -> AgentTraceEvent:
        self.events.append(event)
        return event

    def step(self, agent_id: str, step_name: str,
             input_summary: str = "", output_summary: str = "",
             decision: str = "", status: str = "completed",
             detail: dict | None = None) -> AgentTraceEvent:
        """Shortcut: add a completed step in one call."""
        ev = AgentTraceEvent.pending(agent_id, step_name, input_summary)
        ev.start()
        if status == "completed":
            ev.complete(output_summary, decision, detail)
        elif status == "failed":
            ev.fail(output_summary)
        elif status == "skipped":
            ev.skip(output_summary)
        return self.add(ev)

    def flush(self) -> Path:
        """Write the trace to disk and return the path."""
        total_ms = sum(e.duration_ms for e in self.events)
        agent_summary = {}
        for e in self.events:
            ag = agent_summary.setdefault(e.agent_id, {
                "agent_name": KNOWN_AGENTS.get(e.agent_id, {}).get("name", e.agent_id),
                "total_steps": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "total_duration_ms": 0.0,
            })
            ag["total_steps"] += 1
            ag[e.status] = ag.get(e.status, 0) + 1
            ag["total_duration_ms"] += e.duration_ms

        trace_doc = {
            "run_id": self.run_id,
            "started_at": self._started_at.isoformat(timespec="milliseconds"),
            "flushed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "total_duration_ms": total_ms,
            "agents": KNOWN_AGENTS,
            "agent_summary": agent_summary,
            "events": [e.to_dict() for e in self.events],
        }

        trace_path = self.output_dir / "agent_trace.json"
        trace_path.write_text(json.dumps(trace_doc, ensure_ascii=False, indent=2))
        return trace_path

    def events_by_agent(self, agent_id: str) -> list[AgentTraceEvent]:
        return [e for e in self.events if e.agent_id == agent_id]


# ── Convenience: build a recorder from plan ────────────────────────────

def create_trace_recorder(run_dir: Path, case_id: str = "100") -> TraceRecorder:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"dentalclaw_{stamp}_{case_id}"
    return TraceRecorder(run_id=run_id, output_dir=run_dir)
