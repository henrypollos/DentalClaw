#!/usr/bin/env python3
"""Generate a readiness matrix for all platform MVP routes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = THIS_DIR / "method_registry.json"
DEFAULT_PLATFORM_DEMO = REPO_ROOT / "artifacts/platform_mvp_runs/tdd_platform_demo"
DEFAULT_PRIVATE_VALIDATION = REPO_ROOT / "artifacts/platform_mvp_runs/private01_validation"
DEFAULT_TOOTHFAIRY3_PLAN = REPO_ROOT / "artifacts/platform_mvp_runs/toothfairy3_3d_plan_smoke"
DEFAULT_ANOMALY_PLAN = REPO_ROOT / "artifacts/platform_mvp_runs/anomaly_plan_smoke"
DEFAULT_SUPER_RESOLUTION_RUN = REPO_ROOT / "artifacts/platform_mvp_runs/super_resolution_platform_demo"
DEFAULT_TOOTHFAIRY3_QC = (
    REPO_ROOT / "agents/data_curator/reports/cbct_qc/toothfairy3_lps_tiny/cohort_summary.md"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _method_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in registry.get("methods", [])}


def _base_route(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_id": method.get("id"),
        "display_name": method.get("display_name"),
        "registry_status": method.get("status"),
        "task_family": method.get("task_family"),
        "modality": method.get("modality"),
        "modes": method.get("allowed_modes", []),
        "entrypoint": method.get("entrypoint"),
        "readiness": "unknown",
        "evidence": [],
        "current_decision": None,
        "next_action": None,
    }


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    registry = _read_json(args.registry)
    methods = _method_by_id(registry)
    routes: list[dict[str, Any]] = []

    tdd = _base_route(methods["tdd_2d_segmentation_infer_report"])
    tdd_result = _read_json(args.platform_demo_dir / "execution_result.json")
    if tdd_result.get("status") == "completed":
        tdd["readiness"] = "executable_verified"
        tdd["current_decision"] = "keep_as_primary_demo"
        tdd["metrics"] = tdd_result.get("metrics", {})
        tdd["evidence"] = [
            _rel(args.platform_demo_dir / "platform_summary.md"),
            _rel(args.platform_demo_dir / "execution_result.json"),
            tdd_result.get("delegate_summary"),
            tdd_result.get("report_html"),
            tdd_result.get("report_overlay"),
        ]
        tdd["next_action"] = "Use as the current meeting demo and regression route."
    else:
        tdd["readiness"] = "executable_unverified"
        tdd["current_decision"] = "rerun_demo"
        tdd["next_action"] = "Run platform_mvp with --execute and collect execution_result.json."
    routes.append(tdd)

    private2d = _base_route(methods["private_2d_segmentation_train"])
    private_report = _read_json(args.private_validation_dir / "private2d_validation_report.json")
    private2d["evidence"] = [
        _rel(args.private_validation_dir / "private2d_validation_report.json"),
        _rel(args.private_validation_dir / "private2d_validation_report.md"),
    ]
    private2d["private_validation"] = {
        "image_count": private_report.get("image_count"),
        "mask_count": private_report.get("mask_count"),
        "paired_case_count": private_report.get("paired_case_count"),
        "terminal_policy": private_report.get("terminal_policy"),
    }
    if private_report.get("can_execute_requested_mode"):
        private2d["readiness"] = "ready_for_adapter_execution"
        private2d["current_decision"] = "can_build_training_adapter"
        private2d["next_action"] = "Implement private2d export/spec generation and call nnUNet training entrypoint."
    else:
        private2d["readiness"] = "blocked_by_missing_labels"
        private2d["current_decision"] = private_report.get("terminal_policy", "stop_without_training")
        private2d["next_action"] = "Provide paired masks/labels or annotation export before supervised private training."
    routes.append(private2d)

    toothfairy = _base_route(methods["toothfairy3_3d_segmentation_infer_or_train"])
    tf_plan = _read_json(args.toothfairy3_plan_dir / "platform_plan.json")
    toothfairy["evidence"] = [
        _rel(args.toothfairy3_plan_dir / "platform_plan.json"),
        _rel(args.toothfairy3_plan_dir / "platform_summary.md"),
        _rel(args.toothfairy3_qc_md) if args.toothfairy3_qc_md.exists() else None,
    ]
    toothfairy["readiness"] = "planned_with_qc_basis"
    toothfairy["current_decision"] = "not_executable_yet"
    toothfairy["plan_supported"] = tf_plan.get("supported")
    toothfairy["plan_executable"] = tf_plan.get("executable")
    toothfairy["next_action"] = "Promote CBCT QC or 3D inference into a maintained executable adapter."
    routes.append(toothfairy)

    anomaly = _base_route(methods["dental_anomaly_detection"])
    anomaly_plan = _read_json(args.anomaly_plan_dir / "platform_plan.json")
    anomaly["evidence"] = [
        _rel(args.anomaly_plan_dir / "platform_plan.json"),
        _rel(args.anomaly_plan_dir / "platform_summary.md"),
    ]
    anomaly["readiness"] = "planned_missing_entrypoint"
    anomaly["current_decision"] = "reject_execution_until_adapter_bound"
    anomaly["plan_supported"] = anomaly_plan.get("supported")
    anomaly["plan_executable"] = anomaly_plan.get("executable")
    anomaly["next_action"] = "Define anomaly label contract, select baseline, then bind a stable entrypoint."
    routes.append(anomaly)

    super_resolution = _base_route(methods["dental_2d_super_resolution"])
    sr_plan = _read_json(args.super_resolution_run_dir / "platform_plan.json")
    sr_result = _read_json(args.super_resolution_run_dir / "execution_result.json")
    super_resolution["evidence"] = [
        _rel(args.super_resolution_run_dir / "platform_plan.json"),
        _rel(args.super_resolution_run_dir / "platform_summary.md"),
        _rel(args.super_resolution_run_dir / "execution_result.json"),
        _rel(args.super_resolution_run_dir / "super_resolution/super_resolution_summary.md"),
        _rel(args.super_resolution_run_dir / "super_resolution/comparisons/100.png"),
    ]
    super_resolution["plan_supported"] = sr_plan.get("supported")
    super_resolution["plan_executable"] = sr_plan.get("executable")
    if sr_result.get("status") == "completed":
        super_resolution["readiness"] = "executable_verified"
        super_resolution["current_decision"] = "keep_as_super_resolution_demo"
        super_resolution["metrics"] = sr_result.get("metrics", {})
        super_resolution["next_action"] = "Replace or complement the bicubic MVP with a learned baseline for paper experiments."
    else:
        super_resolution["readiness"] = "executable_unverified"
        super_resolution["current_decision"] = "rerun_super_resolution_demo"
        super_resolution["next_action"] = "Run the super-resolution route with --execute and collect execution_result.json."
    routes.append(super_resolution)

    summary = {
        "total_routes": len(routes),
        "executable_verified": sum(1 for route in routes if route["readiness"] == "executable_verified"),
        "blocked_by_missing_labels": sum(1 for route in routes if route["readiness"] == "blocked_by_missing_labels"),
        "planned_with_qc_basis": sum(1 for route in routes if route["readiness"] == "planned_with_qc_basis"),
        "planned_missing_entrypoint": sum(1 for route in routes if route["readiness"] == "planned_missing_entrypoint"),
    }
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "registry": _rel(args.registry),
        "summary": summary,
        "routes": routes,
    }


def build_matrix_md(matrix: dict[str, Any]) -> str:
    summary = matrix["summary"]
    lines = [
        "# DentalClaw 平台路线 Readiness Matrix",
        "",
        "## 1. 总览",
        "",
        f"- Total routes: `{summary['total_routes']}`",
        f"- Executable and verified: `{summary['executable_verified']}`",
        f"- Blocked by missing labels: `{summary['blocked_by_missing_labels']}`",
        f"- Planned with QC basis: `{summary['planned_with_qc_basis']}`",
        f"- Planned missing entrypoint: `{summary['planned_missing_entrypoint']}`",
        "",
        "## 2. 路线状态表",
        "",
        "| Route | Registry Status | Readiness | Decision | Next Action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for route in matrix["routes"]:
        lines.append(
            "| "
            f"`{route['route_id']}` | "
            f"`{route['registry_status']}` | "
            f"`{route['readiness']}` | "
            f"`{route.get('current_decision')}` | "
            f"{route.get('next_action')} |"
        )

    lines += [
        "",
        "## 3. 已验证执行路线",
        "",
    ]
    verified_routes = [route for route in matrix["routes"] if route["readiness"] == "executable_verified"]
    for route in verified_routes:
        metrics = route.get("metrics") or {}
        lines += [
            f"### {route['route_id']}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        for key, value in metrics.items():
            lines.append(f"| {key} | {_format_metric(value)} |")
        lines += [
            "",
            "Evidence:",
            "",
        ]
        for evidence in route.get("evidence", []):
            if evidence:
                lines.append(f"- `{evidence}`")
        lines.append("")

    private2d = next(route for route in matrix["routes"] if route["route_id"] == "private_2d_segmentation_train")
    validation = private2d.get("private_validation") or {}
    lines += [
        "",
        "## 4. 私有训练路线阻塞原因",
        "",
        f"- Image count: `{validation.get('image_count')}`",
        f"- Mask count: `{validation.get('mask_count')}`",
        f"- Paired case count: `{validation.get('paired_case_count')}`",
        f"- Terminal policy: `{validation.get('terminal_policy')}`",
        "",
        "结论：私有训练属于老师要求的核心方向，但当前 `data/private01` 缺少 mask/label，因此平台必须停止监督训练，等待配对标注或 annotation export。",
        "",
        "## 5. 汇报口径",
        "",
        "这张表用于回答“多任务现在做到哪一步”：当前不是声称所有任务都完成，而是把每条路线的可执行性、证据和下一步门槛说清楚。已完成的是 TDD 2D 分割推理报告闭环和一个超分 MVP baseline；私有训练被前置检查正确拦截；3D 分割已有 QC 基础；异常检测等待 baseline/entrypoint 绑定。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the DentalClaw platform route readiness matrix.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/platform_mvp_runs/readiness_matrix_latest",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--platform-demo-dir", type=Path, default=DEFAULT_PLATFORM_DEMO)
    parser.add_argument("--private-validation-dir", type=Path, default=DEFAULT_PRIVATE_VALIDATION)
    parser.add_argument("--toothfairy3-plan-dir", type=Path, default=DEFAULT_TOOTHFAIRY3_PLAN)
    parser.add_argument("--toothfairy3-qc-md", type=Path, default=DEFAULT_TOOTHFAIRY3_QC)
    parser.add_argument("--anomaly-plan-dir", type=Path, default=DEFAULT_ANOMALY_PLAN)
    parser.add_argument("--super-resolution-run-dir", type=Path, default=DEFAULT_SUPER_RESOLUTION_RUN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = build_matrix(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "readiness_matrix.json", matrix)
    (args.out_dir / "readiness_matrix.md").write_text(
        build_matrix_md(matrix),
        encoding="utf-8",
    )
    print("Platform readiness matrix generated.")
    print(f"Matrix JSON: {_rel(args.out_dir / 'readiness_matrix.json')}")
    print(f"Matrix MD: {_rel(args.out_dir / 'readiness_matrix.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
