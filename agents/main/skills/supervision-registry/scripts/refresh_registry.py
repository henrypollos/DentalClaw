#!/usr/bin/env python3
"""Refresh filesystem-derived registries for datasets, models, and task runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[5]
REGISTRY_ROOT = REPO_ROOT / "registry"
DATASET_RAW_ROOT = REPO_ROOT / "artifacts" / "datasets" / "nnUNet" / "nnUNet_raw"
DATASET_REPORT_ROOT = REPO_ROOT / "artifacts" / "datasets"
QC_REPORT_ROOT = REPO_ROOT / "artifacts" / "results" / "reports" / "datasets_qc"
MODEL_ROOT = REPO_ROOT / "artifacts" / "models"
TRAINING_RESULTS_ROOT = REPO_ROOT / "artifacts" / "results" / "training"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file())


def parse_dataset_id(name: str) -> Optional[int]:
    match = re.match(r"Dataset(\d{3})_", name)
    return int(match.group(1)) if match else None


def scan_dataset_qc_reports() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not QC_REPORT_ROOT.is_dir():
        return items
    for report_path in sorted(QC_REPORT_ROOT.glob("*.qc.json")):
        payload = read_json(report_path)
        summary = payload.get("summary", {}) or {}
        items.append({
            "report_key": payload.get("report_key") or report_path.stem.replace(".qc", ""),
            "dataset_name": payload.get("dataset_name"),
            "dataset_root": payload.get("dataset_root"),
            "dataset_mode": payload.get("dataset_mode"),
            "status": summary.get("dataset_status"),
            "case_count": summary.get("case_count"),
            "ready_case_count": summary.get("ready_case_count"),
            "manual_review_case_count": summary.get("manual_review_case_count"),
            "blocked_case_count": summary.get("blocked_case_count"),
            "error_count": summary.get("error_count"),
            "warning_count": summary.get("warning_count"),
            "path": str(report_path),
            "path_relative": rel(report_path),
            "markdown_path": payload.get("report_md"),
            "generated_at": payload.get("generated_at"),
        })
    return items


def scan_datasets(qc_reports: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    qc_by_key = {item["report_key"]: item for item in (qc_reports or []) if item.get("report_key")}
    if not DATASET_RAW_ROOT.is_dir():
        return items
    for dataset_dir in sorted(path for path in DATASET_RAW_ROOT.iterdir() if path.is_dir()):
        if not re.match(r"Dataset\d{3}_", dataset_dir.name):
            continue
        dataset_json_path = dataset_dir / "dataset.json"
        dataset_json = read_json(dataset_json_path) if dataset_json_path.is_file() else {}
        folder_name = dataset_dir.name
        delivery_json = DATASET_REPORT_ROOT / f"nnUNet_delivery_{folder_name}.json"
        delivery_status = DATASET_REPORT_ROOT / f"nnUNet_delivery_{folder_name}.status.json"
        delivery_payload = read_json(delivery_json) if delivery_json.is_file() else {}
        delivery_status_payload = read_json(delivery_status) if delivery_status.is_file() else {}
        qc_payload = qc_by_key.get(folder_name)
        labels = dataset_json.get("labels", {})
        label_names = [name for name in labels.keys() if name != "background"]
        items.append({
            "dataset_id": parse_dataset_id(folder_name),
            "folder_name": folder_name,
            "name": dataset_json.get("name"),
            "path": str(dataset_dir),
            "path_relative": rel(dataset_dir),
            "file_ending": dataset_json.get("file_ending"),
            "description": dataset_json.get("description"),
            "label_count_excluding_background": max(len(labels) - 1, 0),
            "label_names_preview": label_names[:16],
            "counts": {
                "imagesTr": count_files(dataset_dir / "imagesTr"),
                "labelsTr": count_files(dataset_dir / "labelsTr"),
                "imagesTs": count_files(dataset_dir / "imagesTs"),
                "labelsTs": count_files(dataset_dir / "labelsTs"),
                "imagesVal": count_files(dataset_dir / "imagesVal"),
                "labelsVal": count_files(dataset_dir / "labelsVal"),
            },
            "delivery_report": str(delivery_json) if delivery_json.is_file() else None,
            "delivery_status": delivery_status_payload.get("status") if delivery_status_payload else None,
            "qc_status": (qc_payload or {}).get("status"),
            "qc_report": (qc_payload or {}).get("path"),
            "qc_report_relative": (qc_payload or {}).get("path_relative"),
            "task": delivery_payload.get("task"),
            "generated_at": delivery_payload.get("generated_at"),
            "last_modified": file_mtime_iso(dataset_dir),
            "ready": dataset_json_path.is_file() and (dataset_dir / "imagesTr").is_dir() and (dataset_dir / "labelsTr").is_dir(),
        })
    return items


def scan_artifact_models() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not MODEL_ROOT.exists():
        return items
    for model_path in sorted(MODEL_ROOT.rglob("*")):
        if not model_path.is_file():
            continue
        if model_path.name not in {"checkpoint_best.pth", "checkpoint_final.pth", "model.pth"}:
            continue
        dataset_match = next((part for part in model_path.parts if re.match(r"Dataset\d{3}_", part)), None)
        items.append({
            "source": "artifacts",
            "model_name": model_path.stem,
            "path": str(model_path),
            "path_relative": rel(model_path),
            "dataset_folder": dataset_match,
            "dataset_id": parse_dataset_id(dataset_match) if dataset_match else None,
            "status": "completed",
            "last_modified": file_mtime_iso(model_path),
        })
    return items


def scan_workspace_models() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for root in run_workspace_roots():
        if not root.exists():
            continue
        for summary_path in sorted(root.rglob("run_summary.json")):
            payload = read_json(summary_path)
            best_model = payload.get("best_model", {}) or {}
            model_path = payload.get("best_model_export_path") or best_model.get("best_model_path")
            if not model_path:
                continue
            inference = payload.get("inference") or {}
            main_handoff = payload.get("main_handoff") or {}
            artifacts = best_model.get("artifacts") or {}
            items.append({
                "source": "workspace",
                "model_name": best_model.get("model_name"),
                "path": str(model_path),
                "path_relative": rel(Path(model_path)),
                "workspace": str(summary_path.parent),
                "workspace_relative": rel(summary_path.parent),
                "task_id": best_model.get("task_id"),
                "status": best_model.get("status", "completed"),
                "mean_dice": (best_model.get("metrics") or {}).get("mean_dice"),
                "mean_iou": (best_model.get("metrics") or {}).get("mean_iou"),
                "test_mean_dice": inference.get("mean_dice"),
                "test_mean_hd95": inference.get("mean_hd95"),
                "test_mean_iou": inference.get("mean_iou"),
                "test_pixel_accuracy": inference.get("pixel_accuracy"),
                "training_curve_path": artifacts.get("training_curve_path"),
                "validation_summary_path": artifacts.get("validation_summary_path"),
                "main_handoff_path": main_handoff.get("workspace_md") or main_handoff.get("artifact_md"),
                "search_strategy_path": (payload.get("search_strategy") or {}).get("md"),
                "last_modified": file_mtime_iso(summary_path),
            })
    return items


def run_workspace_roots() -> List[Path]:
    roots = [TRAINING_RESULTS_ROOT]
    deduped: List[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def _scan_launcher_statuses(root: Path) -> Dict[Path, Dict[str, Any]]:
    statuses: Dict[Path, Dict[str, Any]] = {}
    if not root.exists():
        return statuses
    for status_path in sorted(root.rglob("launcher_status.json")):
        try:
            statuses[status_path.parent.resolve()] = read_json(status_path)
        except Exception:
            continue
    return statuses


def _scan_run_statuses(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("run_status.json"))


def scan_task_runs() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    launcher_by_workspace: Dict[Path, Dict[str, Any]] = {}
    for root in run_workspace_roots():
        launcher_by_workspace.update(_scan_launcher_statuses(root))
    for root in run_workspace_roots():
        for status_path in _scan_run_statuses(root):
            payload = read_json(status_path)
            launcher_payload = launcher_by_workspace.get(status_path.parent.resolve()) or {}
            items.append({
                "kind": "experimentation_run",
                "run_id": status_path.parent.name,
                "status": payload.get("status"),
                "stage": payload.get("stage"),
                "task_id": payload.get("task_id"),
                "dataset_root": payload.get("dataset_root"),
                "completed_trials": payload.get("completed_trials"),
                "max_trials": payload.get("max_trials"),
                "path": str(status_path),
                "path_relative": rel(status_path),
                "workspace": str(status_path.parent),
                "workspace_relative": rel(status_path.parent),
                "updated_at": payload.get("updated_at"),
                "current_experiment": payload.get("current_experiment"),
                "best_model_path": payload.get("best_model_path"),
                "run_summary_path": payload.get("run_summary_path"),
                "main_handoff_path": payload.get("main_handoff_path"),
                "error": payload.get("error"),
                "launcher_status": launcher_payload.get("status"),
                "launcher_pid": launcher_payload.get("pid"),
                "launcher_status_path": str(status_path.parent / "launcher_status.json") if (status_path.parent / "launcher_status.json").exists() else None,
                "controller_stdout_log": launcher_payload.get("stdout_log"),
                "controller_stderr_log": launcher_payload.get("stderr_log"),
            })
        for summary_path in sorted(root.rglob("run_summary.json")):
            payload = read_json(summary_path)
            items.append({
                "kind": "experimentation_summary",
                "run_id": summary_path.parent.name,
                "status": "completed",
                "stage": "completed",
                "task_id": ((payload.get("best_model") or {}).get("task_id")),
                "dataset_root": ((payload.get("dataset_info") or {}).get("root")),
                "path": str(summary_path),
                "path_relative": rel(summary_path),
                "workspace": str(summary_path.parent),
                "workspace_relative": rel(summary_path.parent),
                "updated_at": file_mtime_iso(summary_path),
                "best_model_path": payload.get("best_model_export_path"),
            })
    for status_path in sorted(DATASET_REPORT_ROOT.glob("nnUNet_delivery_*.status.json")):
        payload = read_json(status_path)
        items.append({
            "kind": "dataset_export",
            "run_id": status_path.stem.replace(".status", ""),
            "status": payload.get("status"),
            "stage": payload.get("stage"),
            "task": payload.get("task"),
            "dataset_id": payload.get("dataset_id"),
            "dataset_name": payload.get("dataset_name"),
            "dataset_root": payload.get("dataset_root"),
            "path": str(status_path),
            "path_relative": rel(status_path),
            "updated_at": payload.get("updated_at"),
            "delivery_report": payload.get("delivery_report"),
        })
    return items


def build_lineage(datasets: Iterable[Dict[str, Any]], models: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    dataset_by_id = {item["dataset_id"]: item for item in datasets if item.get("dataset_id") is not None}
    links = []
    for model in models:
        dataset_id = model.get("dataset_id")
        if dataset_id is None or dataset_id not in dataset_by_id:
            continue
        links.append({
            "dataset_id": dataset_id,
            "dataset_folder": dataset_by_id[dataset_id]["folder_name"],
            "model_path": model["path"],
            "model_source": model["source"],
        })
    return {
        "generated_at": utc_now_iso(),
        "links": links,
    }


def build_overview(
    datasets: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    task_runs: List[Dict[str, Any]],
    qc_reports: List[Dict[str, Any]],
) -> str:
    running = [item for item in task_runs if item.get("status") == "running"]
    failed = [item for item in task_runs if item.get("status") == "failed"]
    completed_models = [item for item in models if item.get("status") == "completed"]
    qc_fail = [item for item in qc_reports if item.get("status") == "fail"]
    lines = [
        "# DentalClaw Registry Overview",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Available datasets: {len(datasets)}",
        f"- Known models: {len(models)}",
        f"- QC reports: {len(qc_reports)}",
        f"- QC failures: {len(qc_fail)}",
        f"- Running tasks: {len(running)}",
        f"- Failed tasks: {len(failed)}",
        "",
        "## Datasets",
        "",
    ]
    if datasets:
        for item in datasets:
            lines.append(
                f"- `{item['folder_name']}` | id={item.get('dataset_id')} | labels={item['label_count_excluding_background']} | "
                f"train={item['counts']['imagesTr']} | test={item['counts']['imagesTs']} | ready={item['ready']} | qc={item.get('qc_status')}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Dataset QC", ""])
    if qc_reports:
        for item in qc_reports[:20]:
            lines.append(
                f"- `{item.get('report_key')}` | status={item.get('status')} | "
                f"errors={item.get('error_count')} | warnings={item.get('warning_count')} | path=`{item.get('path_relative')}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Models", ""])
    if completed_models:
        for item in completed_models[:20]:
            lines.append(
                f"- `{item.get('model_name') or Path(item['path']).name}` | source={item['source']} | "
                f"path=`{item['path_relative']}` | test_dice={item.get('test_mean_dice')} | test_iou={item.get('test_mean_iou')} | curve=`{rel(Path(item['training_curve_path'])) if item.get('training_curve_path') else 'n/a'}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Running Tasks", ""])
    if running:
        for item in running[:20]:
            lines.append(
                f"- `{item['kind']}` | run=`{item['run_id']}` | stage=`{item.get('stage')}` | updated=`{item.get('updated_at')}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Failed Tasks", ""])
    if failed:
        for item in failed[:20]:
            lines.append(
                f"- `{item['kind']}` | run=`{item['run_id']}` | error=`{item.get('error')}`"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> None:
    qc_reports = scan_dataset_qc_reports()
    datasets = scan_datasets(qc_reports)
    models = scan_artifact_models() + scan_workspace_models()
    task_runs = scan_task_runs()
    lineage = build_lineage(datasets, models)

    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REGISTRY_ROOT / "datasets.json", {
        "generated_at": utc_now_iso(),
        "count": len(datasets),
        "items": datasets,
    })
    write_json(REGISTRY_ROOT / "dataset_qc_reports.json", {
        "generated_at": utc_now_iso(),
        "count": len(qc_reports),
        "items": qc_reports,
    })
    write_json(REGISTRY_ROOT / "models.json", {
        "generated_at": utc_now_iso(),
        "count": len(models),
        "items": models,
    })
    write_json(REGISTRY_ROOT / "task_runs.json", {
        "generated_at": utc_now_iso(),
        "count": len(task_runs),
        "items": task_runs,
    })
    write_json(REGISTRY_ROOT / "lineage.json", lineage)
    (REGISTRY_ROOT / "overview.md").write_text(build_overview(datasets, models, task_runs, qc_reports), encoding="utf-8")

    print(json.dumps({
        "registry_root": str(REGISTRY_ROOT),
        "datasets": len(datasets),
        "dataset_qc_reports": len(qc_reports),
        "models": len(models),
        "task_runs": len(task_runs),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
