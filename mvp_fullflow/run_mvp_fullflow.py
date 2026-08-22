#!/usr/bin/env python3
"""Run the minimal DentalClaw full-flow MVP experiment.

This script intentionally reuses existing DentalClaw entrypoints instead of
reimplementing inference or report generation:

- agents/experimentation/skills/tooth_autoinfer_nnunet/scripts/run_inference.py
- agents/clinical_result/skills/clinical_report/run_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D"
)
DEFAULT_DELIVERY_STATUS = (
    REPO_ROOT
    / "artifacts/datasets/nnUNet/nnUNet_delivery_Dataset501_TDDTeethBinary2D.status.json"
)
DEFAULT_DELIVERY_REPORT = (
    REPO_ROOT
    / "artifacts/datasets/nnUNet/nnUNet_delivery_Dataset501_TDDTeethBinary2D.md"
)
DEFAULT_QC_JSON = (
    REPO_ROOT
    / "artifacts/results/reports/datasets_qc/Dataset501_TDDTeethBinary2D.qc.json"
)
DEFAULT_QC_MD = (
    REPO_ROOT
    / "artifacts/results/reports/datasets_qc/Dataset501_TDDTeethBinary2D.qc.md"
)
DEFAULT_DATASET_SPEC = REPO_ROOT / "artifacts/results/specs/dataset_spec_501_binary.json"
DEFAULT_TASK_SPEC = REPO_ROOT / "artifacts/results/specs/task_spec_501_binary.json"
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "artifacts/training_runs/trial_501_binary_baseline/best_model/checkpoint_best.pth"
)
DEFAULT_TRAINING_RUN = REPO_ROOT / "artifacts/training_runs/trial_501_binary_baseline"
DEFAULT_INFERENCE_SCRIPT = (
    REPO_ROOT
    / "agents/experimentation/skills/tooth_autoinfer_nnunet/scripts/run_inference.py"
)
DEFAULT_REPORT_SCRIPT = (
    REPO_ROOT / "agents/clinical_result/skills/clinical_report/run_report.py"
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(stdout_path.parent.parent / "mpl_cache"))
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return _rel(dst)


def _metrics_from_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    metric_keys = ["mean_dice", "mean_iou", "mean_hd95", "pixel_accuracy"]
    metrics = {key: payload.get(key) for key in metric_keys if key in payload}
    if not metrics and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
        metrics = {key: summary.get(key) for key in metric_keys if key in summary}
    metrics["case_count"] = len(payload.get("metrics_per_case", {}) or {})
    return metrics


def _resolve_best_nnunet_config(training_run: Path) -> dict[str, str]:
    summary_path = training_run / "report" / "summary.json"
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    best = payload.get("best_experiment") or {}
    config = best.get("config") or {}
    result: dict[str, str] = {}
    if config.get("nnunet_trainer"):
        result["nnunet_trainer"] = str(config["nnunet_trainer"])
    if config.get("fold"):
        result["fold"] = str(config["fold"])
    if config.get("configuration"):
        result["nnunet_configuration"] = str(config["configuration"])
    return result


def _make_runtime_task_spec(
    source_task_spec: Path,
    training_run: Path,
    output_path: Path,
    *,
    trainer: str | None,
    fold: str | None,
    configuration: str | None,
) -> Path:
    payload = _read_json(source_task_spec)
    extra = dict(payload.get("extra") or {})
    best_config = _resolve_best_nnunet_config(training_run)

    resolved_trainer = trainer or best_config.get("nnunet_trainer")
    resolved_fold = fold or best_config.get("fold") or "all"
    resolved_configuration = (
        configuration
        or best_config.get("nnunet_configuration")
        or extra.get("nnunet_configuration")
        or "2d"
    )

    if resolved_trainer:
        extra["nnunet_trainer"] = resolved_trainer
    extra["fold"] = resolved_fold
    extra["nnunet_configuration"] = resolved_configuration
    payload["extra"] = extra

    _write_json(output_path, payload)
    return output_path


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _case_image(dataset_root: Path, case_id: str) -> Path:
    return dataset_root / "imagesTs" / f"{case_id}_0000.png"


def _case_mask(inference_dir: Path, case_id: str) -> Path:
    return inference_dir / f"{case_id}.png"


def _build_summary_md(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    delivery_status: dict[str, Any],
    qc_payload: dict[str, Any],
    inference_metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    qc_source = qc_payload.get("summary") if isinstance(qc_payload.get("summary"), dict) else qc_payload
    qc_summary = {
        "dataset_status": qc_source.get("dataset_status", qc_source.get("status")),
        "case_count": qc_source.get("case_count"),
        "ready_case_count": qc_source.get("ready_case_count"),
        "manual_review_case_count": qc_source.get("manual_review_case_count"),
        "blocked_case_count": qc_source.get("blocked_case_count"),
        "error_count": qc_source.get("error_count"),
        "warning_count": qc_source.get("warning_count"),
    }
    lines = [
        "# DentalClaw 最小全流程 MVP 实验报告",
        "",
        "## 1. 实验目标",
        "",
        "本 MVP 使用当前项目中已经完成的 TDD 二值牙齿分割数据集、QC 产物和训练 checkpoint，调用现有 DentalClaw 推理与报告脚本，串联出一个可复查的最小全流程：",
        "",
        "1. 数据与 QC 产物确认",
        "2. 已训练模型来源确认",
        "3. 测试集推理与指标评估",
        "4. 单病例结构化报告与 overlay 生成",
        "5. 运行证据包汇总",
        "",
        "该实验用于验证现有工程流程可以真实闭环，不用于证明自然语言 intent-to-workflow 泛化能力。",
        "",
        "## 2. 本次运行目录",
        "",
        f"- Run directory: `{_rel(run_dir)}`",
        f"- Manifest: `{manifest['manifest_path']}`",
        "",
        "## 3. 输入资产",
        "",
        f"- Dataset root: `{_rel(args.dataset_root)}`",
        f"- Dataset spec: `{_rel(args.dataset_spec)}`",
        f"- Task spec: `{_rel(args.task_spec)}`",
        f"- Runtime task spec: `{manifest['runtime_task_spec']}`",
        f"- Model checkpoint: `{_rel(args.model_path)}`",
        f"- Training run: `{_rel(args.training_run)}`",
        f"- Delivery status: `{_rel(args.delivery_status)}`",
        f"- QC report: `{_rel(args.qc_json)}`",
        "",
        "## 4. 数据与 QC 摘要",
        "",
        f"- Delivery status: `{delivery_status.get('status', 'unknown')}`",
        f"- Dataset ID: `{delivery_status.get('dataset_id', 'n/a')}`",
        f"- Dataset name: `{delivery_status.get('dataset_name', 'n/a')}`",
    ]
    for key, value in qc_summary.items():
        lines.append(f"- {key}: `{value}`")

    lines += [
        "",
        "## 5. 推理与评估结果",
        "",
        f"- Inference output: `{manifest['inference_dir']}`",
        f"- Inference stdout: `{manifest['inference_stdout']}`",
        f"- Inference stderr: `{manifest['inference_stderr']}`",
        f"- Summary JSON: `{manifest['inference_summary_json']}`",
        f"- Summary MD: `{manifest['inference_summary_md']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| case_count | {_format_metric(inference_metrics.get('case_count'))} |",
        f"| mean_dice | {_format_metric(inference_metrics.get('mean_dice'))} |",
        f"| mean_iou | {_format_metric(inference_metrics.get('mean_iou'))} |",
        f"| mean_hd95 | {_format_metric(inference_metrics.get('mean_hd95'))} |",
        f"| pixel_accuracy | {_format_metric(inference_metrics.get('pixel_accuracy'))} |",
        "",
        "## 6. 报告产物",
        "",
        f"- Case ID: `{args.case_id}`",
        f"- Case image: `{_rel(_case_image(args.dataset_root, args.case_id))}`",
        f"- Case prediction mask: `{_rel(_case_mask(run_dir / 'inference', args.case_id))}`",
        f"- Report directory: `{manifest['report_dir']}`",
        f"- Report markdown: `{manifest['report_md']}`",
        f"- Report HTML: `{manifest['report_html']}`",
        f"- Report overlay: `{manifest['report_overlay']}`",
        f"- Report stdout: `{manifest['report_stdout']}`",
        f"- Report stderr: `{manifest['report_stderr']}`",
        "",
        "## 7. MVP 结论",
        "",
        "本次 MVP 证明：在固定任务设定下，现有 DentalClaw 工程产物可以串联为一个完整可复查流程。该流程覆盖数据/QC、模型、推理评估、病例报告和运行证据汇总。",
        "",
        "当前 MVP 的边界：",
        "",
        "- 未验证真实 LLM 从开放自然语言请求自动生成恰当工作流。",
        "- 未重新训练完整模型；训练阶段复用已有 checkpoint 和训练记录。",
        "- 报告组件用于研究审查和结果展示，不作为临床决策支持验证。",
        "",
        "实现说明：本脚本会在运行目录中生成 `runtime_task_spec.json`，把已有训练 summary 中的最佳 nnUNet trainer 和 fold 写入任务配置副本，以便现有推理入口能够定位已训练结果目录；原始 task spec 不会被修改。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal DentalClaw TDD binary segmentation full-flow MVP."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/mvp_runs" / f"tdd_binary_fullflow_{_now_stamp()}",
        help="Output directory for this MVP run.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--case-id", default="100", help="Case ID for report generation.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--task-spec", type=Path, default=DEFAULT_TASK_SPEC)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--delivery-status", type=Path, default=DEFAULT_DELIVERY_STATUS)
    parser.add_argument("--delivery-report", type=Path, default=DEFAULT_DELIVERY_REPORT)
    parser.add_argument("--qc-json", type=Path, default=DEFAULT_QC_JSON)
    parser.add_argument("--qc-md", type=Path, default=DEFAULT_QC_MD)
    parser.add_argument("--inference-script", type=Path, default=DEFAULT_INFERENCE_SCRIPT)
    parser.add_argument("--report-script", type=Path, default=DEFAULT_REPORT_SCRIPT)
    parser.add_argument(
        "--nnunet-trainer",
        default=None,
        help="Override nnUNet trainer for inference. Default: read best trainer from training run summary.",
    )
    parser.add_argument(
        "--nnunet-fold",
        default=None,
        help="Override nnUNet fold for inference. Default: read best fold, then all.",
    )
    parser.add_argument(
        "--nnunet-configuration",
        default=None,
        help="Override nnUNet configuration for inference. Default: read best config, then 2d.",
    )
    parser.add_argument(
        "--reuse-inference",
        action="store_true",
        help="Reuse existing inference_summary.json and masks in run-dir/inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    inference_dir = run_dir / "inference"
    report_dir = run_dir / f"report_case_{args.case_id}"
    workspace_dir = run_dir / "report_workspace"
    logs_dir = run_dir / "logs"

    _require_dir(args.dataset_root, "Dataset root")
    _require_file(args.dataset_root / "imagesTs" / f"{args.case_id}_0000.png", "Case image")
    _require_file(args.dataset_root / "labelsTs" / f"{args.case_id}.png", "Case label")
    _require_file(args.dataset_spec, "Dataset spec")
    _require_file(args.task_spec, "Task spec")
    _require_file(args.model_path, "Model checkpoint")
    _require_dir(args.training_run, "Training run")
    _require_file(args.delivery_status, "Delivery status")
    _require_file(args.qc_json, "QC JSON report")
    _require_file(args.inference_script, "Inference script")
    _require_file(args.report_script, "Report script")

    run_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(args.delivery_status, run_dir / "evidence" / args.delivery_status.name)
    _copy_if_exists(args.delivery_report, run_dir / "evidence" / args.delivery_report.name)
    _copy_if_exists(args.qc_json, run_dir / "evidence" / args.qc_json.name)
    _copy_if_exists(args.qc_md, run_dir / "evidence" / args.qc_md.name)
    _copy_if_exists(args.training_run / "run_status.json", run_dir / "evidence" / "training_run_status.json")
    _copy_if_exists(args.training_run / "report" / "summary.md", run_dir / "evidence" / "training_summary.md")

    delivery_status = _read_json(args.delivery_status)
    qc_payload = _read_json(args.qc_json)
    runtime_task_spec = _make_runtime_task_spec(
        args.task_spec,
        args.training_run,
        run_dir / "runtime_task_spec.json",
        trainer=args.nnunet_trainer,
        fold=args.nnunet_fold,
        configuration=args.nnunet_configuration,
    )

    inference_stdout = logs_dir / "inference_stdout.log"
    inference_stderr = logs_dir / "inference_stderr.log"
    report_stdout = logs_dir / "report_stdout.log"
    report_stderr = logs_dir / "report_stderr.log"

    if not args.reuse_inference or not (inference_dir / "inference_summary.json").exists():
        inference_cmd = [
            args.python,
            str(args.inference_script),
            "--model-path",
            str(args.model_path),
            "--dataset-spec",
            str(args.dataset_spec),
            "--task-spec",
            str(runtime_task_spec),
            "--output-dir",
            str(inference_dir),
            "--input-dir",
            str(args.dataset_root / "imagesTs"),
            "--gt-dir",
            str(args.dataset_root / "labelsTs"),
        ]
        _run_command(
            inference_cmd,
            cwd=REPO_ROOT,
            stdout_path=inference_stdout,
            stderr_path=inference_stderr,
        )

    mask_path = _case_mask(inference_dir, args.case_id)
    _require_file(mask_path, "Case prediction mask")

    report_cmd = [
        args.python,
        str(args.report_script),
        "--case_id",
        args.case_id,
        "--image_path",
        str(_case_image(args.dataset_root, args.case_id)),
        "--mask_path",
        str(mask_path),
        "--out_dir",
        str(report_dir),
        "--workspace_dir",
        str(workspace_dir),
    ]
    _run_command(
        report_cmd,
        cwd=REPO_ROOT,
        stdout_path=report_stdout,
        stderr_path=report_stderr,
    )

    inference_summary_json = inference_dir / "inference_summary.json"
    inference_summary_md = inference_dir / "inference_summary.md"
    report_md = report_dir / "report.md"
    report_html = report_dir / "report.html"
    report_overlay = report_dir / "overlay.png"

    _require_file(inference_summary_json, "Inference summary JSON")
    _require_file(report_md, "Report markdown")
    _require_file(report_html, "Report HTML")
    _require_file(report_overlay, "Report overlay")

    inference_metrics = _metrics_from_summary(inference_summary_json)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": _rel(run_dir),
        "manifest_path": _rel(run_dir / "manifest.json"),
        "dataset_root": _rel(args.dataset_root),
        "dataset_spec": _rel(args.dataset_spec),
        "task_spec": _rel(args.task_spec),
        "runtime_task_spec": _rel(runtime_task_spec),
        "model_path": _rel(args.model_path),
        "training_run": _rel(args.training_run),
        "delivery_status": _rel(args.delivery_status),
        "qc_json": _rel(args.qc_json),
        "case_id": args.case_id,
        "inference_dir": _rel(inference_dir),
        "inference_summary_json": _rel(inference_summary_json),
        "inference_summary_md": _rel(inference_summary_md),
        "inference_stdout": _rel(inference_stdout),
        "inference_stderr": _rel(inference_stderr),
        "report_dir": _rel(report_dir),
        "report_md": _rel(report_md),
        "report_html": _rel(report_html),
        "report_overlay": _rel(report_overlay),
        "report_stdout": _rel(report_stdout),
        "report_stderr": _rel(report_stderr),
        "metrics": inference_metrics,
    }
    _write_json(run_dir / "manifest.json", manifest)

    summary_md = _build_summary_md(
        run_dir=run_dir,
        args=args,
        delivery_status=delivery_status,
        qc_payload=qc_payload,
        inference_metrics=inference_metrics,
        manifest=manifest,
    )
    (run_dir / "mvp_summary.md").write_text(summary_md, encoding="utf-8")

    print("DentalClaw MVP full-flow run completed.")
    print(f"Run directory: {_rel(run_dir)}")
    print(f"Summary: {_rel(run_dir / 'mvp_summary.md')}")
    print(f"Manifest: {_rel(run_dir / 'manifest.json')}")
    print(
        "Metrics: "
        f"Dice={_format_metric(inference_metrics.get('mean_dice'))}, "
        f"IoU={_format_metric(inference_metrics.get('mean_iou'))}, "
        f"HD95={_format_metric(inference_metrics.get('mean_hd95'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
