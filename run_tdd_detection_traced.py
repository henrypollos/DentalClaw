from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_trace.trace_recorder import (
    end_run,
    record_orchestrator_action,
    start_run,
    traced_tool_call_sync,
)
from skills.tdd_mask_to_bbox_skill import (
    convert_tdd_masks_to_yolo,
)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def audit_yolo_dataset(
    dataset_root: str,
) -> Dict[str, Any]:
    root = Path(dataset_root)

    result: Dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "splits": {},
        "ready_for_training": True,
        "issues": [],
    }

    for split_name in ["train", "val", "test"]:
        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name

        image_files = (
            [
                path
                for path in image_dir.iterdir()
                if path.is_file() or path.is_symlink()
            ]
            if image_dir.exists()
            else []
        )

        label_files = (
            list(label_dir.glob("*.txt"))
            if label_dir.exists()
            else []
        )

        image_stems = {
            path.stem for path in image_files
        }
        label_stems = {
            path.stem for path in label_files
        }

        missing_labels = sorted(
            image_stems - label_stems
        )

        empty_labels = []

        for path in label_files:
            if not path.read_text(
                encoding="utf-8"
            ).strip():
                empty_labels.append(path.stem)

        result["splits"][split_name] = {
            "image_count": len(image_files),
            "label_count": len(label_files),
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
        }

        if split_name in {"train", "val"}:
            if not image_files:
                result["ready_for_training"] = False
                result["issues"].append(
                    {
                        "split": split_name,
                        "issue_type": "no_images",
                    }
                )

            if missing_labels:
                result["ready_for_training"] = False
                result["issues"].append(
                    {
                        "split": split_name,
                        "issue_type": (
                            "missing_detection_labels"
                        ),
                        "count": len(missing_labels),
                        "preview": missing_labels[:20],
                    }
                )

    data_yaml = root / "data.yaml"

    if not data_yaml.exists():
        result["ready_for_training"] = False
        result["issues"].append(
            {
                "issue_type": "missing_data_yaml",
            }
        )

    result["data_yaml"] = str(data_yaml)

    return result


def train_detector(
    data_yaml: str,
    model_path: str,
    workspace: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    workers: int,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    workspace_path = Path(workspace)
    train_project = workspace_path / "training"

    model = YOLO(model_path)

    train_result = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(train_project),
        name="tdd_tooth_detection",
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(
        str(
            getattr(
                train_result,
                "save_dir",
                train_project
                / "tdd_tooth_detection",
            )
        )
    )

    best_model_path = (
        save_dir / "weights" / "best.pt"
    )
    last_model_path = (
        save_dir / "weights" / "last.pt"
    )

    if not best_model_path.exists():
        candidates = list(
            train_project.glob(
                "**/weights/best.pt"
            )
        )

        if candidates:
            best_model_path = candidates[0]

    if not best_model_path.exists():
        raise RuntimeError(
            "Training completed but best.pt "
            "was not found under {}".format(
                train_project
            )
        )

    return {
        "status": "completed",
        "model_source": model_path,
        "save_dir": str(save_dir),
        "best_model_path": str(
            best_model_path
        ),
        "last_model_path": str(
            last_model_path
        ),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": device,
    }


def validate_detector(
    best_model_path: str,
    data_yaml: str,
    workspace: str,
    device: str,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(best_model_path)

    metrics = model.val(
        data=data_yaml,
        split="val",
        device=device,
        project=str(
            Path(workspace) / "validation"
        ),
        name="tdd_tooth_detection_val",
        exist_ok=True,
        verbose=True,
    )

    box_metrics = getattr(metrics, "box", None)

    return {
        "status": "completed",
        "mAP50": _to_float(
            getattr(box_metrics, "map50", None)
        ),
        "mAP50_95": _to_float(
            getattr(box_metrics, "map", None)
        ),
        "precision": _to_float(
            getattr(box_metrics, "mp", None)
        ),
        "recall": _to_float(
            getattr(box_metrics, "mr", None)
        ),
        "validation_save_dir": str(
            getattr(metrics, "save_dir", "")
        ),
    }


def run_detection_inference(
    best_model_path: str,
    source_dir: str,
    workspace: str,
    device: str,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(best_model_path)

    results = model.predict(
        source=source_dir,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        device=device,
        project=str(
            Path(workspace) / "predictions"
        ),
        name="val_predictions",
        exist_ok=True,
        verbose=False,
    )

    save_dir = ""

    if results:
        save_dir = str(
            getattr(results[0], "save_dir", "")
        )

    return {
        "status": "completed",
        "prediction_count": len(results),
        "prediction_save_dir": save_dir,
    }


def generate_detection_report(
    workspace: str,
    conversion: Dict[str, Any],
    audit: Dict[str, Any],
    training: Dict[str, Any],
    metrics: Dict[str, Any],
    inference: Dict[str, Any],
) -> Dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "task": "TDD 2D tooth detection",
        "annotation_conversion": (
            "segmentation_mask_to_bounding_box"
        ),
        "conversion": conversion,
        "dataset_audit": audit,
        "training": training,
        "metrics": metrics,
        "inference": inference,
        "generated_at": (
            datetime.now().astimezone().isoformat()
        ),
    }

    report_path = (
        workspace_path
        / "detection_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "report_path": str(report_path),
        "metrics": metrics,
        "best_model_path": training[
            "best_model_path"
        ],
        "prediction_save_dir": inference[
            "prediction_save_dir"
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a traced TDD 2D tooth "
            "detection workflow."
        )
    )

    parser.add_argument(
        "--source-root",
        default=str(
            REPO_ROOT / "data" / "pano2d"
        ),
    )

    parser.add_argument(
        "--detection-root",
        default=str(
            REPO_ROOT
            / "data"
            / "tdd_detection_yolo"
        ),
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Ultralytics checkpoint name or "
            "local .pt path."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--link-mode",
        choices=[
            "symlink",
            "hardlink",
            "copy",
        ],
        default="symlink",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    intent_id = "DET_TDD_STANDARD_001"

    prompt = (
        "确认将TDD牙齿分割掩码转换为边界框，"
        "训练二维牙齿检测模型，并输出mAP、"
        "Precision和Recall。"
    )

    workspace = (
        REPO_ROOT
        / "workspace"
        / "detection_trace"
        / stamp
    )

    run_id = start_run(
        intent_id=intent_id,
        prompt=prompt,
        dataset="TDD",
        task_type="2d_tooth_detection",
        session_id="offline_{}".format(stamp),
    )

    print("[TRACE] run_id:", run_id)
    print("[TRACE] workspace:", workspace)

    try:
        record_orchestrator_action(
            agent="CoordinatorAgent",
            action="workflow_start",
            reason=(
                "用户已确认将TDD分割掩码转换为"
                "边界框并执行二维牙齿检测。"
            ),
            target={
                "dataset": "TDD",
                "task_type": "2d_detection",
            },
        )

        conversion = traced_tool_call_sync(
            agent="DataCurationAgent",
            tool_name="mask_to_bbox_conversion",
            arguments={
                "source_root": args.source_root,
                "output_root": args.detection_root,
                "class_count": 32,
                "min_area": args.min_area,
                "link_mode": args.link_mode,
            },
            invoke=lambda: (
                convert_tdd_masks_to_yolo(
                    source_root=args.source_root,
                    output_root=args.detection_root,
                    class_count=32,
                    min_area=args.min_area,
                    link_mode=args.link_mode,
                )
            ),
            decision_summary=(
                "检测任务需要边界框，因此将"
                "牙齿级分割掩码转换为YOLO边界框。"
            ),
            next_action="dataset_audit",
        )

        audit = traced_tool_call_sync(
            agent="DataCurationAgent",
            tool_name="detection_dataset_audit",
            arguments={
                "dataset_root": args.detection_root,
            },
            invoke=lambda: audit_yolo_dataset(
                args.detection_root
            ),
            decision_summary=(
                "检查转换后的检测图像、标签、"
                "空标注和缺失标注。"
            ),
            next_action="detection_training",
        )

        if not audit["ready_for_training"]:
            record_orchestrator_action(
                agent="DataCurationAgent",
                action="warn_and_stop",
                reason=(
                    "转换后的检测数据未通过审计，"
                    "停止训练。"
                ),
                target={
                    "issues": audit["issues"],
                },
            )

            raise RuntimeError(
                "Detection dataset audit failed: "
                + json.dumps(
                    audit["issues"],
                    ensure_ascii=False,
                )
            )

        training = traced_tool_call_sync(
            agent="ExperimentationAgent",
            tool_name="detection_model_training",
            arguments={
                "data_yaml": audit["data_yaml"],
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "workspace": str(workspace),
            },
            invoke=lambda: train_detector(
                data_yaml=audit["data_yaml"],
                model_path=args.model,
                workspace=str(workspace),
                epochs=args.epochs,
                image_size=args.imgsz,
                batch_size=args.batch,
                device=args.device,
                workers=args.workers,
            ),
            decision_summary=(
                "使用YOLO执行二维牙齿目标检测训练。"
            ),
            next_action="detection_validation",
        )

        metrics = traced_tool_call_sync(
            agent="ExperimentationAgent",
            tool_name="detection_model_validation",
            arguments={
                "best_model_path": training[
                    "best_model_path"
                ],
                "data_yaml": audit["data_yaml"],
                "device": args.device,
            },
            invoke=lambda: validate_detector(
                best_model_path=training[
                    "best_model_path"
                ],
                data_yaml=audit["data_yaml"],
                workspace=str(workspace),
                device=args.device,
            ),
            decision_summary=(
                "在验证集计算mAP、Precision和Recall。"
            ),
            next_action="prediction_export",
        )

        inference = traced_tool_call_sync(
            agent="ExperimentationAgent",
            tool_name="detection_prediction_export",
            arguments={
                "best_model_path": training[
                    "best_model_path"
                ],
                "source_dir": str(
                    Path(args.detection_root)
                    / "images"
                    / "val"
                ),
                "device": args.device,
            },
            invoke=lambda: run_detection_inference(
                best_model_path=training[
                    "best_model_path"
                ],
                source_dir=str(
                    Path(args.detection_root)
                    / "images"
                    / "val"
                ),
                workspace=str(workspace),
                device=args.device,
            ),
            decision_summary=(
                "导出验证集预测框、置信度和可视化。"
            ),
            next_action="report_generation",
        )

        report = traced_tool_call_sync(
            agent="ReportingAgent",
            tool_name="detection_report_generation",
            arguments={
                "workspace": str(workspace),
                "metrics": metrics,
                "best_model_path": training[
                    "best_model_path"
                ],
            },
            invoke=lambda: generate_detection_report(
                workspace=str(workspace),
                conversion=conversion,
                audit=audit,
                training=training,
                metrics=metrics,
                inference=inference,
            ),
            decision_summary=(
                "聚合数据转换、审计、训练、"
                "检测指标和预测结果。"
            ),
            next_action="workflow_complete",
        )

        final_response = (
            "TDD二维牙齿检测工作流完成。"
            "mAP@0.5={}; mAP@0.5:0.95={}; "
            "Precision={}; Recall={}。".format(
                metrics["mAP50"],
                metrics["mAP50_95"],
                metrics["precision"],
                metrics["recall"],
            )
        )

        end_run(
            status="completed",
            final_response=final_response,
            workflow_config={
                "run_id": run_id,
                "workspace": str(workspace),
                "report": report,
            },
        )

        print(final_response)

        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:
        end_run(
            status="failed",
            final_response=(
                "TDD二维牙齿检测工作流失败。"
            ),
            workflow_config={
                "workspace": str(workspace),
                "detection_root": (
                    args.detection_root
                ),
            },
            error=str(exc),
        )

        print("[ERROR]", exc)
        raise


if __name__ == "__main__":
    main()
