#!/usr/bin/env python3
"""Generate a single advisor-facing report pack for the platform MVP."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = THIS_DIR / "method_registry.json"
DEFAULT_PLATFORM_DEMO = REPO_ROOT / "artifacts/platform_mvp_runs/tdd_platform_demo"
DEFAULT_FULLFLOW_RUN = REPO_ROOT / "artifacts/mvp_runs/tdd_binary_fullflow_20260709_085641"
DEFAULT_PRIVATE_VALIDATION = REPO_ROOT / "artifacts/platform_mvp_runs/private01_validation"
DEFAULT_READINESS_MATRIX = REPO_ROOT / "artifacts/platform_mvp_runs/readiness_matrix_latest"
DEFAULT_SUPER_RESOLUTION_RUN = REPO_ROOT / "artifacts/platform_mvp_runs/super_resolution_platform_demo"
DEFAULT_TOOTHFAIRY3_PLAN = REPO_ROOT / "artifacts/platform_mvp_runs/toothfairy3_3d_plan_smoke"
DEFAULT_TOOTHFAIRY3_QC = (
    REPO_ROOT / "agents/data_curator/reports/cbct_qc/toothfairy3_lps_tiny/cohort_summary.md"
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return _rel(dst)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _method_table(registry: dict[str, Any]) -> list[str]:
    lines = [
        "| Route | Status | Mode | Task | Modality | Entrypoint |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in registry.get("methods", []):
        mode = ", ".join(method.get("allowed_modes", []))
        lines.append(
            "| "
            f"`{method.get('id')}` | "
            f"`{method.get('status')}` | "
            f"{mode} | "
            f"{method.get('task_family')} | "
            f"{method.get('modality')} | "
            f"`{method.get('entrypoint')}` |"
        )
    return lines


def build_report(
    *,
    registry: dict[str, Any],
    platform_result: dict[str, Any],
    private_validation: dict[str, Any],
    toothfairy3_plan: dict[str, Any],
    readiness_matrix: dict[str, Any],
    super_resolution_result: dict[str, Any],
    copied: dict[str, str | None],
    args: argparse.Namespace,
) -> str:
    metrics = platform_result.get("metrics") or {}
    readiness_summary = readiness_matrix.get("summary") or {}
    lines = [
        "# DentalClaw 平台底座 MVP 汇报材料",
        "",
        "## 1. 本次汇报结论",
        "",
        "本次工作把方向从 benchmark 主线调整为工具平台 MVP：平台先收窄到牙科 CV，使用离线方法表做可控方法选择，并用已有 DentalClaw 入口打通真实执行路线。当前已经能从一句话请求选择 TDD 2D 分割推理报告路线，并接通 2D 超分 MVP baseline，产出可复查的指标、报告、overlay 和超分对比图。",
        "",
        "## 2. 老师要求与当前落实",
        "",
        "| 老师要求 | 当前落实证据 |",
        "| --- | --- |",
        "| 工具平台，不是 benchmark | `platform_mvp/run_platform_mvp.py` 负责平台编排；benchmark 仅作为后续测试工具 |",
        "| CV 范围收窄 | `method_registry.json` 的 domain 固定为 `dental_cv` |",
        "| 支持推理和私有数据训练 | registry 的 mode 限定为 `inference` / `private_train` |",
        "| 一句话自动选方法 | `--intent` 输入后解析 dataset/task/modality/mode 并查离线方法表 |",
        "| 先跑通核心闭环 | TDD 2D 分割推理报告路线已执行完成 |",
        "| 多任务实证边界 | 2D 分割、私有训练、3D 分割、异常检测、超分已登记路线和状态 |",
        "",
        "## 3. 当前平台路线表",
        "",
    ]
    lines.extend(_method_table(registry))
    lines += [
        "",
        "## 4. 已跑通路线：TDD 2D 分割推理报告",
        "",
        f"- Platform run: `{_rel(args.platform_demo_dir)}`",
        f"- Full-flow run: `{_rel(args.fullflow_run_dir)}`",
        f"- Execution status: `{platform_result.get('status', 'unknown')}`",
        f"- Report HTML: `{platform_result.get('report_html', 'n/a')}`",
        f"- Overlay: `{platform_result.get('report_overlay', 'n/a')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| case_count | {_format_metric(metrics.get('case_count'))} |",
        f"| mean_dice | {_format_metric(metrics.get('mean_dice'))} |",
        f"| mean_iou | {_format_metric(metrics.get('mean_iou'))} |",
        f"| mean_hd95 | {_format_metric(metrics.get('mean_hd95'))} |",
        f"| pixel_accuracy | {_format_metric(metrics.get('pixel_accuracy'))} |",
        "",
        "## 5. 私有数据训练预检查",
        "",
        f"- Data root: `{private_validation.get('data_root', 'n/a')}`",
        f"- Image count: `{private_validation.get('image_count', 'n/a')}`",
        f"- Mask count: `{private_validation.get('mask_count', 'n/a')}`",
        f"- Paired case count: `{private_validation.get('paired_case_count', 'n/a')}`",
        f"- Decision: `{private_validation.get('terminal_policy', 'n/a')}`",
        f"- Reason: {private_validation.get('reason', 'n/a')}",
        "",
        "这个结果用于汇报时说明：平台会支持私有数据训练，但不会在缺少标注时假装可以监督训练。`data/private01` 当前有 139 个 DICOM 图像，但没有检测到 mask/label，因此只能进入登记/QC或后续推理计划，不能直接进入监督训练。",
        "",
        "## 6. 3D 分割路线当前基础",
        "",
        f"- ToothFairy3 platform plan: `{_rel(args.toothfairy3_plan_dir)}`",
        f"- Parsed task: `{(toothfairy3_plan.get('intent') or {}).get('task_family', 'n/a')}`",
        f"- Parsed modality: `{(toothfairy3_plan.get('intent') or {}).get('modality', 'n/a')}`",
        f"- Selected method: `{(toothfairy3_plan.get('selected_method') or {}).get('id', 'n/a')}`",
        f"- Executable now: `{toothfairy3_plan.get('executable', 'n/a')}`",
        f"- Existing CBCT QC summary copy: `{copied.get('toothfairy3_qc_summary')}`",
        "",
        "这里的汇报口径是：3D 分割已经进入平台能力表，并且仓库已有 ToothFairy3 CBCT QC 证据；但 3D 分割训练/推理 adapter 还没有标为 executable，下一步应先把 QC 或 inference demo 接成第二条可执行路线。",
        "",
        "## 7. 已接通路线：2D 超分 MVP",
        "",
        f"- Platform run: `{_rel(args.super_resolution_run_dir)}`",
        f"- Execution status: `{super_resolution_result.get('status', 'unknown')}`",
        f"- Summary: `{super_resolution_result.get('delegate_summary', 'n/a')}`",
        f"- Comparison image copy: `{copied.get('super_resolution_comparison')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| case_count | {_format_metric((super_resolution_result.get('metrics') or {}).get('case_count'))} |",
        f"| mean_psnr | {_format_metric((super_resolution_result.get('metrics') or {}).get('mean_psnr'))} |",
        f"| mean_ssim | {_format_metric((super_resolution_result.get('metrics') or {}).get('mean_ssim'))} |",
        "",
        "这里的汇报口径是：超分任务已经不再只是 registry 占位，当前有一个可执行的 bicubic MVP baseline，可生成超分图、对比图、PSNR/SSIM 和报告。论文实验阶段可替换为学习型 baseline。",
        "",
        "## 8. 多任务 Readiness Matrix",
        "",
        f"- Matrix summary: `{copied.get('readiness_matrix_md')}`",
        f"- Total routes: `{readiness_summary.get('total_routes', 'n/a')}`",
        f"- Executable and verified: `{readiness_summary.get('executable_verified', 'n/a')}`",
        f"- Blocked by missing labels: `{readiness_summary.get('blocked_by_missing_labels', 'n/a')}`",
        f"- Planned with QC basis: `{readiness_summary.get('planned_with_qc_basis', 'n/a')}`",
        f"- Planned missing entrypoint: `{readiness_summary.get('planned_missing_entrypoint', 'n/a')}`",
        "",
        "这张表用于回答“多任务现在做到哪一步”：已执行、被输入条件阻塞、有 QC 基础、缺 baseline/entrypoint 的路线会分开列出，避免把 planned adapter 误说成已完成。",
        "",
        "## 9. 汇报文件索引",
        "",
        f"- Platform summary copy: `{copied.get('platform_summary')}`",
        f"- Platform execution JSON copy: `{copied.get('execution_result')}`",
        f"- Full-flow summary copy: `{copied.get('fullflow_summary')}`",
        f"- Private validation report copy: `{copied.get('private_validation_md')}`",
        f"- ToothFairy3 plan copy: `{copied.get('toothfairy3_plan_summary')}`",
        f"- ToothFairy3 QC copy: `{copied.get('toothfairy3_qc_summary')}`",
        f"- Super-resolution summary copy: `{copied.get('super_resolution_summary')}`",
        f"- Super-resolution comparison copy: `{copied.get('super_resolution_comparison')}`",
        f"- Readiness matrix copy: `{copied.get('readiness_matrix_md')}`",
        f"- Manifest: `{copied.get('manifest')}`",
        "",
        "## 10. 下次推进",
        "",
        "1. 将 `private_2d_segmentation_train` 从 planned adapter 推进到可执行 adapter：补齐私有 images/masks 输入包、QC、nnUNet export 和训练调用。",
        "2. ToothFairy3 先接入 3D QC 或 inference demo，形成第二类 CV 任务证据。",
        "3. 超分路线从 bicubic MVP 升级为学习型 baseline，并在真实配对数据上评估。",
        "4. 异常检测先明确标签要求和无标签策略：无标签时只允许推理/热图，不允许训练指标。",
        "",
        "## 11. 建议会议表述",
        "",
        "> 我已经把工作从 benchmark 调整成平台底座 MVP。现在平台能接收一句话，查离线方法表，选择可执行路线，并调用已有 DentalClaw 代码产出真实报告。TDD 2D 分割推理报告路线已经跑通；超分路线也已经有一个可执行 MVP baseline，能生成图像、PSNR/SSIM 和对比报告；私有数据训练已经补了输入契约和预检查，当前 `data/private01` 因缺少 mask 被正确拦截。3D 分割已有 QC 基础，异常检测还需要绑定 baseline/entrypoint。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a report pack for the DentalClaw platform MVP.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/platform_mvp_runs" / f"advisor_report_pack_{_now_stamp()}",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--platform-demo-dir", type=Path, default=DEFAULT_PLATFORM_DEMO)
    parser.add_argument("--fullflow-run-dir", type=Path, default=DEFAULT_FULLFLOW_RUN)
    parser.add_argument("--private-validation-dir", type=Path, default=DEFAULT_PRIVATE_VALIDATION)
    parser.add_argument("--readiness-matrix-dir", type=Path, default=DEFAULT_READINESS_MATRIX)
    parser.add_argument("--super-resolution-run-dir", type=Path, default=DEFAULT_SUPER_RESOLUTION_RUN)
    parser.add_argument("--toothfairy3-plan-dir", type=Path, default=DEFAULT_TOOTHFAIRY3_PLAN)
    parser.add_argument("--toothfairy3-qc-md", type=Path, default=DEFAULT_TOOTHFAIRY3_QC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = _read_json(args.registry)
    platform_result = _read_json(args.platform_demo_dir / "execution_result.json")
    private_validation = _read_json(args.private_validation_dir / "private2d_validation_report.json")
    readiness_matrix = _read_json(args.readiness_matrix_dir / "readiness_matrix.json")
    super_resolution_result = _read_json(args.super_resolution_run_dir / "execution_result.json")
    toothfairy3_plan = _read_json(args.toothfairy3_plan_dir / "platform_plan.json")

    copied = {
        "platform_summary": _copy_if_exists(
            args.platform_demo_dir / "platform_summary.md",
            args.out_dir / "evidence/platform_summary.md",
        ),
        "execution_result": _copy_if_exists(
            args.platform_demo_dir / "execution_result.json",
            args.out_dir / "evidence/execution_result.json",
        ),
        "fullflow_summary": _copy_if_exists(
            args.fullflow_run_dir / "mvp_summary.md",
            args.out_dir / "evidence/fullflow_mvp_summary.md",
        ),
        "private_validation_md": _copy_if_exists(
            args.private_validation_dir / "private2d_validation_report.md",
            args.out_dir / "evidence/private2d_validation_report.md",
        ),
        "toothfairy3_plan_summary": _copy_if_exists(
            args.toothfairy3_plan_dir / "platform_summary.md",
            args.out_dir / "evidence/toothfairy3_3d_plan_summary.md",
        ),
        "toothfairy3_qc_summary": _copy_if_exists(
            args.toothfairy3_qc_md,
            args.out_dir / "evidence/toothfairy3_cbct_qc_summary.md",
        ),
        "super_resolution_summary": _copy_if_exists(
            args.super_resolution_run_dir / "super_resolution/super_resolution_summary.md",
            args.out_dir / "evidence/super_resolution_summary.md",
        ),
        "super_resolution_comparison": _copy_if_exists(
            args.super_resolution_run_dir / "super_resolution/comparisons/100.png",
            args.out_dir / "evidence/super_resolution_comparison_100.png",
        ),
        "readiness_matrix_md": _copy_if_exists(
            args.readiness_matrix_dir / "readiness_matrix.md",
            args.out_dir / "evidence/readiness_matrix.md",
        ),
        "readiness_matrix_json": _copy_if_exists(
            args.readiness_matrix_dir / "readiness_matrix.json",
            args.out_dir / "evidence/readiness_matrix.json",
        ),
        "manifest": _rel(args.out_dir / "report_pack_manifest.json"),
    }

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "out_dir": _rel(args.out_dir),
        "registry": _rel(args.registry),
        "platform_demo_dir": _rel(args.platform_demo_dir),
        "fullflow_run_dir": _rel(args.fullflow_run_dir),
        "private_validation_dir": _rel(args.private_validation_dir),
        "readiness_matrix_dir": _rel(args.readiness_matrix_dir),
        "super_resolution_run_dir": _rel(args.super_resolution_run_dir),
        "toothfairy3_plan_dir": _rel(args.toothfairy3_plan_dir),
        "toothfairy3_qc_md": _rel(args.toothfairy3_qc_md),
        "copied": copied,
        "report_md": _rel(args.out_dir / "老师汇报材料.md"),
    }
    _write_json(args.out_dir / "report_pack_manifest.json", manifest)
    (args.out_dir / "老师汇报材料.md").write_text(
        build_report(
            registry=registry,
            platform_result=platform_result,
            private_validation=private_validation,
            toothfairy3_plan=toothfairy3_plan,
            readiness_matrix=readiness_matrix,
            super_resolution_result=super_resolution_result,
            copied=copied,
            args=args,
        ),
        encoding="utf-8",
    )

    print("Advisor report pack generated.")
    print(f"Report: {_rel(args.out_dir / '老师汇报材料.md')}")
    print(f"Manifest: {_rel(args.out_dir / 'report_pack_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
