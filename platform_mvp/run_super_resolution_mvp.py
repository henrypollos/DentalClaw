#!/usr/bin/env python3
"""Run a lightweight 2D dental image super-resolution MVP.

This adapter is a deterministic baseline for platform validation. It creates a
synthetic low-resolution image by downsampling the original image, restores it
with bicubic interpolation, and reports PSNR/SSIM against the original image.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D/imagesTs"
)
RESAMPLE = getattr(Image, "Resampling", Image)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _case_id(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.png"):
        return name[:-9]
    return path.stem


def _collect_images(input_dir: Path, case_ids: list[str], limit: int) -> list[Path]:
    if case_ids:
        images = []
        for case_id in case_ids:
            candidate = input_dir / f"{case_id}_0000.png"
            if not candidate.exists():
                raise FileNotFoundError(f"Case image not found: {candidate}")
            images.append(candidate)
        return images
    images = sorted(input_dir.glob("*.png"))
    if limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No PNG images found in {input_dir}")
    return images


def _psnr(original: np.ndarray, restored: np.ndarray) -> float:
    mse = float(np.mean((original - restored) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim(original: np.ndarray, restored: np.ndarray) -> float:
    x = original.astype(np.float64)
    y = restored.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _normalize_to_uint8(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image
    return image.convert("L")


def _make_comparison(original: Image.Image, low_preview: Image.Image, restored: Image.Image) -> Image.Image:
    width, height = original.size
    label_h = 28
    canvas = Image.new("L", (width * 3, height + label_h), color=255)
    canvas.paste(original, (0, label_h))
    canvas.paste(low_preview, (width, label_h))
    canvas.paste(restored, (width * 2, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original", fill=0)
    draw.text((width + 8, 8), "Low-res preview", fill=0)
    draw.text((width * 2 + 8, 8), "Bicubic SR", fill=0)
    return canvas


def run_super_resolution(
    *,
    input_dir: Path,
    out_dir: Path,
    case_ids: list[str],
    limit: int,
    scale: int,
) -> dict[str, Any]:
    if scale < 2:
        raise ValueError("--scale must be >= 2")
    images = _collect_images(input_dir, case_ids, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original").mkdir(exist_ok=True)
    (out_dir / "low_resolution").mkdir(exist_ok=True)
    (out_dir / "super_resolution").mkdir(exist_ok=True)
    (out_dir / "comparisons").mkdir(exist_ok=True)

    cases = []
    for image_path in images:
        case_id = _case_id(image_path)
        original = _normalize_to_uint8(Image.open(image_path))
        width, height = original.size
        low_size = (max(1, width // scale), max(1, height // scale))
        low = original.resize(low_size, RESAMPLE.BICUBIC)
        restored = low.resize(original.size, RESAMPLE.BICUBIC)
        low_preview = low.resize(original.size, RESAMPLE.NEAREST)

        original_arr = np.asarray(original, dtype=np.float64)
        restored_arr = np.asarray(restored, dtype=np.float64)
        psnr = _psnr(original_arr, restored_arr)
        ssim = _ssim(original_arr, restored_arr)

        original_out = out_dir / "original" / f"{case_id}.png"
        low_out = out_dir / "low_resolution" / f"{case_id}.png"
        sr_out = out_dir / "super_resolution" / f"{case_id}.png"
        comparison_out = out_dir / "comparisons" / f"{case_id}.png"
        original.save(original_out)
        low.save(low_out)
        restored.save(sr_out)
        _make_comparison(original, low_preview, restored).save(comparison_out)

        cases.append(
            {
                "case_id": case_id,
                "input": _rel(image_path),
                "original": _rel(original_out),
                "low_resolution": _rel(low_out),
                "super_resolution": _rel(sr_out),
                "comparison": _rel(comparison_out),
                "original_size": [width, height],
                "low_resolution_size": list(low_size),
                "scale": scale,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

    mean_psnr = float(np.mean([case["psnr"] for case in cases]))
    mean_ssim = float(np.mean([case["ssim"] for case in cases]))
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "adapter": "dental_2d_super_resolution_bicubic_mvp",
        "method": "synthetic low-resolution degradation + bicubic interpolation",
        "input_dir": _rel(input_dir),
        "out_dir": _rel(out_dir),
        "scale": scale,
        "case_count": len(cases),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "cases": cases,
        "limitations": [
            "This is an executable platform MVP baseline, not a learned super-resolution model.",
            "Low-resolution inputs are synthetically generated from available TDD images.",
            "PSNR/SSIM are valid for this paired synthetic degradation setting, not for unpaired clinical enhancement claims.",
        ],
    }
    _write_json(out_dir / "super_resolution_summary.json", summary)
    (out_dir / "super_resolution_summary.md").write_text(build_report_md(summary), encoding="utf-8")
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report_md(summary: dict[str, Any]) -> str:
    lines = [
        "# DentalClaw 2D 超分 MVP 报告",
        "",
        "## 1. 实验目标",
        "",
        "本 adapter 用一个确定性的 bicubic baseline 接入 DentalClaw 平台，用于证明超分任务已经可以进入统一的平台执行与证据收集流程。该实验不声称达到学习型超分模型效果。",
        "",
        "## 2. 输入与方法",
        "",
        f"- Input directory: `{summary['input_dir']}`",
        f"- Scale factor: `{summary['scale']}`",
        f"- Method: `{summary['method']}`",
        f"- Case count: `{summary['case_count']}`",
        "",
        "## 3. 汇总指标",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| mean_psnr | {_fmt(summary['mean_psnr'])} |",
        f"| mean_ssim | {_fmt(summary['mean_ssim'])} |",
        "",
        "## 4. 病例结果",
        "",
        "| Case | PSNR | SSIM | Comparison |",
        "| --- | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['case_id']}` | {_fmt(case['psnr'])} | {_fmt(case['ssim'])} | `{case['comparison']}` |"
        )
    lines += [
        "",
        "## 5. 当前边界",
        "",
    ]
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines += [
        "",
        "## 6. 汇报口径",
        "",
        "这条路线说明超分任务已经不再只是 registry 占位：平台现在可以执行一个最小超分 adapter，生成图像、指标和报告。下一步若要用于论文主实验，应替换或补充学习型超分 baseline，并在真实低清/高清配对数据上评估。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight 2D super-resolution MVP adapter.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/platform_mvp_runs" / f"super_resolution_demo_{_now_stamp()}",
    )
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_super_resolution(
        input_dir=args.input_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        case_ids=args.case_ids,
        limit=args.limit,
        scale=args.scale,
    )
    print("DentalClaw 2D super-resolution MVP completed.")
    print(f"Run directory: {_rel(args.out_dir)}")
    print(f"Summary JSON: {_rel(args.out_dir / 'super_resolution_summary.json')}")
    print(f"Summary MD: {_rel(args.out_dir / 'super_resolution_summary.md')}")
    print(f"Metrics: PSNR={_fmt(summary['mean_psnr'])}, SSIM={_fmt(summary['mean_ssim'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
