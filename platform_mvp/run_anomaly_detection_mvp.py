#!/usr/bin/env python3
"""Dental 2D anomaly detection MVP baseline.

Uses a pretrained ResNet18 backbone to extract image features, then applies
IsolationForest for unsupervised anomaly scoring.  This is a real baseline
(same principle as the super‑resolution bicubic adapter): it produces
meaningful output without requiring anomaly labels or training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D/imagesTs"
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_images(image_dir: Path, max_images: int = 200) -> list[tuple[str, np.ndarray]]:
    """Load PNG images, return (case_id, RGB array)."""
    pairs = []
    for p in sorted(image_dir.glob("*.png")):
        if len(pairs) >= max_images:
            break
        try:
            img = Image.open(p).convert("RGB")
            pairs.append((p.stem.split("_")[0], np.array(img)))
        except Exception:
            continue
    return pairs


def extract_features(images: list[tuple[str, np.ndarray]], device: str) -> np.ndarray:
    """Extract 512‑d features from a pretrained ResNet18 (no classifier)."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # strip classifier
    model = model.to(device).eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    batch_size = 16
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        tensors = torch.stack([transform(Image.fromarray(arr)) for _, arr in batch]).to(device)
        with torch.no_grad():
            feats = model(tensors).cpu().numpy()
        features.append(feats)
    return np.concatenate(features, axis=0)


def run_anomaly_detection(
    image_dir: Path,
    out_dir: Path,
    contamination: float = 0.1,
    max_images: int = 200,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load images
    images = load_images(image_dir, max_images=max_images)
    if not images:
        return {"status": "blocked_no_images", "reason": f"No PNG images found in {image_dir}"}

    # 2. Extract features
    features = extract_features(images, device)

    # 3. IsolationForest anomaly scoring
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(features)   # -1 = anomaly, 1 = normal
    scores = clf.score_samples(features)  # lower = more anomalous

    # 4. Per-case results
    case_results = []
    for idx, (case_id, _) in enumerate(images):
        case_results.append({
            "case_id": case_id,
            "anomaly_label": int(preds[idx]),
            "anomaly_score": round(float(scores[idx]), 6),
            "is_anomaly": bool(preds[idx] == -1),
        })

    anomaly_count = sum(1 for c in case_results if c["is_anomaly"])
    normal_count = len(case_results) - anomaly_count

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "ResNet18-ImageNet-features + IsolationForest",
        "device": device,
        "input_dir": _rel(image_dir),
        "total_cases": len(case_results),
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "contamination": contamination,
        "case_results": case_results,
        "limitations": [
            "Unsupervised baseline; anomaly labels are inferred from feature distribution, not ground truth.",
            "Pretrained on ImageNet, not dental-specific.",
            "Scores are relative to the input cohort; no absolute threshold.",
        ],
    }

    _write_json(out_dir / "anomaly_detection_summary.json", summary)

    # 5. Markdown summary
    lines = [
        "# Dental 2D Anomaly Detection MVP Baseline",
        "",
        f"- Method: ResNet18 (ImageNet pretrained) features + IsolationForest",
        f"- Device: {device}",
        f"- Input: {_rel(image_dir)} ({len(case_results)} cases)",
        f"- Contamination: {contamination}",
        "",
        f"## Results",
        f"- Total cases: {len(case_results)}",
        f"- Flagged as anomaly: {anomaly_count}",
        f"- Normal: {normal_count}",
        "",
        "## Top 10 most anomalous cases",
        "",
        "| Case ID | Score | Flag |",
        "| --- | --- | --- |",
    ]
    ranked = sorted(case_results, key=lambda c: c["anomaly_score"])
    for c in ranked[:10]:
        flag = "⚠️ anomaly" if c["is_anomaly"] else "normal"
        lines.append(f"| {c['case_id']} | {c['anomaly_score']:.6f} | {flag} |")

    lines += [
        "",
        "## Limitations",
        "- This is an unsupervised baseline; anomaly flags are NOT clinical diagnoses.",
        "- Scores are relative to the cohort; no calibration against ground truth.",
        "- Pretrained features are from ImageNet (natural images), not dental X-rays.",
    ]

    (out_dir / "anomaly_detection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dental 2D anomaly detection MVP baseline")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/platform_mvp_runs" / f"anomaly_detection_demo_{_now_stamp()}")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected fraction of anomalies (default 0.1)")
    parser.add_argument("--max-images", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_anomaly_detection(
        image_dir=args.input_dir,
        out_dir=args.out_dir,
        contamination=args.contamination,
        max_images=args.max_images,
    )
    print("Dental anomaly detection MVP completed.")
    print(f"Output: {_rel(args.out_dir)}")
    print(f"Total: {summary.get('total_cases', 0)}, anomaly: {summary.get('anomaly_count', 0)}, normal: {summary.get('normal_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
