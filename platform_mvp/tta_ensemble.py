#!/usr/bin/env python3
"""DentalClaw TTA + Ensemble inference — platform MVP module.

Wraps agents/clinical_result/skills/ to provide a clean CLI for:
  single   — one model, standard inference
  tta      — one model + test-time augmentation (orig, hflip, vflip → vote)
  ensemble — multiple models → majority-vote merge
  full     — ensemble + per-model TTA

Usage:
  python platform_mvp/tta_ensemble.py --image <path> --models <dir1> [dir2...]
        --output <dir> [--tta] [--checkpoint checkpoint_best.pth]
"""

from __future__ import annotations

import argparse, json, sys, os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from scipy import stats

from agents.clinical_result.skills.nnunet_inference import (
    predict_single_case, predict_ensemble as _nnunet_ensemble,
)


# ── Geometry TTA transforms ────────────────────────────────────────────

def _apply(img: np.ndarray, t: str) -> np.ndarray:
    if t == "orig": return img
    if t == "hflip": return cv2.flip(img, 1)
    if t == "vflip": return cv2.flip(img, 0)
    if t == "rot90": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if t == "rot180": return cv2.rotate(img, cv2.ROTATE_180)
    if t == "rot270": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unknown: {t}")

def _invert(m: np.ndarray, t: str) -> np.ndarray:
    if t == "orig": return m
    if t == "hflip": return np.fliplr(m)
    if t == "vflip": return np.flipud(m)
    if t == "rot90": return cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if t == "rot180": return cv2.rotate(m, cv2.ROTATE_180)
    if t == "rot270": return cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unknown: {t}")

DEFAULT_TTA = ["orig", "hflip", "vflip"]


# ── TTA over a single model ────────────────────────────────────────────

def run_tta(image_path: str, model_folder: str, output_dir: str,
            transforms=None, use_folds=(0,),
            checkpoint="checkpoint_best.pth") -> dict:
    """Single model + TTA. Predicts each transform, inverts, majority-votes."""
    ip, od = Path(image_path), Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    transforms = list(transforms or DEFAULT_TTA)

    img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    masks, pts = [], []

    for tf in transforms:
        td = od / tf; td.mkdir(parents=True, exist_ok=True)
        ti = _apply(img, tf)
        tp = td / ip.name; cv2.imwrite(str(tp), ti)

        mask, _ = predict_single_case(
            str(tp), model_folder, str(td),
            use_folds=use_folds, checkpoint_name=checkpoint, use_tta=False,
        )
        mi = _invert(mask, tf)
        masks.append(mi)
        pts.append({"transform": tf, "foreground": int((mi > 0).sum())})

    stacked = np.stack(masks, axis=0)
    merged = stats.mode(stacked, axis=0, keepdims=False).mode.astype(np.uint8)
    mp = od / f"{ip.stem}_tta.png"; cv2.imwrite(str(mp), merged)
    return {"mask_path": str(mp), "foreground": int((merged > 0).sum()),
            "num_transforms": len(transforms), "per_transform": pts}


# ── Ensemble (multi-model) ─────────────────────────────────────────────

def run_ensemble(image_path: str, model_folders: Sequence[str],
                 output_dir: str, use_tta=False, transforms=None,
                 use_folds=(0,), checkpoint="checkpoint_best.pth") -> dict:
    """Multi-model ensemble. If use_tta, runs TTA per model then merges."""
    ip, od = Path(image_path), Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    if use_tta:
        # TTA per model → collect masks → manual merge
        mask_paths = []
        for idx, mf in enumerate(model_folders, start=1):
            md = od / f"model_{idx}" / "tta"; md.mkdir(parents=True, exist_ok=True)
            r = run_tta(str(ip), str(mf), str(md), transforms, use_folds, checkpoint)
            mask_paths.append(r["mask_path"])

        all_masks = []
        for mp_path in mask_paths:
            all_masks.append(cv2.imread(mp_path, cv2.IMREAD_GRAYSCALE))

        stacked = np.stack(all_masks, axis=0)
        ml = int(stacked.max())
        votes = np.zeros((ml + 1,) + stacked.shape[1:], dtype=np.uint16)
        for c in range(ml + 1):
            votes[c] = (stacked == c).sum(axis=0)
        merged = votes.argmax(axis=0).astype(np.uint8)

        ed = od / "ensemble" / "masks"; ed.mkdir(parents=True, exist_ok=True)
        mp = ed / f"{ip.stem}_ensemble.png"; cv2.imwrite(str(mp), merged)
        return {"mask_path": str(mp), "num_models": len(model_folders),
                "tta": True, "foreground": int((merged > 0).sum())}
    else:
        # Standard nnUNet ensemble (single forward pass per model, merge)
        merged, per_model, _ = _nnunet_ensemble(
            str(ip), list(map(str, model_folders)), str(od),
            use_folds=use_folds, checkpoint_name=checkpoint, use_tta=False,
        )
        ed = od / "ensemble" / "masks"; ed.mkdir(parents=True, exist_ok=True)
        mp = ed / f"{ip.stem}_ensemble.png"; cv2.imwrite(str(mp), merged)
        return {"mask_path": str(mp), "num_models": len(model_folders),
                "tta": False, "foreground": int((merged > 0).sum())}


# ── Pipeline entry ─────────────────────────────────────────────────────

def run_pipeline(image_path: str, model_folders: Sequence[str],
                 output_dir: str, *, use_tta=True, transforms=None,
                 use_folds=(0,), checkpoint="checkpoint_best.pth",
                 case_id=None) -> dict:
    """Auto-selects mode based on inputs."""
    ip, od = Path(image_path), Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    if len(model_folders) == 1 and not use_tta:
        mask, mp = predict_single_case(
            str(ip), str(model_folders[0]), str(od / "single"),
            use_folds=use_folds, checkpoint_name=checkpoint, use_tta=False,
        )
        mode = "single"
        result = {"mask_path": mp, "foreground": int((mask > 0).sum())}
    elif len(model_folders) == 1:
        result = run_tta(str(ip), str(model_folders[0]), str(od / "tta"),
                         transforms, use_folds, checkpoint)
        mode = "tta"
    else:
        result = run_ensemble(str(ip), list(map(str, model_folders)),
                              str(od), use_tta, transforms, use_folds, checkpoint)
        mode = "full" if use_tta else "ensemble"

    elapsed = (datetime.now() - t0).total_seconds()
    s = {
        "pipeline": "tta_ensemble", "mode": mode,
        "case_id": case_id or ip.stem, "image": str(ip),
        "num_models": len(model_folders),
        "models": [Path(m).name for m in model_folders],
        "tta": use_tta,
        "tta_transforms": list(transforms or (DEFAULT_TTA if use_tta else [])),
        "folds": list(use_folds), "checkpoint": checkpoint,
        "mask_path": result["mask_path"],
        "foreground_pixels": result["foreground"],
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (od / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    return {"summary": s, "summary_path": str(od / "summary.json")}


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DentalClaw TTA+Ensemble")
    ap.add_argument("--image", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="nnUNet model folders (with fold_X subdirs)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tta-transforms", nargs="*", default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--case-id", default=None)
    a = ap.parse_args()

    tf = a.tta_transforms if a.tta_transforms else None
    out = run_pipeline(
        image_path=a.image, model_folders=a.models,
        output_dir=a.output, use_tta=a.tta, transforms=tf,
        use_folds=tuple(a.folds), checkpoint=a.checkpoint,
        case_id=a.case_id,
    )
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
