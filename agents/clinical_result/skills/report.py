\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ARTIFACT_ROOT = Path("/data/data2/yiyang/DentalClaw/artifacts")
DEFAULT_CLINICAL_WORKSPACE = ARTIFACT_ROOT / "results" / "reports" / "clinical_result_workspace"


def _ensure_under_artifacts(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must live under {ARTIFACT_ROOT}, got {resolved}") from exc
    return resolved


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _risk_level(foreground_pixels: int) -> Tuple[str, str]:
    if foreground_pixels <= 0:
        return "Low", "🟢"
    if foreground_pixels < 100_000:
        return "Moderate", "🟠"
    if foreground_pixels < 1_000_000:
        return "High", "🔴"
    return "High", "🔴"


def _make_overlay(image: Optional[np.ndarray], mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask.shape}")

    h, w = mask.shape
    if image is None:
        base = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        if image.ndim == 2:
            base = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            base = image.copy()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if base.shape[:2] != (h, w):
            base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = base.copy()
    labels = [int(v) for v in np.unique(mask) if int(v) != 0]
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (138, 110, 212),
        (80, 200, 120),
    ]
    for idx, label in enumerate(labels):
        color = np.array(palette[idx % len(palette)], dtype=np.uint8)
        region = mask == label
        overlay[region] = (
            alpha * overlay[region].astype(np.float32)
            + (1.0 - alpha) * color.astype(np.float32)
        ).astype(np.uint8)
    return overlay


def _to_md_table(rows: List[Tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _html_table(rows: List[Tuple[str, Any]]) -> str:
    out = ["<table class='summary-table'>", "<tr><th>Field</th><th>Value</th></tr>"]
    for k, v in rows:
        out.append(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def _bullet_list(items: List[str]) -> str:
    if not items:
        return "<li>None</li>"
    return "".join(f"<li>{html.escape(str(x))}</li>" for x in items)


def _infer_governance_tags(summary: Dict[str, Any], workflow_meta: Dict[str, Any], model_outputs: List[Dict[str, Any]], evaluation_summary: Optional[Dict[str, Any]]) -> List[str]:
    tags: List[str] = []
    if workflow_meta.get("use_ensemble"):
        tags.append("ensemble_inference")
    if workflow_meta.get("use_tta"):
        tags.append("tta_enabled")
    if workflow_meta.get("full_panoramic_context"):
        tags.append("full_panorama_context")
    if workflow_meta.get("input_scale_mismatch_risk"):
        tags.append("input_scale_mismatch_risk")
    if evaluation_summary is None or evaluation_summary.get("reference_available") is False:
        tags.append("no_reference_mask")
    if summary.get("foreground_pixels", 0) >= 0.8 * max(workflow_meta.get("image_area", 1), 1):
        tags.append("possible_oversegmentation")
    if any(m.get("foreground_pixels", 0) > 0 for m in model_outputs):
        tags.append("model_output_non_empty")
    return sorted(set(tags))


def _case_notes(summary: Dict[str, Any], workflow_meta: Dict[str, Any], evaluation_summary: Optional[Dict[str, Any]]) -> List[str]:
    notes = []
    if workflow_meta.get("full_panoramic_context"):
        notes.append("The case was processed as a full panoramic radiograph rather than a cropped tooth patch.")
    if workflow_meta.get("input_scale_mismatch_risk"):
        notes.append("The trained model was optimized for patch-level inputs, so full-image inference may be scale-sensitive.")
    if summary.get("foreground_pixels", 0) >= 0.8 * max(workflow_meta.get("image_area", 1), 1):
        notes.append("The predicted foreground is extensive and may indicate over-segmentation on the full panorama.")
    if evaluation_summary is None or evaluation_summary.get("reference_available") is False:
        notes.append("No reference mask was provided; quantitative performance is reported from inference outputs only.")
    return notes


def _compute_metrics(prediction: np.ndarray, target: np.ndarray, num_classes: int = 32):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    eps = 1e-5
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)

    per_class_dice = {}
    per_class_hd95 = {}
    dice_values = []
    hd95_values = []

    for class_index in range(1, num_classes + 1):
        pred_mask = prediction == class_index
        target_mask = target == class_index
        denom = pred_mask.sum() + target_mask.sum()
        if denom == 0:
            continue

        dice = (2.0 * float((pred_mask & target_mask).sum()) + eps) / (float(denom) + eps)
        per_class_dice[str(class_index)] = dice
        dice_values.append(dice)

        if not pred_mask.any() and not target_mask.any():
            hd95 = 0.0
        elif not pred_mask.any() or not target_mask.any():
            hd95 = float(np.hypot(*pred_mask.shape))
        else:
            pred_surface = np.logical_xor(pred_mask, binary_erosion(pred_mask, border_value=0))
            target_surface = np.logical_xor(target_mask, binary_erosion(target_mask, border_value=0))
            if not pred_surface.any():
                pred_surface = pred_mask
            if not target_surface.any():
                target_surface = target_mask
            dist_to_target = distance_transform_edt(~target_surface)
            dist_to_pred = distance_transform_edt(~pred_surface)
            surface_distances = np.concatenate([dist_to_target[pred_surface], dist_to_pred[target_surface]])
            hd95 = float(np.percentile(surface_distances, 95.0)) if surface_distances.size else 0.0

        per_class_hd95[str(class_index)] = hd95
        hd95_values.append(hd95)

    return {
        "reference_available": True,
        "mean_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "mean_hd95": float(np.mean(hd95_values)) if hd95_values else 0.0,
        "pixel_accuracy": float((prediction == target).mean()),
        "per_class_dice": per_class_dice,
        "per_class_hd95": per_class_hd95,
    }


def _render_markdown(case_id: str,
                     summary: Dict[str, Any],
                     workflow_meta: Dict[str, Any],
                     model_outputs: List[Dict[str, Any]],
                     evaluation_summary: Optional[Dict[str, Any]],
                     governance_tags: List[str],
                     notes: List[str],
                     input_rel: Optional[str],
                     overlay_rel: str) -> str:
    risk = summary["risk_level"]
    risk_emoji = "🟢" if risk == "Low" else "🟠" if risk == "Moderate" else "🔴"

    lines = [
        "# Clinical Radiographic Report (AI-assisted)",
        "",
        "## 🧾 Case Information",
        _to_md_table([
            ("Case ID", case_id),
            ("Imaging Modality", workflow_meta.get("modality", "Panoramic Dental Radiograph")),
            ("Analysis Type", workflow_meta.get("analysis_type", "AI-assisted segmentation and review")),
        ]),
        "",
        "## 🧭 Workflow Metadata",
        _to_md_table([
            ("Pipeline", workflow_meta.get("pipeline_name", "clinical_result_pipeline")),
            ("Run Time (UTC)", workflow_meta.get("run_time_utc", _now_utc())),
            ("Input Image", workflow_meta.get("image_path", "n/a")),
            ("Output Directory", workflow_meta.get("out_dir", "n/a")),
            ("Use TTA", str(workflow_meta.get("use_tta", False))),
            ("Use Ensemble", str(workflow_meta.get("use_ensemble", False))),
            ("Models Used", ", ".join(workflow_meta.get("model_names", [])) or "n/a"),
            ("Input Size", workflow_meta.get("input_size", "n/a")),
        ]),
        "",
        "## 🗂 Dataset Context & Governance",
        _to_md_table([
            ("Context", workflow_meta.get("dataset_context_name", "Dental panoramic radiograph")),
            ("Governance Tags", ", ".join(governance_tags) if governance_tags else "None"),
        ]),
        "",
    ]

    if input_rel:
        lines += [
            "## 🖼 Input Image",
            f"![Input Image]({input_rel})",
            "",
        ]

    lines += [
        "## 🔍 Imaging Findings",
        _to_md_table([
            ("Foreground pixel count", f'{summary["foreground_pixels"]:,}'),
            ("Present labels", ", ".join(map(str, summary["present_labels"])) if summary["present_labels"] else "None"),
            ("Risk level", f"{risk_emoji} {risk}"),
        ]),
        "",
        "## 🤖 Quantitative Output Summary",
    ]

    if model_outputs:
        lines += ["| Model | Foreground pixels | Labels detected | Notes |", "| --- | --- | --- | --- |"]
        for item in model_outputs:
            labels = ", ".join(map(str, item.get("labels_detected", []))) if item.get("labels_detected") else "None"
            lines.append(f"| {item.get('model_name', 'model')} | {item.get('foreground_pixels', 0):,} | {labels} | {item.get('notes', '')} |")
    else:
        lines.append("- No per-model outputs recorded.")
    lines.append("")

    lines += ["## 📈 Model Performance Summary"]
    if evaluation_summary and evaluation_summary.get("reference_available"):
        lines += [
            _to_md_table([
                ("Mean Dice", f'{evaluation_summary.get("mean_dice", 0.0):.4f}'),
                ("Mean HD95", f'{evaluation_summary.get("mean_hd95", 0.0):.4f}'),
                ("Pixel Accuracy", f'{evaluation_summary.get("pixel_accuracy", 0.0):.4f}'),
            ]),
            "",
        ]
    else:
        lines += [
            "- Reference mask not provided; inference-output statistics are reported instead of ground-truth performance.",
            "",
        ]

    lines += [
        "## 🧠 Review-Relevant Findings",
    ]
    lines += [f"- {n}" for n in notes] if notes else ["- None"]
    lines += [
        "",
        "## 🧠 Clinical Impression",
        summary["clinical_conclusion"],
        "",
        "## 📋 Recommendations",
    ]
    recs = workflow_meta.get("recommendations") or [
        "Review segmentation boundaries for over-segmentation or omission.",
        "Compare the result against the raw image and clinical context.",
        "Prioritize manual review when the predicted foreground is extensive.",
    ]
    lines += [f"- {r}" for r in recs]
    lines += [
        "",
        "## 📌 Structured Summary",
        _to_md_table([
            ("Case ID", case_id),
            ("Foreground pixels", f'{summary["foreground_pixels"]:,}'),
            ("Risk level", f"{risk_emoji} {risk}"),
            ("Conclusion", summary["clinical_conclusion"]),
        ]),
        "",
        "## 🖼 Figure",
        "The overlay image is attached separately.",
        "",
        "## ⚠️ Disclaimer",
        "This result is for clinical review support only and should not replace the clinician's final judgment.",
        "",
    ]
    return "\n".join(lines)

def _render_html(case_id: str,
                 summary: Dict[str, Any],
                 workflow_meta: Dict[str, Any],
                 model_outputs: List[Dict[str, Any]],
                 evaluation_summary: Optional[Dict[str, Any]],
                 governance_tags: List[str],
                 notes: List[str],
                 input_rel: Optional[str],
                 overlay_rel: str,
                 workspace_rel: str,
                 extra_rel: Optional[str],) -> str:
                 
    import html

    def esc(x):
        return html.escape("" if x is None else str(x))

    def fmt_int(x):
        try:
            return format(int(x), ",")
        except Exception:
            return str(x)

    def table(rows):
        out = ["<table class='summary-table'>", "<tr><th>Field</th><th>Value</th></tr>"]
        for k, v in rows:
            out.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
        out.append("</table>")
        return "\n".join(out)

    risk = summary.get("risk_level", "Unknown")
    risk_color = "#10b981" if risk == "Low" else "#f59e0b" if risk == "Moderate" else "#ef4444"
    risk_badge = f"{'🟢' if risk == 'Low' else '🟠' if risk == 'Moderate' else '🔴'} {esc(risk)}"

    fg = summary.get("foreground_pixels", summary.get("positive_pixels", 0))
    labels = summary.get("present_labels", [])

    # ----- model outputs table -----
    model_rows = []
    for item in model_outputs:
        model_name = item.get("model_name", "model")
        pixels = fmt_int(item.get("foreground_pixels", 0))
        labels_detected = ", ".join(map(str, item.get("labels_detected", []))) if item.get("labels_detected") else "None"
        notes_text = item.get("notes", "")
        model_rows.append(
            "<tr>"
            f"<td>{esc(model_name)}</td>"
            f"<td>{esc(pixels)}</td>"
            f"<td>{esc(labels_detected)}</td>"
            f"<td>{esc(notes_text)}</td>"
            "</tr>"
        )

    model_table = (
        "<table class='summary-table compact-table'>"
        "<tr><th>Model</th><th>Pixels</th><th>Labels</th><th>Notes</th></tr>"
        + "".join(model_rows if model_rows else ["<tr><td colspan='4'>No per-model outputs recorded.</td></tr>"])
        + "</table>"
    )

    # ----- evaluation -----
    if evaluation_summary and evaluation_summary.get("reference_available"):
        mean_dice = evaluation_summary.get("mean_dice", 0.0)
        mean_hd95 = evaluation_summary.get("mean_hd95", 0.0)
        pixel_acc = evaluation_summary.get("pixel_accuracy", 0.0)
        perf_html = f"""
        <table class="summary-table compact-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean Dice</td><td>{mean_dice:.4f}</td></tr>
            <tr><td>Mean HD95</td><td>{mean_hd95:.2f}</td></tr>
            <tr><td>Pixel Accuracy</td><td>{pixel_acc:.4f}</td></tr>
        </table>
        """
    else:
        perf_html = """
        <div class="muted-block">
          Reference mask not provided; inference-output statistics are reported instead of ground-truth performance.
        </div>
        """

    # ----- governance / notes -----
    tags_html = "".join(f"<span class='tag'>{esc(t)}</span>" for t in governance_tags) if governance_tags else "<span class='tag muted'>None</span>"
    notes_html = "".join(f"<li>{esc(n)}</li>" for n in notes) if notes else "<li>None</li>"
    recs = workflow_meta.get("recommendations") or [
        "Review segmentation boundaries for over-segmentation or omission.",
        "Compare the result against the raw image and clinical context.",
        "Prioritize manual review when the predicted foreground is extensive.",
    ]
    rec_html = "".join(f"<li>{esc(r)}</li>" for r in recs)

    # ----- image blocks -----
    if input_rel:
        input_img_block = f"""
        <section class="panel image-panel">
          <div class="panel-title">Input Image</div>
          <img class="figure" src="{esc(input_rel)}" alt="input image">
        </section>
        """
    else:
        input_img_block = """
        <section class="panel image-panel">
          <div class="panel-title">Input Image</div>
          <div class="empty-box">Input image unavailable.</div>
        </section>
        """

    overlay_block = f"""
    <section class="panel image-panel">
      <div class="panel-title">Overlay Figure</div>
      <img class="figure" src="{esc(overlay_rel)}" alt="overlay image">
      <div class="caption">Saved to: {esc(workspace_rel)}</div>
    </section>
    """

    extra_img_html = ""
    if extra_rel:
        extra_img_html = f"""
    <section class="panel">
      <div class="panel-body" style="text-align:center;">
        <img src="{extra_rel}" style="max-width:90%; border-radius:8px;">
      </div>
    </section>
    """
    

    # ----- final report -----
    case_id = summary.get("case_id", "")
    risk = summary.get("risk_level", "")
    pixels = summary.get("foreground_pixels", 0)
    labels = summary.get("present_labels", [])

    label_str = ", ".join(map(str, labels)) if labels else "None"

    clinical_impression_text = f"""
    For case <b>{esc(case_id)}</b>, the AI-assisted segmentation identified
    a total of <b>{format(int(pixels), ",")}</b> foreground pixels involving
    anatomical labels <b>{esc(label_str)}</b>.

    The overall risk stratification is assessed as <b>{esc(str(risk))}</b>.
    The extent and distribution of the segmented regions suggest that the
    findings should be interpreted together with the panoramic radiograph
    and the clinical context.

    Because this analysis was performed on a full panoramic image, the result
    may be influenced by scale differences between the training setting and
    the current input. Careful verification of anatomical boundaries is
    recommended to ensure that clinically relevant regions are not over- or
    under-segmented.
    """

    extra_findings = []
    extra_findings.extend(notes)

    extra_findings.append(
        "The analysis was performed using an nnUNet-based segmentation model under full-image inference conditions."
    )

    if workflow_meta.get("input_scale_mismatch_risk"):
        extra_findings.append(
            "Potential scale mismatch may affect segmentation fidelity due to differences between training patches and full-resolution panoramic inputs."
        )

    findings_html = "".join(f"<li>{esc(x)}</li>" for x in extra_findings)

    extra_recs = list(workflow_meta.get("recommendations", []))
    extra_recs.append(
        "Correlate segmentation results with anatomical landmarks and adjacent structures to ensure clinical consistency."
    )
    extra_recs.append(
        "If discrepancies are observed, consider re-evaluation using alternative preprocessing strategies or model configurations."
    )

    rec_html = "".join(f"<li>{esc(x)}</li>" for x in extra_recs)

    final_report_html = f"""
    <section class="panel final-panel">
      <div class="panel-title">Final Report</div>
      <div class="panel-body">
        <div class="report-block">
          <div class="report-title">Clinical Impression</div>
          <div class="report-text">{clinical_impression_text}</div>
        </div>

        <div class="report-block">
          <div class="report-title">Review-Relevant Findings</div>
          <ul class="tight-list">{findings_html}</ul>
        </div>

        <div class="report-block">
          <div class="report-title">Recommendations</div>
          <ul class="tight-list">{rec_html}</ul>
        </div>

        <div class="report-block">
          <div class="report-title">Disclaimer</div>
          <div class="report-text">
            This report is generated by an AI-assisted system and is intended for clinical decision support only.
            Final diagnosis should be made by a qualified dental professional.
          </div>
        </div>
      </div>
    </section>
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Report — {esc(case_id)}</title>
<style>
:root {{
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;
  --shadow: 0 12px 28px rgba(15, 23, 42, .08);
}}

* {{ box-sizing: border-box; }}

body {{
  margin: 0;
  background: linear-gradient(180deg, #eef3f8 0%, #f8fafc 100%);
  color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}}

.page {{
  max-width: 1780px;
  margin: 0 auto;
  padding: 18px;
}}

.hero {{
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
  color: white;
  border-radius: 20px;
  padding: 20px 24px;
  box-shadow: var(--shadow);
  margin-bottom: 14px;
}}

.hero h1 {{
  margin: 6px 0;
  font-size: 24px;
}}

.hero .meta {{
  font-size: 13px;
  gap: 10px;
}}

.grid {{
  display: grid;
  grid-template-columns: 0.85fr 1.55fr 1.25fr; /* 👈 关键：压左，扩右 */
  gap: 12px;
}}

.column {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}}
.column .panel:last-child {{
  flex: 1;               
}}
.middle-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  flex: 1;
}}

.middle-span-2 {{
  grid-column: span 2;
}}

.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}

.panel-title {{
  padding: 10px 12px;
  font-weight: 700;
  font-size: 13.5px;
  border-bottom: 1px solid var(--border);
}}

.panel-body {{
  padding: 10px 12px;
  flex : 1;
}}

.image-panel .figure {{
  width: 100%;
  max-height: 260px;   
  object-fit: contain; 
  background: #fff;
}}

.caption {{
  padding: 6px 10px;
  font-size: 11px;
  text-align: center;
  color: var(--muted);
}}

.summary-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}}

.summary-table th,
.summary-table td {{
  border: 1px solid var(--border);
  padding: 6px 7px;
}}

.summary-table th {{
  background: #f9fafb;
  width: 25%;
}}

.compact-table th,
.compact-table td {{
  font-size: 12px;
}}

.tag {{
  display: inline-block;
  background: #e5e7eb;
  padding: 3px 7px;
  margin: 2px;
  border-radius: 999px;
  font-size: 11px;
}}

.muted-block {{
  font-size: 12px;
  color: var(--muted);
}}

.final-panel {{
  font-size: 13.5px;  /* 👈 整体放大 */
}}

.final-panel .report-title {{
  font-size: 14.5px;
}}

.final-panel .report-text {{
  font-size: 13.5px;
  line-height: 1.65;
}}

.tight-list {{
  margin: 5px 0 0 16px;
  font-size: 13px;
}}

.tight-list li {{
  margin: 3px 0;
}}

.empty-box {{
  padding: 18px;
  text-align: center;
  font-size: 12px;
  color: var(--muted);
}}

.footer {{
  margin-top: 12px;
  text-align: center;   /* 👈 居中 */
  font-size: 12px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px;
}}
@media (max-width: 1400px) {{
  .grid {{
    grid-template-columns: 1fr;
  }}
  .middle-grid {{
    grid-template-columns: 1fr;
  }}
  .middle-span-2 {{
    grid-column: span 1;
  }}
}}
</style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="kicker">DentalClaw Clinical Report</div>
      <h1>AI-assisted dental panoramic analysis</h1>
      <div class="meta">
        <span>Case: {esc(case_id)}</span>
        <span>Risk: {risk_badge}</span>
        <span>Foreground pixels: {fmt_int(fg)}</span>
      </div>
    </header>

    <main class="grid">
      <!-- LEFT: evidence -->
      <div class="column">
        {input_img_block}
        {overlay_block}
        {extra_img_html}  
      </div>

      <!-- MIDDLE: metadata + quant + performance -->
      <div class="column">
        <div class="middle-grid">
          <section class="panel">
            <div class="panel-title">Workflow Metadata</div>
            <div class="panel-body">
              <table class="summary-table compact-table">
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Pipeline</td><td>{esc(workflow_meta.get("pipeline_name", "clinical_result_pipeline"))}</td></tr>
                <tr><td>Run Time (UTC)</td><td>{esc(workflow_meta.get("run_time_utc", ""))}</td></tr>
                <tr><td>Use TTA</td><td>{esc(workflow_meta.get("use_tta", False))}</td></tr>
                <tr><td>Use Ensemble</td><td>{esc(workflow_meta.get("use_ensemble", False))}</td></tr>
              </table>
            </div>
          </section>

          <section class="panel">
            <div class="panel-title">Dataset Context & Governance</div>
            <div class="panel-body">
              <table class="summary-table compact-table">
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Context</td><td>{esc(workflow_meta.get("dataset_context_name", "Dental panoramic radiograph"))}</td></tr>
                <tr><td>Tags</td><td>{tags_html}</td></tr>
              </table>
            </div>
          </section>

          <section class="panel">
            <div class="panel-title">Imaging Findings</div>
            <div class="panel-body">
              <table class="summary-table compact-table">
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Foreground pixels</td><td>{fmt_int(fg)}</td></tr>
                <tr><td>Present labels</td><td>{esc(", ".join(map(str, labels)) if labels else "None")}</td></tr>
                <tr><td>Risk level</td><td>{risk_badge}</td></tr>
              </table>
            </div>
          </section>

          <section class="panel">
            <div class="panel-title">Model Performance Summary</div>
            <div class="panel-body">
              {perf_html}
            </div>
          </section>

          <section class="panel middle-span-2">
            <div class="panel-title">Quantitative Output Summary</div>
            <div class="panel-body">
              <table class="summary-table compact-table">
                <colgroup>
                    <col style="width: 20%;">   <!-- Model -->
                    <col style="width: 12%;">   <!-- Pixels（变窄） -->
                    <col style="width: 28%;">   <!-- Labels -->
                    <col style="width: 40%;">   <!-- Notes（变宽） -->
                </colgroup>
                <tr><th>Model</th><th>Pixels</th><th>Labels</th><th>Notes</th></tr>
                {''.join(model_rows) if model_rows else '<tr><td colspan="4">No per-model outputs recorded.</td></tr>'}
              </table>
            </div>
          </section>
        </div>
      </div>

      <!-- RIGHT: final report -->
      <div class="column">
        {final_report_html}
      </div>
    </main>

    <div class="footer">
      This report is for clinical decision support only and does not replace professional judgment.
    </div>
  </div>
</body>
</html>
"""
    return html_doc


def clinical_report_export(
    case: Dict[str, Any],
    mask: np.ndarray,
    out_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    workflow_meta: Optional[Dict[str, Any]] = None,
    model_outputs: Optional[List[Dict[str, Any]]] = None,
    evaluation_summary: Optional[Dict[str, Any]] = None,
    governance_tags: Optional[List[str]] = None,
    case_notes: Optional[List[str]] = None,
    dataset_context: Optional[List[str]] = None,
    reference_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    out_path = _ensure_under_artifacts(Path(out_dir), "clinical report output directory") if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    ws_path = _ensure_under_artifacts(
        Path(workspace_dir) if workspace_dir else DEFAULT_CLINICAL_WORKSPACE / case["id"],
        "clinical report workspace directory",
    )
    ws_path.mkdir(parents=True, exist_ok=True)

    if workflow_meta is None:
        workflow_meta = {}
    if model_outputs is None:
        model_outputs = []
    if governance_tags is None:
        governance_tags = []
    if case_notes is None:
        case_notes = []
    if dataset_context is None:
        dataset_context = []

    foreground_pixels = int((mask > 0).sum())
    present_labels = [int(v) for v in np.unique(mask) if int(v) != 0]
    risk_level, risk_emoji = _risk_level(foreground_pixels)
    clinical_conclusion = (
        "A substantial foreground region was detected and should be reviewed carefully."
        if foreground_pixels > 0
        else "No obvious foreground segmentation region was detected."
    )

    workflow_meta = {
        "pipeline_name": workflow_meta.get("pipeline_name", "clinical_result_pipeline"),
        "modality": workflow_meta.get("modality", "Panoramic Dental Radiograph"),
        "analysis_type": workflow_meta.get("analysis_type", "AI-assisted segmentation and review"),
        "dataset_context_name": workflow_meta.get("dataset_context_name", "Dental panoramic radiograph"),
        "run_time_utc": workflow_meta.get("run_time_utc", _now_utc()),
        "image_path": workflow_meta.get("image_path", case.get("image_path", "n/a")),
        "out_dir": workflow_meta.get("out_dir", str(out_path) if out_path else "n/a"),
        "use_tta": workflow_meta.get("use_tta", False),
        "use_ensemble": workflow_meta.get("use_ensemble", False),
        "model_names": workflow_meta.get("model_names", []),
        "input_size": workflow_meta.get("input_size", "n/a"),
        "image_area": workflow_meta.get("image_area", 0),
        "full_panoramic_context": workflow_meta.get("full_panoramic_context", True),
        "input_scale_mismatch_risk": workflow_meta.get("input_scale_mismatch_risk", True),
        "recommendations": workflow_meta.get("recommendations", [
            "Review segmentation boundaries for over-segmentation or omission.",
            "Compare the result against the raw image and clinical context.",
            "Prioritize manual review when the predicted foreground is extensive.",
        ]),
    }
    
        # ===== 调试增强版 evaluation_summary 生成 =====
    print(f"[DEBUG] report: reference_mask_path = {reference_mask_path}")
    evaluation_summary = None

    if reference_mask_path:
        print("[DEBUG] Entering reference_mask_path branch")
        ref_path = Path(reference_mask_path)
        if ref_path.exists():
            print(f"[DEBUG] Reference file exists: {ref_path}")
            gt = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                print(f"[DEBUG] GT loaded, shape={gt.shape}, dtype={gt.dtype}")
                if gt.shape != mask.shape:
                    print(f"[DEBUG] Resizing GT from {gt.shape} to {mask.shape}")
                    gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                try:
                    evaluation_summary = _compute_metrics(mask, gt)
                    print("[DEBUG] _compute_metrics succeeded")
                    # 确保 evaluation_summary 包含 reference_available=True
                    if evaluation_summary:
                        evaluation_summary["reference_available"] = True
                except Exception as e:
                    print(f"[DEBUG] _compute_metrics failed: {e}")
                    import traceback
                    traceback.print_exc()
                    evaluation_summary = {
                        "reference_available": False,
                        "note": f"metrics computation error: {str(e)}",
                    }
            else:
                print("[DEBUG] GT is None (cv2.imread failed)")
                evaluation_summary = {
                    "reference_available": False,
                    "note": "reference mask unreadable (cv2.imread returned None)",
                }
        else:
            print(f"[DEBUG] Reference file does NOT exist: {ref_path}")
            evaluation_summary = {
                "reference_available": False,
                "note": "reference mask not found",
            }
    else:
        print("[DEBUG] reference_mask_path is None or empty")
        evaluation_summary = {
            "reference_available": False,
            "note": "no reference mask provided",
        }

    print(f"[DEBUG] Final evaluation_summary = {evaluation_summary}")


    summary = {
        "case_id": case["id"],
        "foreground_pixels": foreground_pixels,
        "present_labels": present_labels,
        "risk_level": risk_level,
        "clinical_conclusion": clinical_conclusion,
        "workflow_meta": workflow_meta,
        "dataset_context": dataset_context,
        "governance_tags": governance_tags,
        "case_notes": case_notes,
        "model_outputs": model_outputs,
        "evaluation_summary": evaluation_summary,
    }

    image = None
    image_path = case.get("image_path")
    if image_path:
        raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 3 and raw.shape[2] == 4:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            elif raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            image = raw

    import shutil

    extra_img_src = "/data/data2/yiyang/DentalClaw/agents/clinical_result/dentalclaw1.png"
    extra_img_dst = Path(out_dir) / "dentalclaw1.png"

    if Path(extra_img_src).exists():
        shutil.copy(extra_img_src, extra_img_dst)
        extra_rel = "dentalclaw1.png"
    else:
        extra_rel = None


    overlay = _make_overlay(image, mask)
    overlay_out = None
    overlay_ws = ws_path / "overlay.png"

    if out_path is not None:
        overlay_out = out_path / "overlay.png"
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            cv2.imwrite(str(overlay_out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(overlay_out), overlay)
        shutil.copy2(overlay_out, overlay_ws)
    else:
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            cv2.imwrite(str(overlay_ws), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(overlay_ws), overlay)

    overlay_path = str(overlay_ws if overlay_ws.exists() else overlay_out)
    input_rel = None
    if out_path is not None and image_path:
        src = Path(image_path)
        if src.exists():
            copied = out_path / "input_image.png"
            shutil.copy2(src, copied)
            input_rel = copied.name

    if not governance_tags:
        governance_tags = _infer_governance_tags(summary, workflow_meta, model_outputs, evaluation_summary)
    if not case_notes:
        case_notes = _case_notes(summary, workflow_meta, evaluation_summary)
    
    



    report_md = _render_markdown(
        case_id=case["id"],
        summary=summary,
        workflow_meta=workflow_meta,
        model_outputs=model_outputs,
        evaluation_summary=evaluation_summary,
        governance_tags=governance_tags,
        notes=case_notes,
        input_rel=input_rel,
        overlay_rel=Path(overlay_path).name,
    )
    report_html = _render_html(
        case_id=case["id"],
        summary=summary,
        workflow_meta=workflow_meta,
        model_outputs=model_outputs,
        evaluation_summary=evaluation_summary,
        governance_tags=governance_tags,
        notes=case_notes,
        input_rel=input_rel,
        overlay_rel=Path(overlay_path).name,
        workspace_rel=overlay_path,
        extra_rel=extra_rel,
    )

    html_path = None
    if out_path is not None:
        (out_path / "report.md").write_text(report_md, encoding="utf-8")
        html_path = out_path / "report.html"
        html_path.write_text(report_html, encoding="utf-8")
        (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_path / "review_list.json").write_text(json.dumps(case_notes, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "review_list": case_notes,
        "report": report_md,
        "report_html": report_html,
        "overlay_path": overlay_path,
        "html_path": str(html_path) if html_path else None,
        "input_image_path": str(out_path / "input_image.png") if out_path and input_rel else None,
        "evaluation_summary": evaluation_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an English clinical report + overlay figure.")
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--image_path", required=False, default="")
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workspace_dir", default=str(DEFAULT_CLINICAL_WORKSPACE))
    args = parser.parse_args()

    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask_path)

    report = clinical_report_export(
        case={"id": args.case_id, "image_path": args.image_path},
        mask=mask,
        out_dir=args.out_dir,
        workspace_dir=args.workspace_dir,
    )

    print(report["report"])
    print(f"HTML: {report['html_path']}")
    print(f"Overlay: {report['overlay_path']}")


if __name__ == "__main__":
    main()
