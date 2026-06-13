#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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


def _copy_image_to_output(image_path: Optional[str], out_path: Optional[Path]) -> Optional[str]:
    if not image_path or out_path is None:
        return None

    src = Path(image_path)
    if not src.exists():
        return None

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    dst = out_path / "input_image.png"
    cv2.imwrite(str(dst), img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img)
    # 说明：
    # cv2.imwrite 对 RGB 保存没强制要求，这里为了统一，转回 BGR 保存即可
    # 如果上面这行你担心颜色问题，也可以直接 cv2.imwrite(str(dst), img)
    return dst.name


def _build_subreports(summary: Dict[str, Any], review_list: List[str]) -> List[Dict[str, Any]]:
    fg = summary["foreground_pixels"]
    labels = summary["present_labels"]
    risk = summary["risk_level"]
    conclusion = summary["clinical_conclusion"]

    return [
        {
            "title": "🦷 Segmentation Sub-Report",
            "body": [
                f"Foreground pixels: {fg:,}.",
                f"Detected labels: {', '.join(map(str, labels)) if labels else 'None'}.",
                "The segmentation covers a broad region across the image.",
            ],
        },
        {
            "title": "🧠 AI Interpretation Sub-Report",
            "body": [
                f"Risk level: {risk}.",
                conclusion,
                "The result should be interpreted together with the raw panoramic image and clinical context.",
            ],
        },
        {
            "title": "🏥 Clinical Recommendation Sub-Report",
            "body": review_list,
        },
        {
            "title": "💊 Follow-up & Risk Management",
            "body": [
                "Prioritize manual review when the predicted foreground is extensive.",
                "If the result remains over-segmented, adjust inference scale or use a tiling strategy.",
                "Consider comparing the output with a baseline model or a different preprocessing pipeline.",
            ],
        },
    ]


def _render_markdown(case_id: str, summary: Dict[str, Any], subreports: List[Dict[str, Any]], input_rel: Optional[str], overlay_rel: str) -> str:
    risk = summary["risk_level"]
    risk_emoji = "🟢" if risk == "Low" else "🟠" if risk == "Moderate" else "🔴"

    md = [
        "# Clinical Result Report",
        "",
        "## 🧾 Case Information",
        "| Field | Value |",
        "| --- | --- |",
        f"| Case ID | {case_id} |",
        "| Report Language | English |",
        "| Output Type | Segmentation-based clinical review |",
        "",
    ]

    if input_rel:
        md += [
            "## 🖼 Input Image",
            f"![Input Image]({input_rel})",
            "",
        ]

    md += [
        "## 🔍 Imaging Findings",
        "| Field | Value |",
        "| --- | --- |",
        f'| Foreground pixel count | {summary["foreground_pixels"]:,} |',
        f'| Present labels | {", ".join(map(str, summary["present_labels"])) if summary["present_labels"] else "None"} |',
        f"| Risk level | {risk_emoji} {risk} |",
        "",
    ]

    for sec in subreports:
        md.append(f"## {sec['title']}")
        for line in sec["body"]:
            md.append(f"- {line}")
        md.append("")

    md += [
        "## 📌 Structured Summary",
        "| Field | Value |",
        "| --- | --- |",
        f"| Case ID | {case_id} |",
        f'| Foreground pixels | {summary["foreground_pixels"]:,} |',
        f"| Risk level | {risk_emoji} {risk} |",
        f'| Conclusion | {summary["clinical_conclusion"]} |',
        "",
        "## 🖼 Figure",
        "The overlay image is attached separately.",
        "",
        "## ⚠️ Disclaimer",
        "This result is for clinical review support only and should not replace the clinician's final judgment.",
        "",
    ]
    return "\n".join(md)


def _render_html(case_id: str, summary: Dict[str, Any], subreports: List[Dict[str, Any]], input_rel: Optional[str], overlay_rel: str, workspace_rel: str) -> str:
    def esc(x: Any) -> str:
        return html_lib.escape("" if x is None else str(x))

    risk = summary["risk_level"]
    badge_cls = "low" if risk == "Low" else "moderate" if risk == "Moderate" else "high"
    badge_text = f"{'🟢' if risk == 'Low' else '🟠' if risk == 'Moderate' else '🔴'} {esc(risk)}"

    left_image_html = ""
    if input_rel:
        left_image_html += f"""
        <div class="panel">
          <div class="panel-title">Input Image</div>
          <img class="figure" src="{esc(input_rel)}" alt="input image">
        </div>
        """

    left_image_html += f"""
        <div class="panel">
          <div class="panel-title">Overlay Figure</div>
          <img class="figure" src="{esc(overlay_rel)}" alt="overlay image">
          <div class="caption">Saved to: {esc(workspace_rel)}</div>
        </div>
    """

    mid_cards = []
    for sec in subreports:
        bullets = "".join(f"<li>{esc(line)}</li>" for line in sec["body"])
        mid_cards.append(f"""
        <section class="card">
          <div class="card-head">{esc(sec['title'])}</div>
          <ul>{bullets}</ul>
        </section>
        """)
    mid_html = "\n".join(mid_cards)

    summary_table = f"""
    <table class="summary-table">
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Case ID</td><td>{esc(case_id)}</td></tr>
      <tr><td>Foreground pixels</td><td>{esc(f'{summary["foreground_pixels"]:,}')}</td></tr>
      <tr><td>Present labels</td><td>{esc(", ".join(map(str, summary["present_labels"])) if summary["present_labels"] else "None")}</td></tr>
      <tr><td>Risk level</td><td><span class="badge {badge_cls}">{badge_text}</span></td></tr>
    </table>
    """

    recommendations = "".join(f"<li>{esc(x)}</li>" for x in subreports[2]["body"])

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Result Report — {esc(case_id)}</title>
<style>
:root {{
  --bg: #f4f6fa;
  --panel: #ffffff;
  --ink: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;
  --shadow: 0 18px 40px rgba(15, 23, 42, .08);
  --blue: #3b82f6;
  --green: #10b981;
  --orange: #f59e0b;
  --red: #ef4444;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: linear-gradient(180deg, #eef3f8 0%, #f8fafc 100%);
  color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}}
.page {{
  max-width: 1680px;
  margin: 0 auto;
  padding: 24px;
}}
.hero {{
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
  color: white;
  border-radius: 22px;
  padding: 26px 28px;
  box-shadow: var(--shadow);
  margin-bottom: 20px;
}}
.hero .kicker {{
  font-size: 12px;
  letter-spacing: .15em;
  text-transform: uppercase;
  opacity: .75;
}}
.hero h1 {{
  margin: 8px 0 10px;
  font-size: 30px;
  line-height: 1.15;
}}
.hero .meta {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  color: rgba(255,255,255,.88);
  font-size: 14px;
}}
.grid {{
  display: grid;
  grid-template-columns: 320px minmax(0, 1.15fr) minmax(340px, .9fr);
  gap: 18px;
  align-items: start;
}}
.column {{
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
.panel, .card, .final {{
  background: var(--panel);
  border: 1px solid rgba(229,231,235,.9);
  border-radius: 20px;
  box-shadow: var(--shadow);
  overflow: hidden;
}}
.panel-title, .card-head, .final-head {{
  padding: 14px 16px;
  font-weight: 700;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #fafbfc 0%, #f8fafc 100%);
}}
.panel img.figure {{
  display: block;
  width: 100%;
  height: auto;
  background: #fff;
}}
.panel .caption {{
  padding: 12px 16px 16px;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
}}
.card ul {{
  margin: 12px 18px 16px 32px;
  padding: 0;
  line-height: 1.6;
}}
.card li {{ margin: 6px 0; }}
.final-body {{
  padding: 16px;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
  color: white;
}}
.badge.low {{ background: var(--green); }}
.badge.moderate {{ background: var(--orange); }}
.badge.high {{ background: var(--red); }}
.summary-table {{
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 16px;
  overflow: hidden;
  border-radius: 14px;
}}
.summary-table th, .summary-table td {{
  border: 1px solid var(--border);
  padding: 11px 12px;
  text-align: left;
  vertical-align: top;
}}
.summary-table th {{
  background: #f9fafb;
  width: 38%;
}}
.notice {{
  padding: 14px 16px;
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 16px;
  color: #334155;
  line-height: 1.7;
}}
.section-stack {{
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
@media (max-width: 1200px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <div class="kicker">DentalClaw Clinical Result Report</div>
    <h1>Input: Dental Panoramic X-ray + Inference Result</h1>
    <div class="meta">
      <span>Case: {esc(case_id)}</span>
      <span>Risk: {badge_text}</span>
      <span>Foreground pixels: {esc(f'{summary["foreground_pixels"]:,}')}</span>
    </div>
  </header>

  <main class="grid">
    <div class="column">
      {left_image_html}
    </div>

    <div class="column section-stack">
      {mid_html}
    </div>

    <div class="column">
      <section class="final">
        <div class="final-head">Final Report</div>
        <div class="final-body">
          <div class="notice">
            <strong>Summary of Key Findings</strong><br>
            {esc(summary["clinical_conclusion"])}
          </div>

          <div style="height:12px"></div>

          {summary_table}

          <div class="notice">
            <strong>Clinical Impression</strong><br>
            A substantial foreground region was detected. Manual review is recommended, and the result should be interpreted in the context of the raw panoramic image and clinical note.
          </div>

          <div style="height:12px"></div>

          <div class="notice">
            <strong>Recommendations</strong>
            <ul style="margin:10px 0 0 22px; padding:0;">{recommendations}</ul>
          </div>

          <div style="height:12px"></div>

          <div class="notice">
            <strong>Disclaimer</strong><br>
            This result is for clinical review support only and should not replace the clinician's final judgment.
          </div>
        </div>
      </section>
    </div>
  </main>
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
) -> Dict[str, Any]:
    """
    Generate:
      - report.md
      - report.html
      - summary.json
      - review_list.json
      - overlay.png
      - input_image.png (if available)
    """
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

    foreground_pixels = int((mask > 0).sum())
    present_labels = [int(v) for v in np.unique(mask) if int(v) != 0]
    risk_level, risk_emoji = _risk_level(foreground_pixels)

    if foreground_pixels > 0:
        clinical_conclusion = "A substantial foreground region was detected and should be reviewed carefully."
        review_list = [
            "Review region boundaries for over-segmentation or omission.",
            "Compare the result against the raw image and clinical context.",
            "Prioritize manual review when the predicted foreground is extensive.",
        ]
    else:
        clinical_conclusion = "No obvious foreground segmentation region was detected."
        review_list = [
            "If the case is clinically suspicious, review the raw image and preprocessing steps.",
        ]

    summary = {
        "case_id": case["id"],
        "foreground_pixels": foreground_pixels,
        "present_labels": present_labels,
        "risk_level": risk_level,
        "clinical_conclusion": clinical_conclusion,
    }

    # Copy input image locally for the HTML report if it exists.
    input_image_rel = _copy_image_to_output(case.get("image_path"), out_path) if out_path else None

    # Build overlay from the original image.
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
    overlay_rel = Path(overlay_path).name

    subreports = _build_subreports(summary, review_list)
    report_md = _render_markdown(
        case_id=case["id"],
        summary=summary,
        subreports=subreports,
        input_rel=input_image_rel,
        overlay_rel=overlay_rel,
    )
    report_html = _render_html(
        case_id=case["id"],
        summary=summary,
        subreports=subreports,
        input_rel=input_image_rel,
        overlay_rel=overlay_rel,
        workspace_rel=str(overlay_path),
    )

    html_path = None
    if out_path is not None:
        (out_path / "report.md").write_text(report_md, encoding="utf-8")
        html_path = out_path / "report.html"
        html_path.write_text(report_html, encoding="utf-8")
        (out_path / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_path / "review_list.json").write_text(
            json.dumps(review_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "summary": summary,
        "review_list": review_list,
        "report": report_md,
        "report_html": report_html,
        "overlay_path": overlay_path,
        "html_path": str(html_path) if html_path else None,
        "input_image_path": str(out_path / "input_image.png") if out_path and input_image_rel else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an English clinical report + overlay figure.")
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--image_path", required=False, default="")
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--workspace_dir",
        default=str(DEFAULT_CLINICAL_WORKSPACE),
    )
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
