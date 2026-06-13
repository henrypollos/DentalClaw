\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from agents.clinical_result.agent import ClinicalResultAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clinical result pipeline for a new case.")
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_paths", required=True, help="Comma-separated list of model checkpoint paths.")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _parse_model_paths(raw: str) -> List[str]:
    paths = [p.strip() for p in raw.split(",") if p.strip()]
    if not paths:
        raise ValueError("model_paths cannot be empty")
    return paths


def main() -> None:
    args = parse_args()
    model_paths = _parse_model_paths(args.model_paths)

    case = {
        "id": args.case_id,
        "image_path": args.image_path,
        "modality": "2d",
    }

    agent = ClinicalResultAgent(
        config={
            "use_tta": bool(args.use_tta),
            "use_ensemble": bool(args.use_ensemble),
            "model_paths": model_paths,
            "min_size": 20,
        }
    )

    result = agent.run(case, out_dir=args.out_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "image_path": args.image_path,
                "model_paths": model_paths,
                "use_tta": bool(args.use_tta),
                "use_ensemble": bool(args.use_ensemble),
                "summary": result.get("summary"),
                "overlay_path": result.get("overlay_path"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result.get("text") or result.get("content") or result.get("report_text") or "")
    print(f"Overlay: {result.get('overlay_path', '')}")
    print(f"Output dir: {result.get('out_dir', args.out_dir)}")


if __name__ == "__main__":
    main()
