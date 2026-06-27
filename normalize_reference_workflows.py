from __future__ import annotations

# This file is generated from the experiment design updater.
# It preserves the 30 DentalClaw benchmark intents and regenerates the
# compatibility/evaluation JSONL files from benchmark_intents/intents.jsonl.

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
INTENT_DIR = ROOT / "benchmark_intents"
SOURCE = INTENT_DIR / "intents.jsonl"
COMPAT = INTENT_DIR / "intents.compat.jsonl"
EVAL = INTENT_DIR / "intents.eval.jsonl"
UNMAPPED_FILE = INTENT_DIR / "unmapped_reference_steps.txt"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    rows = read_jsonl(SOURCE)
    if len(rows) != 30:
        raise SystemExit(f"Expected 30 intents, got {len(rows)}")
    compat = []
    eval_rows = []
    for row in rows:
        raw = row.get("reference_workflow_path") or row.get("reference_workflow_raw") or []
        canonical = row.get("reference_workflow") or []
        if not raw or not canonical:
            raise SystemExit(f"Missing workflow for {row.get('id')}")
        compat_row = dict(row)
        compat_row["reference_workflow"] = raw
        compat.append(compat_row)
        eval_row = dict(row)
        eval_row["reference_workflow"] = canonical
        eval_row["reference_workflow_raw"] = raw
        eval_rows.append(eval_row)
    write_jsonl(COMPAT, compat)
    write_jsonl(EVAL, eval_rows)
    UNMAPPED_FILE.write_text("", encoding="utf-8")
    print("生成文件：", COMPAT)
    print("生成文件：", EVAL)
    print("意图数量：", len(rows))


if __name__ == "__main__":
    main()
