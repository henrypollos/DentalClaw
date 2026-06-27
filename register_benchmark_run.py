#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def read_jsonl(path: Path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--intent-id", required=True)
    p.add_argument("--benchmark-runs", default="benchmark_runs")
    p.add_argument("--run-index", default="benchmark_results/run_index.jsonl")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    root = Path(args.benchmark_runs)
    candidates = sorted(
        [x for x in root.glob(f"{args.intent_id}_*") if x.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"未找到 {root}/{args.intent_id}_*")

    run_dir = candidates[0]
    manifest = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    row = {
        "intent_id": args.intent_id,
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "status": manifest.get("status", "unknown"),
        "notes": args.notes,
    }

    index_path = Path(args.run_index)
    rows = [r for r in read_jsonl(index_path) if r.get("intent_id") != args.intent_id]
    rows.append(row)
    rows.sort(key=lambda r: r["intent_id"])
    write_jsonl(index_path, rows)

    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("已写入:", index_path)

if __name__ == "__main__":
    main()
