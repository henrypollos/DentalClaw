#!/usr/bin/env python3
"""
用已保存的 platform_plan.json（/tmp/eval_platform_mvp/<intent_id>/）按当前
intents.platform_mvp_30.jsonl 的期望重新评分，无需重新运行平台。

用法:
    python benchmark_trace/rescore_platform_mvp.py \
        --plans-dir /tmp/eval_platform_mvp \
        --json-report benchmark_results/eval_platform_mvp_30_20260818_rescored.json \
        [--prev-report benchmark_results/eval_platform_mvp_30_20260818.json]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_trace.eval_platform_mvp import (  # noqa: E402
    DIMENSION_WEIGHTS,
    evaluate_one,
    print_report,
)


def load_intents(path: Path):
    intents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                intents.append(json.loads(line))
    return intents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents-file", default="benchmark_intents/intents.platform_mvp_30.jsonl")
    parser.add_argument("--plans-dir", default="/tmp/eval_platform_mvp")
    parser.add_argument("--json-report", default="benchmark_results/eval_platform_mvp_30_rescored.json")
    parser.add_argument("--prev-report", default=None, help="旧报告，用于补充耗时字段")
    args = parser.parse_args()

    intents = load_intents(Path(args.intents_file))
    prev_elapsed = {}
    if args.prev_report and Path(args.prev_report).exists():
        prev = json.load(open(args.prev_report, encoding="utf-8"))
        prev_elapsed = {r["intent_id"]: r.get("elapsed_s", 0) for r in prev.get("results", [])}

    plans_dir = Path(args.plans_dir)
    results = []
    missing = []
    for intent in intents:
        plan_path = plans_dir / intent["id"] / "platform_plan.json"
        if not plan_path.exists():
            missing.append(intent["id"])
            print(f"⚠️ 缺 plan: {intent['id']}", file=sys.stderr)
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        elapsed = prev_elapsed.get(intent["id"], 0)
        results.append(evaluate_one(intent, plan, elapsed, ""))

    if missing:
        print(f"共 {len(missing)} 条缺 plan: {missing}", file=sys.stderr)

    print_report(results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_intents": len(results),
        "avg_score": sum(r["total_score"] for r in results) / len(results) if results else 0,
        "passed": sum(1 for r in results if r["total_score"] >= 0.70),
        "dimension_averages": {
            d: sum(r["dimension_scores"].get(d, 0) for r in results) / len(results)
            for d in DIMENSION_WEIGHTS
        },
        "results": results,
        "note": "rescore with corrected expected_platform_result (2026-08-18); plans unchanged from first run",
    }
    with open(args.json_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
