#!/usr/bin/env python3
"""
D2/T2: 对通用 AI 工具（Codex/Copilot 等）的 baseline 决策输出按同一六维 rubric 评分。

输入：<results-dir>/<intent_id>.json（每个意图一个文件，schema 见 baseline_prompt_pack.md）
输出：该 baseline 的均值 / 类别均分 / 与平台（0.9583）的对比

用法:
    python benchmark_trace/score_baseline.py --results-dir benchmark_results/baselines/codex \
        --label codex [--platform-report benchmark_results/eval_platform_mvp_30_20260818_v2.json]
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

INTENTS_FILE = Path(__file__).resolve().parents[1] / "benchmark_intents" / "intents.platform_mvp_30.jsonl"


def load_intents() -> list:
    with open(INTENTS_FILE, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def baseline_to_plan(blob: dict) -> dict:
    """把 baseline 输出 JSON 转成 eval 期望的 plan 结构。
    reason 映射到 selected_method.agent_note，与平台口径一致（外部提案维度需 agent_note 长度 >20 才满分）。
    """
    sm = blob.get("selected_method") or {}
    return {
        "supported": blob.get("supported"),
        "executable": blob.get("executable"),
        "selected_method": {
            "id": sm.get("id"),
            "status": sm.get("status", ""),
            "agent_note": blob.get("reason", ""),
        },
        "intent": {
            "dataset": (blob.get("intent") or {}).get("dataset", "unknown"),
            "task_family": (blob.get("intent") or {}).get("task_family", "unknown"),
            "mode": (blob.get("intent") or {}).get("mode", "unknown"),
        },
        "workflow": [],
        "reason": blob.get("reason", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--platform-report", default=None)
    args = parser.parse_args()

    intents = load_intents()
    by_id = {i["id"]: i for i in intents}
    results_dir = Path(args.results_dir)

    results, missing, failed = [], [], []
    for intent in intents:
        iid = intent["id"]
        f = results_dir / f"{iid}.json"
        if not f.exists():
            missing.append(iid)
            continue
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            failed.append((iid, f"JSON 解析失败: {e}"))
            continue
        try:
            plan = baseline_to_plan(blob)
            r = evaluate_one(intent, plan, 0, "")
            r["model"] = blob.get("model", "?")
            r["date"] = blob.get("date", "?")
            results.append(r)
        except Exception as e:
            failed.append((iid, f"评分失败: {e}"))

    print(f"\n[{args.label}] 已评分 {len(results)}/{len(intents)} 条")
    if missing:
        print(f"缺文件: {missing}")
    if failed:
        for iid, err in failed:
            print(f"  ⚠️ {iid}: {err}")

    if not results:
        sys.exit(1)

    print_report(results)

    avg = sum(r["total_score"] for r in results) / len(results)
    print(f"\n[{args.label}] 均值 = {avg:.4f}")

    if args.platform_report and Path(args.platform_report).exists():
        plat = json.load(open(args.platform_report, encoding="utf-8"))
        plat_avg = plat.get("avg_score")
        print(f"[平台 DentalClaw] 均值 = {plat_avg:.4f}")
        print(f"[对比] {args.label} - DentalClaw = {avg - plat_avg:+.4f}")

    out = args.results_dir + "_scored.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline": args.label,
        "total_intents": len(results),
        "avg_score": avg,
        "passed": sum(1 for r in results if r["total_score"] >= 0.70),
        "dimension_averages": {
            d: sum(r["dimension_scores"].get(d, 0) for r in results) / len(results)
            for d in DIMENSION_WEIGHTS
        },
        "results": results,
        "missing": missing,
        "failed": failed,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"评分报告: {out}")


if __name__ == "__main__":
    main()
