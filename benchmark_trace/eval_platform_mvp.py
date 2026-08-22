#!/usr/bin/env python3
"""
Platform MVP Eval — 运行所有 platform_mvp 意图并评测 platform_plan.json 的决策正确性。

用法：
    # Dry-run (只评分已有的 plan，不重新执行):
    python benchmark_trace/eval_platform_mvp.py --dry-run --run-dir <dir>

    # 执行全部 25 条评测:
    python benchmark_trace/eval_platform_mvp.py --all

    # 只执行前 N 条:
    python benchmark_trace/eval_platform_mvp.py --limit 5

    # 输出 JSON 汇总:
    python benchmark_trace/eval_platform_mvp.py --all --json-report eval_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
INTENTS_FILE = REPO_ROOT / "benchmark_intents" / "intents.platform_mvp.jsonl"
RUNNER = "bash " + str(REPO_ROOT / "platform_mvp" / "run_platform_mvp.sh")

# ==============================================================================
# 评测维度与权重
# ==============================================================================

DIMENSION_WEIGHTS = {
    "planning": 0.35,           # 任务规划：supported/executable/method 是否正确
    "qc_blocking": 0.20,        # QC 阻断：该拒的是否拒了，不该拒的是否放行
    "intent_parsing": 0.15,     # 意图解析：dataset/task/mode 是否识别正确
    "external_proposal": 0.15,  # 外部方案提议：无 registry 匹配时 Agent 是否给出合理建议
    "ambiguity": 0.10,          # 歧义处理：模糊意图是否正确拒绝
    "boundary": 0.05,           # 边界识别：超出范围是否正确拒绝
}

# ==============================================================================
# 核心评测函数
# ==============================================================================

def load_intents() -> List[Dict[str, Any]]:
    intents = []
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                intents.append(json.loads(line))
    return intents


def run_intent(intent_zh: str, run_dir: Path, timeout: int = 120) -> Tuple[Optional[Dict], float, str]:
    """运行单条意图，返回 (platform_plan, elapsed_seconds, stderr_summary)。"""
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cmd = f"timeout {timeout} {RUNNER} --intent '{intent_zh}' --run-dir {run_dir}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return None, timeout, "TIMEOUT"

    elapsed = time.time() - t0
    stderr = result.stderr[-500:] if result.stderr else ""

    plan_path = run_dir / "platform_plan.json"
    if not plan_path.exists():
        return None, elapsed, f"NO_PLAN_FILE\nstderr:{stderr}"

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return None, elapsed, f"JSON_ERROR: {e}"

    return plan, elapsed, ""


def parse_actual(plan: Dict[str, Any]) -> Dict[str, Any]:
    """从 platform_plan.json 提取实际决策。"""
    sm = plan.get("selected_method") or {}

    # Determine executable status
    supported = plan.get("supported")
    executable = plan.get("executable")
    method_id = sm.get("id")
    method_status = sm.get("status", "")

    # External suggestion detection
    is_external = (method_id == "agent_external_suggestion" or
                   method_status == "external_suggestion" or
                   "external" in str(method_id).lower())

    # Intent parsing accuracy
    intent = plan.get("intent") or {}
    parsed_dataset = intent.get("dataset", "unknown")
    parsed_task = intent.get("task_family", "unknown")
    parsed_mode = intent.get("mode", "unknown")

    return {
        "supported": supported,
        "executable": executable,
        "method": method_id,
        "is_external": is_external,
        "parsed_dataset": parsed_dataset,
        "parsed_task": parsed_task,
        "parsed_mode": parsed_mode,
        "workflow": plan.get("workflow", []),
        "has_web_search": "web.search" in str(plan.get("workflow", [])),
        "reason": plan.get("reason", ""),
        "agent_note": sm.get("agent_note") or sm.get("reasoning", ""),
        "selection_reasons": plan.get("selection_reasons", []),
    }


def score_planning(expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：规划决策 (supported/executable/method) 是否与预期一致。"""
    checks = []

    # supported
    exp_sup = expected.get("supported")
    act_sup = actual["supported"]
    if exp_sup is not None and act_sup is not None:
        checks.append(("supported", exp_sup == act_sup,
                       f"exp:{exp_sup} act:{act_sup}"))

    # executable
    exp_exec = expected.get("executable")
    act_exec = actual["executable"]
    if exp_exec is not None and act_exec is not None:
        checks.append(("executable", exp_exec == act_exec,
                       f"exp:{exp_exec} act:{act_exec}"))

    # method: exact match OR external_suggestion counts as acceptable alternative
    exp_method = expected.get("method")
    act_method = actual["method"]
    if exp_method is not None:
        method_ok = (exp_method == act_method)
        # If platform returns external_suggestion and expected is None/unsupported,
        # that's actually a valid outcome (Agent did its job searching)
        if not method_ok and actual["is_external"] and exp_method is None:
            method_ok = True  # external suggestion is better than nothing
        checks.append(("method", method_ok,
                       f"exp:{exp_method} act:{act_method}"))

    if not checks:
        return 0.5, {"reason": "no_checks"}

    passed = sum(1 for _, ok, _ in checks if ok)
    score = round(passed / len(checks), 3)
    return score, {
        "checks": [{"label": l, "ok": o, "detail": d} for l, o, d in checks],
        "passed": passed, "total": len(checks),
    }


def score_qc_blocking(expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：QC 阻断——该拦的是否拦住，该放的是否放行。"""
    exp_should = expected.get("should_execute", True)
    exp_sup = expected.get("supported", True)

    act_sup = actual["supported"]
    act_exec = actual["executable"]
    is_external = actual["is_external"]

    checks = []

    if not exp_should:
        # Expected NOT to execute → check that platform blocked it
        correctly_blocked = (act_exec is False)
        checks.append(("blocked", correctly_blocked,
                       f"should NOT exec, actual exec={act_exec}"))
    elif not exp_sup:
        # Expected unsupported → check platform rejected
        correctly_rejected = (act_sup is False)
        checks.append(("rejected", correctly_rejected,
                       f"should be unsupported, actual supported={act_sup}"))
    else:
        # Expected supported & executable
        correctly_passed = (act_sup is True and act_exec is True)
        checks.append(("allowed", correctly_passed,
                       f"should exec, actual sup={act_sup} exec={act_exec}"))

    # If external suggestion was generated for an unsupported intent, that's a half-win
    if is_external and not exp_should:
        checks.append(("external_fallback", True,
                       "Agent provided external suggestion even though no registry match"))

    if not checks:
        return 0.5, {"reason": "no_checks"}

    passed = sum(1 for _, ok, _ in checks if ok)
    # If external fallback exists, don't penalize too hard
    score = round(passed / len(checks), 3)
    return score, {
        "checks": [{"label": l, "ok": o, "detail": d} for l, o, d in checks],
        "passed": passed, "total": len(checks),
    }


def score_intent_parsing(intent_def: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：意图解析——dataset/task_family/mode 是否正确。"""
    checks = []

    exp_dataset = intent_def.get("dataset", "unknown")
    act_dataset = actual["parsed_dataset"]
    if exp_dataset != "unknown":
        # Normalize: TDD→TDD, ToothFairy3→ToothFairy3, private2d→private2d etc.
        checks.append(("dataset", exp_dataset.lower() in act_dataset.lower() or act_dataset.lower() in exp_dataset.lower(),
                       f"exp:{exp_dataset} act:{act_dataset}"))

    exp_task = intent_def.get("task_family", "unknown")
    act_task = actual["parsed_task"]
    if exp_task != "unknown":
        checks.append(("task", exp_task.lower() in act_task.lower() or act_task.lower() in exp_task.lower(),
                       f"exp:{exp_task} act:{act_task}"))

    exp_mode = None
    # Infer expected mode from intent_zh / expected_behavior
    intent_zh = intent_def.get("intent_zh", "")
    expected_behavior = intent_def.get("expected_behavior", "")
    if "推理" in intent_zh or "inference" in expected_behavior.lower():
        exp_mode = "inference"
    elif "训练" in intent_zh or "train" in expected_behavior.lower():
        exp_mode = "train"
    elif "异常" in intent_zh or "anomaly" in expected_behavior.lower():
        exp_mode = "inference"

    if exp_mode:
        act_mode = actual["parsed_mode"]
        # Normalize: private_train → train, private_train → train
        act_mode_norm = act_mode.replace("private_", "")
        checks.append(("mode", exp_mode == act_mode_norm,
                       f"exp:{exp_mode} act:{act_mode}"))

    # Trap intents: correctly parsing "unknown" is the expected behavior → full marks
    if not checks:
        if intent_def.get("intent_category") in ("trap",):
            return 1.0, {"reason": "trap intent, correctly parsed as unknown"}
        return 0.5, {"reason": "no_parsing_checks"}

    passed = sum(1 for _, ok, _ in checks if ok)
    score = round(passed / len(checks), 3)
    return score, {
        "checks": [{"label": l, "ok": o, "detail": d} for l, o, d in checks],
        "passed": passed, "total": len(checks),
    }


def score_external_proposal(intent_def: Dict[str, Any], expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：外部方案提议——当 registry 无匹配时 Agent 是否给出了合理的建议。"""
    is_external = actual["is_external"]
    exp_method = expected.get("method")
    exp_supported = expected.get("supported", True)
    cat = intent_def.get("intent_category", "standard")

    # Cases where external proposal is expected:
    # 1. No registry method available but Agent should try to find something
    # 2. Method is explicitly "agent_external_suggestion"

    if is_external:
        # Agent did find an external suggestion → good
        if actual.get("agent_note") and len(actual["agent_note"]) > 20:
            return 1.0, {"detail": "Agent provided external proposal with reasoning"}
        else:
            return 0.7, {"detail": "Agent flagged as external but note is brief"}

    cat = intent_def.get("intent_category", "standard")

    if exp_method is None and not exp_supported:
        # Expected no method, actual no method.
        # For trap intents, skipping web search is correct → full marks
        if cat in ("trap",):
            return 1.0, {"detail": "Trap intent correctly rejected without web search"}
        # For boundary/ambiguous, web search could help but not required
        if actual.get("has_web_search"):
            return 0.6, {"detail": "Web search performed but no suggestion generated"}
        return 0.5, {"detail": "Expected no method and none provided (no web search)"}

    # Not an external-suggestion case → N/A, give full score
    return 1.0, {"detail": "N/A (registry match found)"}


def score_ambiguity(intent_def: Dict[str, Any], expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：歧义处理——模糊/不完整的意图是否被正确拒绝。"""
    cat = intent_def.get("intent_category", "standard")
    is_ambiguous = cat in ("ambiguous",)

    if is_ambiguous:
        act_exec = actual["executable"]
        correctly_rejected = (act_exec is False)
        score = 1.0 if correctly_rejected else 0.0
        return score, {"is_ambiguous": True, "correctly_rejected": correctly_rejected,
                       "detail": "correctly blocked" if correctly_rejected else "SHOULD have blocked"}
    else:
        return 1.0, {"is_ambiguous": False, "detail": "not ambiguous"}


def score_boundary(intent_def: Dict[str, Any], expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """评分：边界识别——超出平台范围的意图是否被正确拒绝。"""
    cat = intent_def.get("intent_category", "standard")
    exp_should = expected.get("should_execute", True)
    is_boundary = cat in ("boundary", "trap") or exp_should is False

    if is_boundary:
        # Should be blocked or flagged as external
        act_exec = actual["executable"]
        act_sup = actual["supported"]
        is_external = actual["is_external"]

        # If it's executable but via external suggestion, that's a half-win
        if is_external:
            score = 0.7
            detail = "Agent proposed external solution (acceptable for boundary case)"
        elif not act_exec:
            score = 1.0
            detail = "correctly blocked"
        else:
            score = 0.0
            detail = "SHOULD have blocked but executed"

        return score, {"is_boundary": True, "detail": detail}
    else:
        return 1.0, {"is_boundary": False, "detail": "not boundary"}


# ==============================================================================
# 总体评估
# ==============================================================================

def evaluate_one(intent_def: Dict[str, Any], plan: Dict[str, Any],
                 elapsed: float, stderr: str) -> Dict[str, Any]:
    """评估单条意图。"""
    intent_id = intent_def["id"]
    expected = intent_def.get("expected_platform_result") or {}
    actual = parse_actual(plan)

    dims = {}
    dims["planning"], d_planning = score_planning(expected, actual)
    dims["qc_blocking"], d_qc = score_qc_blocking(expected, actual)
    dims["intent_parsing"], d_parsing = score_intent_parsing(intent_def, actual)
    dims["external_proposal"], d_ext = score_external_proposal(intent_def, expected, actual)
    dims["ambiguity"], d_amb = score_ambiguity(intent_def, expected, actual)
    dims["boundary"], d_bnd = score_boundary(intent_def, expected, actual)

    total = sum(dims[d] * DIMENSION_WEIGHTS.get(d, 0) for d in dims)
    total = round(total, 3)

    return {
        "intent_id": intent_id,
        "intent_zh": intent_def.get("intent_zh", ""),
        "category": intent_def.get("intent_category", "standard"),
        "elapsed_s": round(elapsed, 1),
        "total_score": total,
        "dimension_scores": dims,
        "dimension_details": {
            "planning": d_planning,
            "qc_blocking": d_qc,
            "intent_parsing": d_parsing,
            "external_proposal": d_ext,
            "ambiguity": d_amb,
            "boundary": d_bnd,
        },
        "expected": expected,
        "actual": {
            "supported": actual["supported"],
            "executable": actual["executable"],
            "method": actual["method"],
            "is_external": actual["is_external"],
        },
        "stderr": stderr[:200] if stderr else "",
    }


# ==============================================================================
# 报告输出
# ==============================================================================

def print_report(results: List[Dict[str, Any]]) -> None:
    """打印评测报告。"""
    print(f"\n{'='*80}")
    print(f"DentalClaw Platform MVP 评测报告")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"{'='*80}")

    # 总分统计
    scores = [r["total_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for r in results if r["total_score"] >= 0.70)
    print(f"\n📊 总体: {len(results)} 条意图, 均分={avg_score:.1%}, 通过率={passed}/{len(results)}={passed/len(results)*100:.0f}%")

    # 维度均分
    print(f"\n📏 维度得分:")
    for dim, weight in DIMENSION_WEIGHTS.items():
        dim_scores = [r["dimension_scores"].get(dim, 0) for r in results]
        avg = sum(dim_scores) / len(dim_scores) if dim_scores else 0
        bar = "█" * int(avg * 30) + "░" * (30 - int(avg * 30))
        print(f"  {dim:20s} [{weight:.0%}] {bar} {avg:.3f}")

    # 逐条详情
    print(f"\n{'─'*80}")
    print(f"{'ID':20s} {'类别':10s} {'总分':>6} {'耗时':>6} {'plan':>5} {'qc':>5} {'parse':>5} {'ext':>5} {'amb':>5} {'bnd':>5} {'详情'}")
    print(f"{'─'*80}")
    for r in results:
        ds = r["dimension_scores"]
        icon = "✅" if r["total_score"] >= 0.70 else ("⚠️" if r["total_score"] >= 0.40 else "❌")
        issues = []
        if ds.get("planning", 1) < 0.7: issues.append("plan")
        if ds.get("qc_blocking", 1) < 0.7: issues.append("qc")
        if ds.get("intent_parsing", 1) < 0.7: issues.append("parse")
        print(f"  {icon} {r['intent_id']:16s} {r['category']:10s} {r['total_score']:>5.0%} {r['elapsed_s']:>4.0f}s "
              f"{ds.get('planning', 0):>4.0%} {ds.get('qc_blocking', 0):>4.0%} {ds.get('intent_parsing', 0):>4.0%} "
              f"{ds.get('external_proposal', 0):>4.0%} {ds.get('ambiguity', 0):>4.0%} {ds.get('boundary', 0):>4.0%} "
              f"{','.join(issues) if issues else '✓'}")

    # 方法论统计
    print(f"\n{'─'*80}")
    print(f"方法论分布:")
    method_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        m = r["actual"]["method"] or "NO_METHOD"
        if m not in method_stats:
            method_stats[m] = {"count": 0, "scores": [], "is_external": r["actual"]["is_external"]}
        method_stats[m]["count"] += 1
        method_stats[m]["scores"].append(r["total_score"])

    for m, s in sorted(method_stats.items()):
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        tag = " [external]" if s["is_external"] else ""
        print(f"  {m:45s} x{s['count']}  avg={avg:.1%}{tag}")

    print(f"\n{'='*80}\n")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Platform MVP Eval")
    parser.add_argument("--all", action="store_true", help="Run all 25 intents")
    parser.add_argument("--limit", type=int, default=0, help="Run first N intents only")
    parser.add_argument("--dry-run", action="store_true", help="Only score existing plans, don't re-run")
    parser.add_argument("--run-dir", help="Directory with platform_plan.json for dry-run")
    parser.add_argument("--json-report", help="Output JSON report")
    parser.add_argument("--timeout", type=int, default=120, help="Per-intent timeout in seconds")
    parser.add_argument("--intent-ids", nargs="*", help="Specific intent IDs to run")
    parser.add_argument("--intents-file", default=None,
                        help="Path to intents jsonl (default: benchmark_intents/intents.platform_mvp.jsonl)")

    args = parser.parse_args()

    # Allow overriding the intents file
    global INTENTS_FILE
    if args.intents_file:
        INTENTS_FILE = Path(args.intents_file)

    # Load intents
    all_intents = load_intents()
    if args.intent_ids:
        all_intents = [i for i in all_intents if i["id"] in args.intent_ids]
    if args.limit and args.limit > 0:
        all_intents = all_intents[:args.limit]

    print(f"加载 {len(all_intents)} 条意图")

    results = []

    if args.dry_run:
        if args.run_dir:
            # Evaluate a single run directory
            plan_path = Path(args.run_dir) / "platform_plan.json"
            if not plan_path.exists():
                print(f"错误: 找不到 {plan_path}")
                sys.exit(1)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            # Find matching intent
            for intent in all_intents:
                results.append(evaluate_one(intent, plan, 0, ""))
        else:
            print("dry-run 需要 --run-dir")
            sys.exit(1)
    else:
        # Run each intent through the platform MVP pipeline
        import tempfile, shutil
        tmp_root = Path(tempfile.gettempdir()) / "eval_platform_mvp"
        tmp_root.mkdir(parents=True, exist_ok=True)

        for i, intent in enumerate(all_intents):
            intent_id = intent["id"]
            intent_zh = intent.get("intent_zh", "")
            cat = intent.get("intent_category", "standard")

            print(f"\n[{i+1}/{len(all_intents)}] {intent_id} ({cat})...", end=" ", flush=True)

            run_dir = tmp_root / intent_id
            plan, elapsed, stderr = run_intent(intent_zh, run_dir, timeout=args.timeout)

            if plan is None:
                # Failed to get a plan
                results.append({
                    "intent_id": intent_id,
                    "intent_zh": intent_zh,
                    "category": cat,
                    "elapsed_s": round(elapsed, 1),
                    "total_score": 0.0,
                    "dimension_scores": {d: 0.0 for d in DIMENSION_WEIGHTS},
                    "dimension_details": {},
                    "expected": intent.get("expected_platform_result", {}),
                    "actual": {"supported": None, "executable": None, "method": None, "is_external": False},
                    "stderr": stderr[:200],
                    "error": stderr,
                })
                print(f"❌ FAILED ({elapsed:.0f}s): {stderr[:80]}")
            else:
                result = evaluate_one(intent, plan, elapsed, stderr)
                results.append(result)
                icon = "✅" if result["total_score"] >= 0.70 else ("⚠️" if result["total_score"] >= 0.40 else "❌")
                print(f"{icon} score={result['total_score']:.0%} ({elapsed:.0f}s) "
                      f"s={result['actual']['supported']} e={result['actual']['executable']} "
                      f"m={result['actual']['method']}")

    # Print report
    print_report(results)

    # JSON output
    if args.json_report:
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
        }
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
