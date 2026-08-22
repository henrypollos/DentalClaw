#!/usr/bin/env python3
"""
把 benchmark_intents/intents.jsonl（30 条全集）映射为平台 MVP 评测格式
→ benchmark_intents/intents.platform_mvp_30.jsonl

映射原则（与 2026-07 已评测的 11 条口径一致）：
- standard：注册表有匹配路由 → {supported:true, executable:true, method:<route>, should_execute:true}
- boundary（无路由但合理任务）→ {supported:true, executable:false, method:"agent_external_suggestion", should_execute:false}
- ambiguous → {supported:true, executable:false, method:"platform_clarification", should_execute:false}
- trap（应拒绝）→ {supported:false, executable:false, method:null, should_execute:false}
- QC 条件执行陷阱（012/014/024：训练路由存在、阻断发生在执行级）→ 决策层正确行为
  是"路由 + QC 门"，故按 standard 路由编码，并在 note 中说明；执行级阻断不在本协议测量范围。
- 类别以平台语义为准（30 条原类别保留在 orig_category 字段）。
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent / "intents.jsonl"
DST = Path(__file__).resolve().parent / "intents.platform_mvp_30.jsonl"

TD = "tdd_2d_segmentation_infer_report"
TF3 = "toothfairy3_3d_segmentation_infer_or_train"
PRV = "private_2d_segmentation_train"


def std(m):
    return {"supported": True, "executable": True, "method": m, "should_execute": True}


def ext():
    return {"supported": True, "executable": False, "method": "agent_external_suggestion", "should_execute": False}


def clr():
    return {"supported": True, "executable": False, "method": "platform_clarification", "should_execute": False}


def rej():
    return {"supported": False, "executable": False, "method": None, "should_execute": False}


# id → (平台类别, 期望结果, 说明)
# 注意：以下映射已按平台实际决策语义（2026-08-18 首跑）修正：
#  - 文本无任务关键词（分割/检测/分类等）→ 平台 Step 0 保守解析 → 澄清（clr），task_family 置 unknown
#  - 分类无资产 → 平台按 scope_policy 拒绝（rej）
#  - 格式整理≠训练 → 外部提案（ext）
MAP = {
    "DCI-TDD-SEG2D-001": ("boundary", ext(), "平台无 TDD 训练路由（注册表仅推理）→ 外部提案"),
    "DCI-TDD-SEG2D-002": ("boundary", ext(), "32 类训练无注册表路由 → 外部提案"),
    "DCI-TDD-SEG2D-003": ("ambiguous", clr(), "任务模糊（区域分割模型）→ 澄清"),
    "DCI-TDD-SEG2D-004": ("standard", std(TD), "已有二值模型推理+指标+叠加图 → TDD 推理路由"),
    "DCI-TDD-SEG2D-005": ("ambiguous", clr(), "未指明病例与报告类型 → 澄清"),
    "DCI-TDD-DET-006": ("boundary", ext(), "COCO 检测数据集整理无路由 → 外部提案"),
    "DCI-TDD-DET-007": ("boundary", ext(), "检测训练无路由 → 外部提案"),
    "DCI-TDD-DET-008": ("trap", rej(), "bbox QC 无路由，应拒绝而非执行"),
    "DCI-TDD-CLS-009": ("boundary", rej(), "分类无资产，scope_policy=reject_or_explain → 拒绝为正确行为"),
    "DCI-TDD-CLS-010": ("trap", rej(), "分类+质量标签核验无路由，应拒绝"),
    "DCI-TF3-SEG3D-011": ("standard", std(TF3), "TF3 训练 → TF3 路由"),
    "DCI-TF3-SEG3D-012": ("standard", std(TF3), "QC 条件执行陷阱：决策层正确行为=路由+QC 门；执行级阻断不在本协议测量范围"),
    "DCI-TF3-SEG3D-013": ("standard", std(TF3), "TF3 推理 → TF3 路由"),
    "DCI-TF3-SEG3D-014": ("trap", clr(), "QC 优先陷阱（先检查标签值），文本无任务关键词 → Step 0 保守澄清即正确阻断行为"),
    "DCI-TF3-SEG3D-015": ("ambiguous", clr(), "已有模型推理但未指明任务类型 → 澄清（保守解析）"),
    "DCI-TF3-DET-016": ("boundary", ext(), "3D bbox 导出无路由 → 外部提案"),
    "DCI-TF3-DET-017": ("boundary", ext(), "3D 检测训练无路由 → 外部提案"),
    "DCI-TF3-CLS-018": ("ambiguous", clr(), "质量分类任务模糊 → 澄清"),
    "DCI-TF3-CLS-019": ("ambiguous", clr(), "子集分类实验模糊 → 澄清"),
    "DCI-TF3-CLS-020": ("boundary", ext(), "usable/needs_review 分类无路由 → 外部提案"),
    "DCI-P2D-SEG2D-021": ("standard", std(PRV), "私有训练 → private 路由"),
    "DCI-P2D-SEG2D-022": ("boundary", ext(), "私有推理无独立路由（仅 private_train）→ 外部提案"),
    "DCI-P2D-SEG2D-023": ("ambiguous", clr(), "报告类型未指明 → 澄清"),
    "DCI-P2D-SEG2D-024": ("trap", clr(), "QC 优先陷阱（先检查 split 泄漏），文本无任务关键词 → 保守澄清即正确阻断行为"),
    "DCI-P2D-SEG2D-025": ("boundary", ext(), "格式整理≠训练，无独立导出路由 → 外部提案"),
    "DCI-P2D-DET-026": ("boundary", ext(), "私有检测训练无路由 → 外部提案"),
    "DCI-P2D-DET-027": ("boundary", ext(), "mask→bbox 检测无路由 → 外部提案"),
    "DCI-P2D-DET-028": ("trap", rej(), "缺框标注应停止 → 拒绝"),
    "DCI-P2D-CLS-029": ("boundary", rej(), "分类无资产，scope_policy=reject_or_explain → 拒绝为正确行为"),
    "DCI-P2D-CLS-030": ("ambiguous", clr(), "标注可用性判断模糊 → 澄清"),
}

# 文本不含任务关键词 → 平台解析为 unknown，评测文件 task_family 同步置 unknown
# （这正是歧义/QC 优先意图的设计语义：任务未在文本中声明）
TASK_FAMILY_OVERRIDE = {
    "DCI-TDD-SEG2D-005": "unknown",
    "DCI-TF3-SEG3D-014": "unknown",
    "DCI-TF3-SEG3D-015": "unknown",
    "DCI-P2D-SEG2D-023": "unknown",
    "DCI-P2D-SEG2D-024": "unknown",
    "DCI-P2D-CLS-030": "unknown",
}

KEEP_FIELDS = ["id", "intent_id", "dataset", "task_family", "task_type", "task",
               "intent_zh", "prompt", "expected_behavior", "expected_terminal_status",
               "success_criteria", "trap_data", "trap_description"]


def main():
    with open(SRC, encoding="utf-8") as f:
        src = [json.loads(l) for l in f if l.strip()]

    assert len(src) == 30, f"intents.jsonl 应有 30 条，实际 {len(src)}"
    src_ids = {d["id"] for d in src}
    missing = set(MAP) - src_ids
    unmapped = src_ids - set(MAP)
    assert not missing, f"MAP 中有不在源文件里的 id: {missing}"
    assert not unmapped, f"源文件中有未映射的 id: {unmapped}"

    out = []
    for d in src:
        cat, expected, note = MAP[d["id"]]
        rec = {k: d[k] for k in KEEP_FIELDS if k in d}
        rec["intent_category"] = cat
        rec["orig_category"] = d.get("intent_category", "")
        rec["expected_platform_result"] = expected
        rec["note"] = note
        if d["id"] in TASK_FAMILY_OVERRIDE:
            rec["task_family"] = TASK_FAMILY_OVERRIDE[d["id"]]
            rec["orig_task_family"] = d.get("task_family", "")
        out.append(rec)

    with open(DST, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cats = {}
    for rec in out:
        cats[rec["intent_category"]] = cats.get(rec["intent_category"], 0) + 1
    print(f"写出 {len(out)} 条 → {DST}")
    print("平台类别分布:", cats)


if __name__ == "__main__":
    main()
