"""
DentalClaw Platform MVP 国际化 (i18n) 模块。

用法:
    from platform_mvp.i18n import i18n, detect_language

    lang = detect_language(intent)  # 自动检测: "zh" or "en"
    msg = i18n(lang, "out_of_scope_reject")
    print(msg)
"""

import re
from typing import Optional, Union

# ── 语言检测 ────────────────────────────────────────────────────────────────

def _has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符。"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def detect_language(intent_text: str, override: Optional[str] = None) -> str:
    """
    检测语言: "zh" 或 "en"。
    
    检测规则: 含中文字符 → zh, 否则 → en。
    override 参数可强制指定语言。
    """
    if override and override in ("zh", "en", "auto"):
        if override != "auto":
            return override
    return "zh" if _has_chinese(intent_text) else "en"


# ── 消息字典 ────────────────────────────────────────────────────────────────

_MESSAGES = {
    # ── 平台定位 / 超出范围拒绝 ──
    "out_of_scope_reject": {
        "zh": (
            "DentalClaw 是专注牙科医学影像的 CV 平台，当前仅支持牙科全景片/CBCT/私有牙科数据的"
            "分割、检测、分类、超分辨率及异常检测任务。"
            "您提出的请求暂不在平台覆盖范围内，建议提供明确的牙科影像分析需求（如「用 TDD 全景片做牙齿分割」）。"
        ),
        "en": (
            "DentalClaw is a CV platform specialized in dental medical imaging. "
            "It currently supports segmentation, detection, classification, super-resolution, "
            "and anomaly detection tasks on dental panoramic/CBCT/private dental data. "
            "Your request is outside the current scope. Please provide a clear dental imaging request "
            '(e.g., "Perform tooth segmentation on TDD panoramic images").'
        ),
    },

    # ── 追问澄清 (部分信息缺失) ──
    "clarification_missing": {
        "zh": "未能确定: {fields}。DentalClaw 是牙科 CV 专用平台，请补充上述信息以便为您匹配最佳方案。",
        "en": "Could not determine: {fields}. DentalClaw is a dental CV platform. Please provide the missing information so we can find the best method for you.",
    },

    "clarification_field_dataset": {
        "zh": "dataset (可选: TDD/全景, ToothFairy3/CBCT, Private2D/私有/本院)",
        "en": "dataset (available: TDD/Panoramic, ToothFairy3/CBCT, Private2D/Custom)",
    },
    "clarification_field_task": {
        "zh": "task (可选: 分割/segmentation, 检测/detection, 分类/classification, 超分/super-resolution, 异常/anomaly)",
        "en": "task (available: segmentation, detection, classification, super-resolution, anomaly)",
    },

    # ── 可用数据集/任务列表 (展示用) ──
    "available_datasets": {
        "zh": ["TDD (全景片)", "ToothFairy3 (3D CBCT)", "Private2D (私有数据)"],
        "en": ["TDD (Panoramic)", "ToothFairy3 (3D CBCT)", "Private2D (Custom Data)"],
    },
    "available_tasks": {
        "zh": ["segmentation (分割)", "detection (检测)", "classification (分类)",
               "super_resolution (超分)", "anomaly_detection (异常检测)"],
        "en": ["segmentation", "detection", "classification",
               "super_resolution", "anomaly_detection"],
    },

    # ── Agent 决策系统提示 ──
    "agent_system_prompt": {
        "zh": (
            "你是 DentalClaw 牙科 CV 平台的智能决策助手。\n"
            "## 能力范围\n"
            "- 数据集: TDD(全景片), ToothFairy3(3D CBCT), Private2D(私有)\n"
            "- 任务: 分割, 检测, 分类, 超分, 异常检测\n"
            "- 模式: 推理(inference), 训练(train)\n\n"
            "## 决策规则\n"
            "1. 如果方法表有匹配且 status=executable → decision='registry_match'\n"
            "2. 如果方法表匹配但 status=planned_adapter → decision='planned_adapter'\n"
            "3. 如果无匹配但 web 搜索和代码库有可用的外部方案 → decision='external_proposal'\n"
            "4. 否则 → decision='unsupported'\n\n"
            "## 输出格式\n"
            '请返回 JSON: {"decision": "...", "confidence": 0.0-1.0, '
            '"reasoning": "...", "entrypoint": "...", "risks": [...], '
            '"missing_prerequisites": [...]}'
        ),
        "en": (
            "You are the intelligent decision assistant for DentalClaw, a dental CV platform.\n"
            "## Capabilities\n"
            "- Datasets: TDD (panoramic), ToothFairy3 (3D CBCT), Private2D (custom)\n"
            "- Tasks: segmentation, detection, classification, super-resolution, anomaly detection\n"
            "- Modes: inference, training\n\n"
            "## Decision Rules\n"
            "1. If registry has a match with status=executable → decision='registry_match'\n"
            "2. If registry matches but status=planned_adapter → decision='planned_adapter'\n"
            "3. If no registry match but web search + codebase has viable external → decision='external_proposal'\n"
            "4. Otherwise → decision='unsupported'\n\n"
            "## Output Format\n"
            'Return JSON: {"decision": "...", "confidence": 0.0-1.0, '
            '"reasoning": "...", "entrypoint": "...", "risks": [...], '
            '"missing_prerequisites": [...]}'
        ),
    },

    # ── Agent 决策用户消息模板 (完整版，包含代码库资产 + web 结果 + registry + 规则) ──
    "agent_user_prompt_full": {
        "zh": (
            "你是 DentalClaw 平台决策 Agent。\n"
            "任务：综合 web 搜索结果、方法表匹配和代码库资产，做出最佳方案决策。\n\n"
            "=== 代码库资产 ===\n"
            "- nnU-Net v2 (nnunetv2): 二值分割训练+推理\n"
            "- TDD 2D 二值分割: mvp_fullflow/run_mvp_fullflow.py\n"
            "- 自动训练: agents/experimentation/skills/tooth_autotrain_nnunet/\n"
            "- 超分辨率: platform_mvp/run_super_resolution_mvp.py\n"
            "- 异常检测: platform_mvp/run_anomaly_detection_mvp.py (ResNet+IsolationForest)\n"
            "- YOLO 检测: run_tdd_detection_traced.py\n"
            "- TTA + Ensemble: platform_mvp/tta_ensemble.py\n"
            "- TDD 数据集: Dataset501_TDDTeethBinary2D (全景片, 二值 mask)\n"
            "- ToothFairy3: 3D CBCT 数据集\n\n"
            "=== WEB 搜索结果 ===\n{web_text}\n\n"
            "=== 方法表状态 ===\n{registry_text}\n\n"
            "=== 用户请求 ===\n{intent}\n\n"
            "=== 解析意图 ===\n"
            "task={task}, dataset={dataset}, modality={modality}, mode={mode}\n\n"
            "=== 决策规则 ===\n"
            "1. 如果方法表有 EXECUTABLE 匹配且与用户模式相符 → decision='use_registry'\n"
            "2. 如果方法表匹配但模式不符, 或 web 搜索发现更好的方案 → decision='external_proposal'\n"
            "3. 如果方法表无匹配, 用 web 搜索+代码库知识提议 → decision='external_proposal'\n"
            "4. 仅当请求确实不可能完成时拒绝 → decision='reject', confidence=0\n\n"
            "只返回 JSON:\n"
            '{{"decision": "use_registry|external_proposal|reject", '
            '"proposed_entrypoint": "...", "pip_package": "...", '
            '"confidence": 0.0, "reasoning": "..."}}'
        ),
        "en": (
            "You are the DentalClaw platform decision agent.\n"
            "Your job: review web search results AND the registry match, then decide the BEST approach.\n\n"
            "=== CODEBASE ASSETS ===\n"
            "- nnU-Net v2 (nnunetv2): binary segmentation training + inference\n"
            "- TDD 2D binary seg: mvp_fullflow/run_mvp_fullflow.py\n"
            "- Auto-train: agents/experimentation/skills/tooth_autotrain_nnunet/\n"
            "- Super-resolution: platform_mvp/run_super_resolution_mvp.py\n"
            "- Anomaly detection: platform_mvp/run_anomaly_detection_mvp.py (ResNet+IsolationForest)\n"
            "- YOLO detection: run_tdd_detection_traced.py\n"
            "- TTA + Ensemble: platform_mvp/tta_ensemble.py\n"
            "- TDD dataset: Dataset501_TDDTeethBinary2D (panoramic X-ray, binary masks)\n"
            "- ToothFairy3: 3D CBCT dataset\n\n"
            "=== WEB SEARCH RESULTS ===\n{web_text}\n\n"
            "=== REGISTRY STATUS ===\n{registry_text}\n\n"
            "=== USER REQUEST ===\n{intent}\n\n"
            "=== PARSED INTENT ===\n"
            "task={task}, dataset={dataset}, modality={modality}, mode={mode}\n\n"
            "=== DECISION RULES ===\n"
            "1. If registry has an EXECUTABLE match that fits the user's mode, prefer it (decision=use_registry).\n"
            "2. If registry match exists but mode mismatch, OR web search found a clearly better approach, "
            "propose external (decision=external_proposal).\n"
            "3. If no registry match, use web search + codebase knowledge to propose (decision=external_proposal).\n"
            "4. Only reject if the request is truly impossible (decision=reject, confidence=0).\n\n"
            "Return ONLY valid JSON:\n"
            '{{"decision": "use_registry|external_proposal|reject", '
            '"proposed_entrypoint": "...", "pip_package": "...", '
            '"confidence": 0.0, "reasoning": "..."}}'
        ),
    },

    # ── Agent 决策用户消息模板 (简版，仅含基本信息) ──
    "agent_user_prompt": {
        "zh": (
            "用户意图: {intent}\n"
            "解析: dataset={dataset}, task={task_family}, modality={modality}, mode={mode}\n"
            "方法表匹配: {registry_match}\n"
            "方法表结果: {registry_reasons}\n"
            "Web 搜索结果数: {search_count}\n"
            "Web 搜索摘要: {search_summary}\n\n"
            "请综合以上信息做出决策。"
        ),
        "en": (
            "User intent: {intent}\n"
            "Parsed: dataset={dataset}, task={task_family}, modality={modality}, mode={mode}\n"
            "Registry match: {registry_match}\n"
            "Registry reasons: {registry_reasons}\n"
            "Web search results: {search_count}\n"
            "Web search summary: {search_summary}\n\n"
            "Please make a decision based on the above information."
        ),
    },

    # ── 执行 / 状态消息 ──
    "executing_route": {
        "zh": "▶ 执行路线: {method_id}",
        "en": "▶ Executing route: {method_id}",
    },
    "execution_complete": {
        "zh": "✅ 执行完成: {method_id}",
        "en": "✅ Execution complete: {method_id}",
    },
    "execution_external_suggestion": {
        "zh": (
            "⚠️ Agent 提出了外部方案（置信度 {confidence:.0%}），需人工确认后方可执行。\n"
            "方案入口: {entrypoint}\n"
            "推理: {reasoning}"
        ),
        "en": (
            "⚠️ Agent proposed an external solution (confidence {confidence:.0%}) — manual review required before execution.\n"
            "Entrypoint: {entrypoint}\n"
            "Reasoning: {reasoning}"
        ),
    },
    "execution_planned_adapter": {
        "zh": "⚠️ 方法 {method_id} 已登记但尚未实现可执行 adapter。",
        "en": "⚠️ Method {method_id} is registered but the executable adapter is not yet implemented.",
    },
    "execution_unsupported": {
        "zh": "❌ 当前平台不支持该请求。原因: {reason}",
        "en": "❌ This request is not currently supported. Reason: {reason}",
    },

    # ── 计划摘要 ──
    "plan_summary_executable": {
        "zh": "方法 {method_id}: {display_name} (可执行)",
        "en": "Method {method_id}: {display_name} (executable)",
    },
    "plan_summary_external": {
        "zh": "Agent 外部提议: {entrypoint} (需人工确认)",
        "en": "Agent external proposal: {entrypoint} (review required)",
    },
    "plan_summary_clarification": {
        "zh": "需要用户补充信息",
        "en": "Additional information needed",
    },
    "plan_summary_unsupported": {
        "zh": "不支持",
        "en": "Not supported",
    },

    # ── 意图解析 ──
    "intent_parse_summary": {
        "zh": "dataset={dataset}, task={task_family}, mode={mode}",
        "en": "dataset={dataset}, task={task_family}, mode={mode}",
    },

    # ── Web 搜索 ──
    "web_search_summary": {
        "zh": "找到 {count} 条结果",
        "en": "{count} results found",
    },
    "web_search_none": {
        "zh": "未找到结果",
        "en": "No results found",
    },

    # ── 方法表查询 ──
    "registry_lookup_match": {
        "zh": "匹配: {method_id} (status={status})",
        "en": "Match: {method_id} (status={status})",
    },
    "registry_lookup_none": {
        "zh": "无匹配",
        "en": "No match",
    },

    # ── 外部方案状态标签 ──
    "external_suggestion_status": {
        "zh": "外部提议",
        "en": "External Proposal",
    },
    "awaiting_clarification_status": {
        "zh": "等待补充信息",
        "en": "Awaiting Clarification",
    },
    "unsupported_status": {
        "zh": "不支持",
        "en": "Unsupported",
    },
}


def _(lang: str, key: str, **kwargs) -> str:
    """
    获取指定语言的消息文本。
    
    Args:
        lang: "zh" 或 "en"
        key: 消息键
        **kwargs: 格式化参数 (用于 {placeholder})
    
    Returns:
        格式化后的消息文本。如果键或语言不存在，返回 key 本身。
    """
    entry = _MESSAGES.get(key)
    if not entry:
        return key
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass  # missing format args, return raw
    return text


def i18n(lang: str) -> callable:
    """
    返回一个绑定了语言的翻译函数。
    
    用法:
        _t = i18n("zh")
        print(_t("out_of_scope_reject"))
        print(_t("execution_complete", method_id="tdd_2d_segmentation_infer_report"))
    """
    return lambda key, **kwargs: _(lang, key, **kwargs)
