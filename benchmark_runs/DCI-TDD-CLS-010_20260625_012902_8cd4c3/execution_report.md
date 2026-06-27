# DentalClaw Benchmark Run: DCI-TDD-CLS-010

- Prompt: 用 TDD 的图像质量标签训练可用/需复核分类器；训练前先核验质量标签清单，标签不完整就停止。
- Dataset: TDD (/data/data2/yiyang/DentalClaw/data/TDD)
- Task family: classification
- Intent category: trap
- Expected behavior: warn_and_stop
- Terminal status: blocked_missing_curated_quality_labels
- Exception handling: warn_and_stop
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset
- run_dataset_qc

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=TDD、任务族=classification、任务=image_quality_classification、意图类别=trap。
- MainAgent.route_to_data_curator: 任务需要数据探查、标注/QC检查或导出，因此先路由到 Data Curator，避免未经验证直接训练。
- MainAgent.warn_and_stop: 任务表述清晰，但数据/QC 检查暴露出已知缺陷；为了避免污染实验结果，停止训练或推理并要求人工确认。
- MainAgent.stop_without_training: 陷阱意图的数据缺陷已被验证，必须阻断训练/推理并保留人工复核入口。

## Reference Workflow
1. coordinator.intent_parse
2. coordinator.dataset_registry_lookup
3. coordinator.capability_check
4. data.dataset_import
5. data.format_parse
6. data.case_structuring
7. data.metadata_parse
8. data.data_integrity_check
9. data.missing_annotation_check
10. data.split_verification
11. data.issue_warning
12. coordinator.stop_without_training
13. shared.audit_log_finalize
