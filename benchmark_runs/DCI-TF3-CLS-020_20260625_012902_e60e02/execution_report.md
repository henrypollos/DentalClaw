# DentalClaw Benchmark Run: DCI-TF3-CLS-020

- Prompt: 将 ToothFairy3 病例分成 usable 和 needs_manual_review，并输出病例级分类结果。
- Dataset: ToothFairy3 (/data/data2/yiyang/JoD/ToothFairy3_LPS)
- Task family: classification
- Intent category: standard
- Expected behavior: execute_end_to_end
- Terminal status: qc_only_completed
- Exception handling: completed
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset
- run_dataset_qc

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=ToothFairy3、任务族=classification、任务=usable_vs_manual_review_triage、意图类别=standard。
- MainAgent.route_to_data_curator: 任务需要数据探查、标注/QC检查或导出，因此先路由到 Data Curator，避免未经验证直接训练。
- MainAgent.route_to_experimentation: 数据治理和必要 QC 已完成，标准意图可进入实验 Agent 生成训练/推理配置并执行 dry-run。

## Reference Workflow
1. coordinator.intent_parse
2. coordinator.dataset_registry_lookup
3. coordinator.capability_check
4. data.dataset_import
5. data.format_parse
6. data.case_structuring
7. data.metadata_parse
8. data.data_integrity_check
9. data.voxel_spacing_check
10. data.fov_consistency_check
11. data.class_distribution_check
12. shared.audit_log_finalize
