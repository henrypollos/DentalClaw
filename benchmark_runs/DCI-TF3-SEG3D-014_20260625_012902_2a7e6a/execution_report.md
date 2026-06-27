# DentalClaw Benchmark Run: DCI-TF3-SEG3D-014

- Prompt: 训练 ToothFairy3 前先严格检查标签值是否合法；若有非法标签值就阻断训练并列出病例。
- Dataset: ToothFairy3 (/data/data2/yiyang/JoD/ToothFairy3_LPS)
- Task family: segmentation_3d
- Intent category: trap
- Expected behavior: warn_and_stop
- Terminal status: blocked_invalid_label_values
- Exception handling: warn_and_stop
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset
- run_dataset_qc

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=ToothFairy3、任务族=segmentation_3d、任务=label_schema_validation_blocking、意图类别=trap。
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
9. data.annotation_schema_check
10. data.voxel_spacing_check
11. data.fov_consistency_check
12. data.split_verification
13. data.issue_warning
14. coordinator.stop_without_training
15. shared.audit_log_finalize
