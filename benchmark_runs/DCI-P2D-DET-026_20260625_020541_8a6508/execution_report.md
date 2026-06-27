# DentalClaw Benchmark Run: DCI-P2D-DET-026

- Prompt: 用私有 2D 数据训练牙齿检测模型。
- Dataset: Private2D (/data/data2/yiyang/DentalClaw/data/2d)
- Task family: detection
- Intent category: boundary
- Expected behavior: reject_or_explain
- Terminal status: unsupported_missing_detection_annotations
- Exception handling: reject_or_explain
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=Private2D、任务族=detection、任务=private_2d_detector_train_missing_bbox、意图类别=boundary。
- MainAgent.route_to_data_curator: 任务需要数据探查、标注/QC检查或导出，因此先路由到 Data Curator，避免未经验证直接训练。
- MainAgent.unsupported_capability_response: 请求本身清晰合理，但当前平台缺少对应训练/导出/评估后端；输出结构化能力边界说明，而不是伪造结果。
- MainAgent.stop_without_training: 边界意图不得回退到分割训练、临时脚本或虚构指标。

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
10. data.task_data_compatibility_check
11. data.missing_annotation_check
12. data.split_verification
13. data.patient_leakage_check
14. coordinator.unsupported_capability_response
15. coordinator.stop_without_training
16. shared.audit_log_finalize
