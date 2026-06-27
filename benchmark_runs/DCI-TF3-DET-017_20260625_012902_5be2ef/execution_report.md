# DentalClaw Benchmark Run: DCI-TF3-DET-017

- Prompt: 用 ToothFairy3 训练 3D 牙齿检测模型，并报告检测 mAP。
- Dataset: ToothFairy3 (/data/data2/yiyang/JoD/ToothFairy3_LPS)
- Task family: detection
- Intent category: boundary
- Expected behavior: reject_or_explain
- Terminal status: unsupported_backend_after_qc
- Exception handling: reject_or_explain
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset
- run_dataset_qc

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=ToothFairy3、任务族=detection、任务=cbct_detector_train、意图类别=boundary。
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
10. data.voxel_spacing_check
11. data.fov_consistency_check
12. data.task_data_compatibility_check
13. data.missing_annotation_check
14. data.split_verification
15. coordinator.unsupported_capability_response
16. coordinator.stop_without_training
17. shared.audit_log_finalize
