# DentalClaw Benchmark Run: DCI-TDD-DET-006

- Prompt: 把 TDD 的 teeth_bbox.json 整理成 COCO 牙齿检测数据集，供检测模型训练使用。
- Dataset: TDD (/data/data2/yiyang/DentalClaw/data/TDD)
- Task family: detection
- Intent category: standard
- Expected behavior: execute_end_to_end
- Terminal status: completed
- Exception handling: completed
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- export_tdd_to_nnunet
- validate_dataset

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=TDD、任务族=detection、任务=export_teeth_detection_coco、意图类别=standard。
- MainAgent.route_to_data_curator: 任务需要数据探查、标注/QC检查或导出，因此先路由到 Data Curator，避免未经验证直接训练。

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
11. data.bbox_validation
12. data.split_verification
13. report.structured_report_generation
14. shared.audit_log_finalize
