# DentalClaw Benchmark Run: DCI-TDD-SEG2D-001

- Prompt: 用 TDD 全景片训练一个默认牙齿二值分割模型，并保留 10% 测试集。
- Dataset: TDD (/data/data2/yiyang/DentalClaw/data/TDD)
- Task family: segmentation_2d
- Intent category: standard
- Expected behavior: execute_end_to_end
- Terminal status: completed_or_running_detached
- Exception handling: completed
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- export_tdd_to_nnunet
- validate_dataset
- run_dataset_qc
- run_training

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=TDD、任务族=segmentation_2d、任务=teeth_binary_train、意图类别=standard。
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
9. data.annotation_schema_check
10. data.label_normalization
11. data.split_verification
12. data.patient_leakage_check
13. experiment.task_schema_builder
14. experiment.experiment_config_generator
15. experiment.model_training
16. experiment.model_selection
17. shared.audit_log_finalize
