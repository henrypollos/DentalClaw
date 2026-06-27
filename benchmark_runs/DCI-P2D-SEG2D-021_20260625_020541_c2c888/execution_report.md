# DentalClaw Benchmark Run: DCI-P2D-SEG2D-021

- Prompt: 用私有 2D 数据 imagesTr/labelsTr 训练牙齿分割模型，并用 imagesVal/labelsVal 验证。
- Dataset: Private2D (/data/data2/yiyang/DentalClaw/data/2d)
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
- validate_dataset
- run_dataset_qc
- run_training

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=Private2D、任务族=segmentation_2d、任务=private_2d_segmentation_train、意图类别=standard。
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
10. data.image_annotation_pairing_check
11. data.shape_consistency_check
12. data.label_normalization
13. data.split_verification
14. data.patient_leakage_check
15. experiment.task_schema_builder
16. experiment.experiment_config_generator
17. experiment.model_training
18. experiment.model_selection
19. shared.audit_log_finalize
