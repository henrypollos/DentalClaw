# DentalClaw Benchmark Run: DCI-TF3-SEG3D-015

- Prompt: 对 ToothFairy3 images-only CBCT 子集运行已有模型推理，不做监督训练。
- Dataset: ToothFairy3 (/data/data2/yiyang/DentalClaw/agents/data_curator/reports/cbct_qc/toothfairy3F_images_only_subset/dataset)
- Task family: segmentation_3d
- Intent category: standard
- Expected behavior: execute_end_to_end
- Terminal status: completed
- Exception handling: completed
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- probe_dataset
- validate_dataset
- run_dataset_qc
- run_inference

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=ToothFairy3、任务族=segmentation_3d、任务=images_only_segmentation_inference、意图类别=standard。
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
10. data.voxel_spacing_check
11. data.fov_consistency_check
12. experiment.inference
13. report.visualization_builder
14. shared.audit_log_finalize
