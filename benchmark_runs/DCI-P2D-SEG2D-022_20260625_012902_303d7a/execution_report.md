# DentalClaw Benchmark Run: DCI-P2D-SEG2D-022

- Prompt: 对私有 2D imagesVal 做分割推理，并用 labelsVal 计算指标。
- Dataset: Private2D (/data/data2/yiyang/DentalClaw/data/2d)
- Task family: segmentation_2d
- Intent category: standard
- Expected behavior: execute_end_to_end
- Terminal status: completed
- Exception handling: completed
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check
- run_inference

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=Private2D、任务族=segmentation_2d、任务=private_2d_inference_eval、意图类别=standard。
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
12. experiment.inference
13. experiment.standardized_evaluation
14. report.metric_aggregation
15. report.visualization_builder
16. shared.audit_log_finalize
