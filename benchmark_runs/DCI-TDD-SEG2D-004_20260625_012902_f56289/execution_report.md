# DentalClaw Benchmark Run: DCI-TDD-SEG2D-004

- Prompt: 用已有的 TDD 二值分割模型在测试集上推理，并输出 Dice、IoU、HD95 和叠加图。
- Dataset: TDD (/data/data2/yiyang/DentalClaw/data/TDD)
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
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=TDD、任务族=segmentation_2d、任务=teeth_binary_inference_eval、意图类别=standard。
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
10. experiment.inference
11. experiment.standardized_evaluation
12. report.metric_aggregation
13. report.visualization_builder
14. shared.audit_log_finalize
