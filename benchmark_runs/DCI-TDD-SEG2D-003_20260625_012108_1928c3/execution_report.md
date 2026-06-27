# DentalClaw Benchmark Run: DCI-TDD-SEG2D-003

- Prompt: 帮我用 TDD 做一个区域分割模型，训练好后告诉我效果。
- Dataset: TDD (/data/data2/yiyang/DentalClaw/data/TDD)
- Task family: segmentation_2d
- Intent category: ambiguous
- Expected behavior: ask_clarification
- Terminal status: awaiting_clarification
- Exception handling: request_clarification
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=TDD、任务族=segmentation_2d、任务=maxillomandibular_binary_train、意图类别=ambiguous。
- MainAgent.request_clarification: 该 prompt 省略了关键实验信息；直接训练会把用户意图误绑定到错误任务或标签体系，因此先追问。
- MainAgent.await_user_response: 在用户补充任务类型、目标标签或病例/模型前，不启动训练、推理或报告生成。

## Reference Workflow
1. coordinator.intent_parse
2. coordinator.dataset_registry_lookup
3. coordinator.capability_check
4. coordinator.ambiguity_detection
5. coordinator.request_clarification
6. coordinator.await_user_response
7. coordinator.stop_without_training
8. shared.audit_log_finalize
