# DentalClaw Benchmark Run: DCI-TF3-CLS-019

- Prompt: 帮我按 ToothFairy3 的子集类型做一个分类实验。
- Dataset: ToothFairy3 (/data/data2/yiyang/JoD/ToothFairy3_LPS)
- Task family: classification
- Intent category: ambiguous
- Expected behavior: ask_clarification
- Terminal status: awaiting_clarification
- Exception handling: request_clarification
- Dry run: True

## Tool Calls
- dataset_registry_lookup
- capability_registry_check

## Orchestrator Reasoning
- MainAgent.parse_intent: 解析自然语言 prompt，识别数据集=ToothFairy3、任务族=classification、任务=subset_family_classifier、意图类别=ambiguous。
- MainAgent.request_clarification: 该 prompt 省略了关键实验信息；直接训练会把用户意图误绑定到错误任务或标签体系，因此先追问。
- MainAgent.await_user_response: 在用户补充任务类型、目标标签或病例/模型前，不启动训练、推理或报告生成。
- MainAgent.stop_without_training: 模糊意图进入等待澄清状态时必须显式停止训练/推理，避免后台继续执行。

## Reference Workflow
1. coordinator.intent_parse
2. coordinator.dataset_registry_lookup
3. coordinator.capability_check
4. coordinator.ambiguity_detection
5. coordinator.request_clarification
6. coordinator.await_user_response
7. coordinator.stop_without_training
8. shared.audit_log_finalize
