# D2/T2 Agent 工具 baseline —— Prompt 包（使用说明）

用途：用 OpenAI Codex、GitHub Copilot 等通用 AI 工具对 30 条意图做决策对比评测。

## 使用步骤

1. 打开目标工具（Codex 窗口 / Copilot 边栏），**每条意图开一个全新对话**。
2. 粘贴「固定前言」，再粘贴该条意图的「用户请求」一行（可连在一起粘贴）。
3. 工具返回后，把其中的 JSON **原样**保存为：
   `benchmark_results/baselines/<工具名>/<意图id>.json`
   并在 JSON 中补充两个字段（看对话 UI 显示的）：
   ```json
   "model": "<对话中显示的模型名>", "date": "2026-08-19"
   ```
4. 30 条全部完成后运行评分：
   ```bash
   python benchmark_trace/score_baseline.py --results-dir benchmark_results/baselines/codex
   python benchmark_trace/score_baseline.py --results-dir benchmark_results/baselines/copilot
   ```

## 纪律（数据有效性的前提）

- 每条意图新对话，不跨条聊天。
- 前言逐字相同，不解释、不补充、不纠正工具输出。
- 工具输出不完整/格式错时：**重开对话重试一次**，仍失败则记录失败，不要手动改 JSON 内容。
- 不允许工具执行任何代码/命令（只输出 JSON）。

---

## 固定前言（每条意图都贴这一段）

```
你正在参与一项牙科影像 AI 工作流平台的决策评测。请只依据下方信息做出决策。
不要执行任何代码、不要调用任何工具、不要访问文件系统，只输出一个 JSON。

【可用环境】
- 数据集：
  TDD（2D 全景 X 光片 + 牙齿像素级分割标注 + teeth_bbox.json 检测框标注）、
  ToothFairy3（3D CBCT + 牙齿/修复体分割标注）、
  Private2D（私有全景片 + 四类病灶 bbox 标注）
- 已验证方法注册表（平台可自动执行的唯一路由集合）：
  1. tdd_2d_segmentation_infer_report —— TDD 2D 牙齿二值分割推理 + 报告（segmentation, 2d, inference）
  2. private_2d_segmentation_train —— 私有 2D 牙齿分割训练（segmentation, 2d, private_train）
  3. toothfairy3_3d_segmentation_infer_or_train —— ToothFairy3 3D 分割推理或训练（segmentation, 3d, inference/train）
  4. dental_anomaly_detection —— 2D 全景异常检测（anomaly_detection, 2d, inference/train）
  5. dental_2d_super_resolution —— 2D 超分辨率（super_resolution, 2d, inference）
  6. tta_ensemble_inference —— 2D 分割 TTA+集成推理（segmentation, 2d, inference）
- 代码资产：nnU-Net v2 脚本、YOLO 检测模块、数据质检脚本等（可据此提出外部方案，
  但外部方案须经人工批准后方可执行）

【决策规则】
- 请求与某条注册方法匹配 → outcome="use_registry"，selected_method.id=该注册方法 id，
  supported=true，executable=true，requires_human_review=false
- 无注册方法匹配但任务合理 → outcome="external_proposal"，
  selected_method.id="agent_external_suggestion"，supported=true，executable=false，
  requires_human_review=true
- 请求信息不足/任务模糊 → selected_method.id="platform_clarification"，
  supported=true，executable=false
- 请求超出牙科 CV 范围或无可行路线 → outcome="reject"，selected_method=null，
  supported=false，executable=false

【输出格式】（只输出 JSON，不要任何其他文字）
{
  "intent": {"dataset": "TDD|ToothFairy3|Private2D|unknown",
             "task_family": "segmentation|detection|classification|super_resolution|anomaly_detection|unknown",
             "mode": "train|inference|unknown"},
  "outcome": "use_registry|external_proposal|reject",
  "supported": true,
  "executable": false,
  "selected_method": {"id": "…", "status": "…"},
  "requires_human_review": true,
  "reason": "…"
}
```

---

## 30 条意图（每条 = 固定前言 + 下面一行）

1. DCI-TDD-SEG2D-001：用 TDD 全景片训练一个默认牙齿二值分割模型，并保留 10% 测试集。
2. DCI-TDD-SEG2D-002：把 TDD 的牙齿多边形标注导出为 32 类分割，并训练一个 FDI/32 类牙位分割模型。
3. DCI-TDD-SEG2D-003：帮我用 TDD 做一个区域分割模型，训练好后告诉我效果。
4. DCI-TDD-SEG2D-004：用已有的 TDD 二值分割模型在测试集上推理，并输出 Dice、IoU、HD95 和叠加图。
5. DCI-TDD-SEG2D-005：帮我从 TDD 里挑一个病例出一份牙齿相关报告。
6. DCI-TDD-DET-006：把 TDD 的 teeth_bbox.json 整理成 COCO 牙齿检测数据集，供检测模型训练使用。
7. DCI-TDD-DET-007：基于 TDD bbox 标注训练一个牙齿检测模型，并报告验证集 mAP。
8. DCI-TDD-DET-008：检查 TDD 检测框是否存在越界、重复或病例泄漏；如果发现阻断性缺陷就停止并列出证据。
9. DCI-TDD-CLS-009：用 TDD 全景片训练一个病例级疾病分类模型。
10. DCI-TDD-CLS-010：用 TDD 的图像质量标签训练可用/需复核分类器；训练前先核验质量标签清单，标签不完整就停止。
11. DCI-TF3-SEG3D-011：在 ToothFairy3 CBCT 数据上训练 3D 多类别牙齿分割模型。
12. DCI-TF3-SEG3D-012：在 ToothFairy3 全量 CBCT 上训练 3D 分割模型；如果 QC 发现需人工复核或阻断病例，停止并输出病例清单。
13. DCI-TF3-SEG3D-013：用已有 ToothFairy3 3D 分割模型对验证/测试 CBCT 做推理评估。
14. DCI-TF3-SEG3D-014：训练 ToothFairy3 前先严格检查标签值是否合法；若有非法标签值就阻断训练并列出病例。
15. DCI-TF3-SEG3D-015：对 ToothFairy3 images-only CBCT 子集运行已有模型推理，不做监督训练。
16. DCI-TF3-DET-016：从 ToothFairy3 3D 标签中导出牙齿检测框数据集。
17. DCI-TF3-DET-017：用 ToothFairy3 训练 3D 牙齿检测模型，并报告检测 mAP。
18. DCI-TF3-CLS-018：帮我对 ToothFairy3 做质量分类，看看哪些病例风险比较高。
19. DCI-TF3-CLS-019：帮我按 ToothFairy3 的子集类型做一个分类实验。
20. DCI-TF3-CLS-020：将 ToothFairy3 病例分成 usable 和 needs_manual_review，并输出病例级分类结果。
21. DCI-P2D-SEG2D-021：用私有 2D 数据 imagesTr/labelsTr 训练牙齿分割模型，并用 imagesVal/labelsVal 验证。
22. DCI-P2D-SEG2D-022：对私有 2D imagesVal 做分割推理，并用 labelsVal 计算指标。
23. DCI-P2D-SEG2D-023：帮我给私有 2D 数据出一份临床结果报告。
24. DCI-P2D-SEG2D-024：先检查私有 2D 数据的 train/val/test split 是否存在泄漏或错配；发现阻断问题则不要训练。
25. DCI-P2D-SEG2D-025：将私有 2D 数据整理成 DentalClaw canonical case 格式，再导出分割训练格式。
26. DCI-P2D-DET-026：用私有 2D 数据训练牙齿检测模型。
27. DCI-P2D-DET-027：把私有 2D segmentation mask 自动转成检测框并训练 detector。
28. DCI-P2D-DET-028：将私有 2D 数据导出成 COCO 检测格式；如果缺少检测框标注就停止并报告缺陷。
29. DCI-P2D-CLS-029：用私有 2D 牙片训练疾病分类模型。
30. DCI-P2D-CLS-030：帮我判断私有 2D 里哪些病例的标注能用，哪些需要复核。
