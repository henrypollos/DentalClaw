# DentalClaw 论文返修 — 上下文交接文档

> 用途：新对话开始时，把本文件内容发给我（或让我读取本文件），即可恢复全部上下文。
> 建议新对话开场白：
> 「请先阅读 /data/data2/yiyang/DentalClaw/paper_revision/HANDOFF_上下文交接.md，然后继续帮我处理论文返修。」

---

## 一、项目背景

- **期刊**：Journal of Dentistry，稿件号 JJOD-D-26-01758，返修截止 **2026-09-01**，需附 cover letter。
- **论文主题**：DentalClaw —— 自然语言驱动的牙科影像 AI 工作流编排平台（workflow orchestration / platform validation，**不是**新分割模型、**不是**临床决策支持系统）。
- **篇幅要求**：约 6 页（不含参考文献）；Elsevier `elsarticle` 双栏（`5p` 只是版式名，不是页数限制）。
- **当前主线（评审后重新定位）**：intent → LLM 结构化提案 → 离线注册表确定性路由 → 预设期望结果对照评估 → 未验证路由必须人工确认 → 本地确定性执行 → 完整 audit trail。

## 二、关键文件与状态

| 文件 | 状态/角色 |
|---|---|
| `paper_revision/DentalClaw_main_short_v3.tex` | ⭐ **当前投稿主版本**，已多轮修改，内容与证据对齐 |
| `paper_revision/response` | ⭐ 审稿回复信（LaTeX），已按 v3 重写，编译通过（11 页，EXIT:0，无警告） |
| `paper_revision/DentalClaw_main` | 原主稿（长版）。**注意：仍含旧错误，未同步修正** |
| `paper_revision/DentalClaw_main_short.tex` / `_short_v2.tex` | 早期缩减稿（v1 太短 4 页、v2 中间版），已被 v3 取代 |
| `paper_revision/参考建议` | 学长/同行 24 点修改建议（约 1250 行），v3 已按其中合理项落实 |
| `审稿意见清单.md` | 审稿意见汇总：编辑 1 条 + R1 两条 + R4 七条 + R5 九条 |
| `paper_revision/修改建议` | 用户正在看的文件（本对话最后打开的） |
| `paper_revision/逐条修订清单.md`、`INTEGRATION_GUIDE.tex`、`section_3_4.tex`、`section_4_discussion.tex` | 早期修订产物，参考价值低 |
| `benchmark_results/eval_platform_mvp_20260725.json` | ⭐ 论文表 2 的数据来源（11 条意图评测） |
| `benchmark_trace/eval_platform_mvp.py` | 评测脚本（六维权重定义在此） |
| `platform_mvp/run_platform_mvp.py` | 平台主流程（决策/注册表/`--allow-external` 闸门） |

## 三、已核实的硬事实（写论文/回信必须与这些一致）

### 1. 意图评测（v3 表 2）
- **只有 11 条**（25 条平台子集 `benchmark_intents/intents.platform_mvp.jsonl` 中选出；另有 30 条全集）。
- 得分（来自 `eval_platform_mvp_20260725.json`，**不要改**）：
  1. TDD 全景牙齿分割推理 = 1.00（`tdd_2d_segmentation_infer_report`）
  2. TDD 超分增强 = 1.00（`dental_2d_super_resolution`）
  3. ToothFairy3 CBCT 3D 分割 = 1.00（`toothfairy3_3d_segmentation_infer_or_train`）
  4. 私有数据分割训练 = 1.00（`private_2d_segmentation_train`）
  5. 异常检测 = 1.00（`dental_anomaly_detection`）
  6. "用 TDD 建模型"（模糊）= 1.00（`platform_clarification`）
  7. "写牙科 AI 论文"（trap）= 1.00（Rejected）
  8. "视频动作识别"（trap）= 1.00（Rejected）
  9. 私有数据推理未指定任务 = **0.95**（`platform_clarification`）
  10. TDD 牙齿分割训练 = **0.985**（`agent_external_suggestion`）
  11. TDD bbox 检测训练 = **0.81**（`agent_external_suggestion`）
- **总均值 = 0.9768 ≈ 0.977**（简单平均，非加权平均）。
- 六维权重（代码 `DIMENSION_WEIGHTS`，Methods 已写明）：planning 0.35、qc_blocking 0.20、intent_parsing 0.15、external_proposal 0.15、ambiguity 0.10、boundary 0.05；通过线 0.70。
- 论文口径：**"study-defined weighted decision score"**，绝不写成 accuracy/reliability。

### 2. 代码事实（v3 方法部分已引用）
- 推理后端：OpenClaw 运行时 + DeepSeek V4 Pro，**temperature 0.1**，要求输出 JSON，带 fenced/malformed 解析回退。
- 失败安全：仅离线注册表方法可自动执行；外部提案需 `--allow-external` 显式确认；执行由本地版本化模块完成，agent 不直接执行代码。
- 文献检索：Europe PMC + arXiv（辅助上下文，不决定注册表路由）。
- 数据安全：发给外部 API 的 prompt 仅含文本任务描述、注册表元数据、代码资产描述；原始影像/标注/患者标识本地处理。
- 划分：case 级、固定随机种子、重复标识符泄漏检查（**代码不能证实 patient-level，论文未写 patient-level**）。
- TTA/集成路由：`platform_mvp/tta_ensemble.py`（注册表 `tta_ensemble_inference`）。

### 3. 数据集数字
- TDD：1051 入组 → **968** 保留（缺失 mask、图像-标签不匹配）。
- ToothFairy3：63 卷 → **43** 可用（金属伪影、FOV 异常）。
- 私有全景数据：**290** 例，2012–2024，伦理批号 **PKUSSIRB-2025,107,012**，临床专家标注（pulp、根管充填、充填体、根尖周病变 bbox）。
- 工作流耗时（表 1）：全景 audit+prep 14min / train 6h52m / infer+eval+report 31min；CBCT 21min / 19h36m / 1h44m；手动对比 audit+prep 3h26m / 6h12m 等。

### 4. 引用
- **MONAI 引用错误已修**：正文改为 nnU-Net、MONAI（`Cardoso2022MONAI`）、3D Slicer 三处分开引用。
- ⚠️ **cas-refs.bib 不在仓库（在 Overleaf）**，需要在 Overleaf 添加 MONAI 条目（arXiv:2211.02701, Cardoso et al. 2022）。

## 四、v3 已定的表述规范（不要倒退）

- 0.977 → "study-defined weighted decision score / overall average 0.977"；小节名 "Intent-to-workflow decision evaluation"（不是 accuracy）。
- "clinician-facing" → **researcher-oriented**；"expert review" → downstream review；"lowers the technical barrier" → "is designed to reduce the coordination burden"。
- 手动对比 → "workflow demonstration, not a controlled human-factor experiment"；训练时间基本不变（平台是协调层不是加速器）。
- 可复现性 → 设计意图 + traceability 证据（小节名 "Workflow completion and execution traceability"）。
- 标题保留 "Auditable"（有 audit-trail 段落支撑）。
- 局限性段（已精简为 3 句）：代表性任务/自建意图集与权重/无正式可用性实验；保守外部策略的双面性；future work。

## 五、已完成的重要修改（摘要）

1. v1（4 页，太短）→ v2 → **v3**（接近 6 页）逐步扩充。
2. v3 多轮审查：表 2 分数与真实数据对齐、六维权重补全、gold standard 来源说明（研究组预定义）、伦理批号、排除原因、LLM 配置与安全边界、fail-safe 策略、audit trail、MONAI 引用、术语降级、局限性精简。
3. `response` 回复信完全重写对齐 v3：删除虚构承诺（DentalClawBench 30 条、消融、replay、可用性实验、supplementary S1–S7 等），全部 `\pending` 占位符清除，编译通过。

## 六、未决事项 / 下一步候选（新对话可从这里继续）

1. **是否补测 3 条意图**（已给用户方案但未执行）：`007` 分类、`019` ResNet 陷阱、`023` 3D 训练 → 11 变 14 条，表 2 加 3 行、均值重算（评测脚本 `eval_platform_mvp.py` 现成，每条约 50–80 秒）。
2. **是否在 Methods 2.5 补一句抽样原则**（11 条如何覆盖四类行为）——已给文案。
3. **是否加"平台能做什么"能力清单段**（Introduction 末段或 Methods 2.1 之后）——已给 6 条草稿。
4. **是否加 replay 局限性句**（"未做正式独立 replay/重建实验"）——已给文案，用户未定。
5. **主稿 `DentalClaw_main` 的错误未同步**（表 2 分数、MONAI 引用、clinician-facing 等）——若最终投稿用主稿则必须修。
6. **Overleaf 编译确认页数**（本机缺 `elsarticle.cls` 无法编译论文；response 可编译）。
7. 投稿前：response 中页码/行号占位需按最终 line-numbered 版更新；加 MONAI bib 条目。

## 七、环境注意

- 本机 pdflatex 缺 `elsarticle.cls` → **论文无法本地编译**，请用 Overleaf。
- `response` 用标准宏包，本机可编译（验证命令：复制为 `.tex` 后 pdflatex）。
- 论文用**美式英语**（standardized/organization）。
- 所有论文数字以 `benchmark_results/eval_platform_mvp_20260725.json` 与上述硬事实为准，不要凭空改写。
