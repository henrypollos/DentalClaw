# DentalClaw 平台 Adapter 接入规范

本文用于把 planned adapter 变成可执行路线。老师关心的是平台底座是否能持续接入任务，因此每个 adapter 都必须遵守同一套输入、执行、产物和验收规则。

## 1. Adapter 的定义

在当前平台 MVP 中，一个 adapter 是：

```text
离线方法表中的一条 method
  + 输入契约
  + 前置检查
  + 现有代码入口或新绑定脚本
  + 标准化输出
  + 失败/拒绝策略
```

adapter 不是简单写一个命令。它必须告诉平台：

- 什么任务能接；
- 需要什么数据；
- 哪些条件不满足时必须停止；
- 调哪个现有入口；
- 成功后产出什么；
- 失败时如何给医生解释。

## 2. 通用接入门槛

每条路线从 `planned_adapter` 升级为 `executable` 前，至少满足：

| 门槛 | 要求 |
| --- | --- |
| 输入契约 | 明确 images、labels、metadata、split、checkpoint 等必需项 |
| 前置检查 | 有脚本能判断是否可执行，不能只靠人工判断 |
| 入口脚本 | 有稳定的 repo 内入口，不能临时手敲命令 |
| 输出目录 | 统一落到 `artifacts/platform_mvp_runs/` 或 adapter 指定目录 |
| 证据文件 | 至少有 JSON manifest 和 Markdown summary |
| 安全边界 | 缺标签、缺模型、任务不支持时必须 stop/reject，不允许伪造训练或指标 |

## 3. 私有 2D 分割训练 Adapter

当前状态：`planned_adapter`

目标状态：`executable`

### 输入契约

参考：

```text
platform_mvp/private_data_contract.json
```

最小训练输入：

```text
private_dataset/
  images/
    case001.png 或 case001.dcm
  masks/
    case001.png
  metadata.json 可选
  splits.json 可选
```

当前真实数据 `data/private01` 的检查结果：

```text
139 个 DICOM 图像
0 个 mask/label
决策：stop_without_training
```

这说明它可以进入登记/QC，但不能作为监督分割训练输入。

### 升级为 executable 的工作

1. 补齐私有图像对应 mask/label，或指定 annotation export 格式。
2. 用 `validate_private2d_package.py` 通过 `private_train` 检查。
3. 写一个 private2d -> nnUNet export adapter。
4. 生成 dataset spec、task spec、budget spec。
5. 调用 `agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py`。
6. 训练完成后复用 inference/report 证据收集逻辑。

### 验收标准

- `private2d_validation_report.json` 中 `can_execute_requested_mode=true`。
- 训练 run 下有 `run_status.json`。
- 训练 run 下有 best checkpoint。
- 有 inference summary。
- 有 Markdown 汇总。
- 若缺标签，必须停止在前置检查，不允许启动训练。

## 4. ToothFairy3 3D 分割 Adapter

当前状态：`planned_adapter`

目标：先做 3D QC 或 inference demo，再做训练。

已有线索：

```text
agents/data_curator/reports/cbct_qc/toothfairy3_lps_tiny/
agents/data_curator/skills/core/cbct_qc/scripts/audit_cbct_dataset.py
```

### 推荐最小路线

```text
intent.parse
-> registry.method_lookup
-> dataset.cbct_qc
-> dataset.prepare_3d_specs
-> experiment.training_or_inference
-> platform.collect_evidence
```

### 验收标准

- 能对 ToothFairy3 子集生成 CBCT QC 报告。
- 能明确区分有标签训练、images-only inference、不可执行状态。
- 不允许在 labels 缺失时生成监督训练指标。

## 5. 异常检测 Adapter

当前状态：`planned_adapter`

关键问题：当前仓库没有稳定异常检测入口。

### 接入前必须明确

- 异常标签是 image-level、case-level 还是 pixel-level。
- 无标签时是否只允许 inference。
- 输出是 anomaly score、heatmap 还是二分类结果。
- 指标是 AUROC、F1、sensitivity，还是只给可视化。

### 验收标准

- 有明确标签契约。
- 有 entrypoint。
- 有标签时可计算 AUROC/F1。
- 无标签时只输出推理结果或拒绝训练，不伪造指标。

## 6. 超分 Adapter

当前状态：`executable`

老师要求增加超分任务，因此它进入能力表。当前已绑定一个确定性的 bicubic MVP baseline，证明平台可以执行超分任务、生成图像和指标；论文实验阶段应继续接入学习型 baseline。

### 当前已完成

- `platform_mvp/run_super_resolution_mvp.py`
- 从 TDD 图像生成合成低清输入。
- bicubic 恢复到原始尺寸。
- 输出原图、低清图、超分图、对比图。
- 在合成配对设置下输出 PSNR/SSIM。

### 后续升级要求

- 若用于论文主实验，应接入学习型超分 baseline。
- 真实低清/高清配对数据存在时输出 PSNR/SSIM。
- 无配对 GT 时只输出视觉对比和限制说明，不能报告配对指标。

## 7. Status 变更规则

`method_registry.json` 中的 `status` 必须按以下规则更新：

| 状态 | 含义 |
| --- | --- |
| `planned_adapter` | 属于平台范围，但当前不能执行 |
| `executable` | 有输入契约、前置检查、入口脚本和可复查输出 |
| `blocked` | 属于平台范围，但缺关键数据或依赖 |
| `rejected` | 不属于当前 CV 平台范围 |

不能因为路线写进 registry 就声称已完成。只有跑出 evidence package 才能标为 `executable`。

## 8. 下一步优先级

1. 私有 2D 分割训练：优先级最高，因为老师明确强调私有数据训练。
2. ToothFairy3 3D 分割：优先用已有 CBCT QC 产物做最小 demo。
3. 超分：已接入 bicubic MVP，后续升级为学习型 baseline。
4. 异常检测：先定义标签契约，再选方法。
