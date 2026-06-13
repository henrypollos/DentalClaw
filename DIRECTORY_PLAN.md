# DentalClaw 目录规划

## 1. 目标

本规划面向 DentalClaw 的三类核心能力：

1. 数据闭环：数据入口、来源管理、规范化、数据说明与交付。
2. 训练闭环：自动训练、自动调参、自动选优、推理与实验报告。
3. 结果闭环：新病例推理、结果增强、临床导向展示与报告输出。

目录设计目标：

- 清楚区分代码、临时运行现场、正式发布产物。
- 支持多 agent 协作，而不是让 agent 直接读彼此的私有工作区。
- 支持开源整理，便于后续公开数据流程、训练流程、结果流程。
- 支持论文写作、导师汇报、结果回溯。

开源根目录：

- `/data/data2/yiyang/DentalClaw`

## 2. 角色分工映射

### 伊洋

负责数据入口、项目组织、论文出口。

对应目录职责：

- 数据源登记
- 数据整理与标准化
- 数据说明文档
- 项目管理文档
- 对外汇报材料

### 义瑞

负责训练闭环，把现有训练流程封装成 OpenClaw skills。

对应目录职责：

- 2D 全景片训练流程
- 3D CBCT 训练流程
- 自动训练与调参
- 模型选择
- 推理与实验报告

### 航天

负责结果闭环，体现临床可用性。

对应目录职责：

- TTA
- ensemble
- 几何后处理
- 新病例推理
- 面向医生的结果展示与报告

## 3. 总体分层

DentalClaw 根目录建议分成四层：

1. `agents/`
   - 放 agent 身份、规则、局部技能与 agent 级文档。
   - 不作为正式数据和模型的长期仓库。

2. `workspace/`
   - 放每个 agent 的临时运行现场。
   - 允许中断、失败、重跑。
   - 不作为跨 agent 的稳定依赖路径。

3. `artifacts/`
   - 放正式发布的共享产物。
   - 下游 agent 只从这里消费，不直接读别人的 workspace。

4. `registry/`
   - 放数据集、模型、病例批次与谱系关系的索引。
   - 用于项目组织、导师汇报、论文回溯。

## 4. 推荐目录树

```text
DentalClaw/
  agents/
    data_curator/
    trainer/
    clinical_result/

  skills/
    data/
    train/
    infer/
    result/
    shared/

  workspace/
    data_curator/
      intake_runs/
      normalize_runs/
      qa_runs/
    trainer/
      train_runs/
      preprocess_cache/
      tuning_runs/
      infer_runs/
    clinical_result/
      case_runs/
      tta_runs/
      ensemble_runs/
      postprocess_runs/

  artifacts/
    datasets/
      <dataset_name>/
        <dataset_version>/
          manifest.json
          README.md
          source_provenance.json
          schema.json
          qa_report.json
          canonical/
          exports/
    models/
      <task_name>/
        <model_version>/
          model.pth
          model_card.json
          infer_config.json
          metrics.json
          train_summary.json
          dataset_ref.json
    case_batches/
      <batch_name>/
        <batch_version>/
          manifest.json
          cases/
    results/
      <model_version>/
        <batch_version>/
          predictions/
          postprocessed/
          summary.json
          review_items.json
          report.md
    papers/
      outlines/
      figures/
      tables/
      drafts/
    reports/
      weekly/
      milestones/
      advisor_updates/

  registry/
    datasets.json
    models.json
    case_batches.json
    lineage.json

  docs/
    architecture/
    workflows/
    conventions/
```

## 5. 目录职责说明

### 5.1 `agents/`

用途：

- 放 agent 的角色定义、行为规则、局部工作说明。
- 放 agent 私有 skill 或 agent 专属流程说明。

原则：

- 这里是 agent 的“脑”，不是项目的正式产物仓库。
- 正式数据、正式模型、正式报告不要长期沉积在这里。

### 5.2 `skills/`

用途：

- 放可复用流程。
- 面向 OpenClaw 技能化封装。

建议子类：

- `data/`：数据入口、来源登记、规范化、数据质检。
- `train/`：训练、调参、推理、实验管理。
- `infer/`：基础推理、多模型推理、批量推理。
- `result/`：TTA、ensemble、后处理、报告导出。
- `shared/`：跨阶段都要用到的公共工具。

### 5.3 `workspace/`

用途：

- 放运行态现场。
- 放中间缓存。
- 放失败任务的复盘现场。

设计原则：

- 每个 agent 单独一块。
- 每次运行尽量按 run 维度建目录。
- 这里的路径不应成为下游 agent 的长期依赖。

#### `workspace/data_curator/`

适合放：

- 临时 intake 快照
- 原始目录扫描结果
- 规范化中间文件
- 质检失败样本

#### `workspace/trainer/`

适合放：

- 训练 run
- 预处理缓存
- 自动调参多轮结果
- 临时 checkpoint
- 当前 run 的 best model

#### `workspace/clinical_result/`

适合放：

- 新病例批次运行现场
- TTA 中间结果
- ensemble 中间结果
- 几何后处理中间结果
- 可视化缓存

### 5.4 `artifacts/`

这是整个系统最重要的共享层。

原则：

- 只放正式发布产物。
- 下游 agent 消费这里，而不是上游的 workspace。
- 尽量版本化。
- 尽量不可变。

#### `artifacts/datasets/`

由数据端发布，供训练端使用。

每个数据版本建议包含：

- `manifest.json`
- `README.md`
- `source_provenance.json`
- `schema.json`
- `qa_report.json`
- `canonical/`
- `exports/`

其中：

- `canonical/` 是标准化后的内部统一数据。
- `exports/` 是面向训练任务的直接可用数据。

#### `artifacts/models/`

由训练端发布，供结果端使用。

每个模型版本建议包含：

- `model.pth`
- `model_card.json`
- `infer_config.json`
- `metrics.json`
- `train_summary.json`
- `dataset_ref.json`

这样做的好处：

- clinical/result agent 只关心稳定模型地址。
- 训练 run 可以删，但已发布模型不受影响。
- 模型和数据版本之间能建立稳定引用。

#### `artifacts/case_batches/`

用途：

- 统一管理测试病例批次。
- 让结果 agent 和训练/eval 流程使用同一批测试输入。

#### `artifacts/results/`

由结果端发布，供项目汇报、临床展示、论文写作使用。

建议包含：

- `predictions/`
- `postprocessed/`
- `summary.json`
- `review_items.json`
- `report.md`

#### `artifacts/papers/`

用途：

- 论文大纲
- 图表
- 草稿
- 最终材料整理

#### `artifacts/reports/`

用途：

- 阶段性里程碑报告
- 周报
- 导师汇报材料

### 5.5 `registry/`

建议这是项目组织层的“总账本”。

包含：

- `datasets.json`
- `models.json`
- `case_batches.json`
- `lineage.json`

作用：

- 记录有哪些正式数据版本
- 记录有哪些正式模型版本
- 记录模型对应哪个数据集版本
- 记录哪个结果报告来自哪个模型和病例批次

## 6. 三个 agent 的正式交接方式

### 数据端 -> 训练端

正式交接路径：

- `artifacts/datasets/<dataset_name>/<dataset_version>/`

训练端读取：

- `exports/` 中的训练输入
- `manifest.json`
- `schema.json`
- `qa_report.json`

不建议训练端直接读取：

- `workspace/data_curator/`

### 训练端 -> 结果端

正式交接路径：

- `artifacts/models/<task_name>/<model_version>/`

结果端读取：

- `model.pth`
- `infer_config.json`
- `model_card.json`

不建议结果端直接读取：

- `workspace/trainer/train_runs/...`

### 结果端 -> 项目组织/论文/汇报

正式交接路径：

- `artifacts/results/<model_version>/<batch_version>/`
- `artifacts/reports/`
- `artifacts/papers/`

## 7. 命名和版本建议

### 数据版本

建议格式：

- `vYYYYMMDD`
- 或 `vYYYYMMDD_<short_desc>`

示例：

- `v20260325`
- `v20260325_pano_teeth_binary`

### 模型版本

建议格式：

- `<task>_<dataset_version>_<run_tag>`

示例：

- `pano_teeth_seg_v20260325_r001`
- `cbct_tooth_instance_v20260410_r003`

### 测试批次版本

建议格式：

- `batch_YYYYMMDD_<purpose>`

示例：

- `batch_20260325_internal_eval`
- `batch_20260325_clinical_demo`

## 8. 目录与职责的最终对应

### 伊洋

主要负责：

- `artifacts/datasets/`
- `registry/datasets.json`
- `registry/lineage.json`
- `artifacts/reports/`
- `artifacts/papers/`
- `docs/`

### 义瑞

主要负责：

- `skills/train/`
- `workspace/trainer/`
- `artifacts/models/`
- `registry/models.json`

### 航天

主要负责：

- `skills/result/`
- `workspace/clinical_result/`
- `artifacts/results/`
- `registry/case_batches.json`

## 9. 设计结论

最合理的目录设计原则是：

- agent 有自己的私有运行空间
- 项目有独立的公共发布空间
- 正式交接只通过 `artifacts/`
- 项目总览和可追溯关系通过 `registry/`

对应一句话总结：

- 数据由数据端在 `artifacts/datasets/` 发布
- 模型由训练端在 `artifacts/models/` 发布
- 临床导向结果由结果端在 `artifacts/results/` 发布
- 中间运行现场全部留在 `workspace/`

这样最适合 DentalClaw 后续做自动化、做开源、做汇报、做论文，也最适合多个 agent 长期协作。
