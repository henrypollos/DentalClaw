# DentalClaw 平台底座 MVP

本目录用于把上次会议中老师对义瑞部分的要求落到一个可汇报、可运行的最小平台底座上。它不是新的 benchmark，也不是单纯的实验脚本，而是把“用户一句话输入 -> 平台自动选择方法 -> 调用既有 DentalClaw 代码 -> 收集结果证据”串起来。

已有的 `mvp_fullflow/` 负责跑通一个具体任务：TDD 2D 牙齿二值分割推理、评估和报告。本目录在它之上增加平台层：

```text
一句话需求
  -> intent parse
  -> offline method registry lookup
  -> workflow plan
  -> existing DentalClaw entrypoint
  -> evidence package
```

## 1. 对应老师的要求

会议中老师的核心要求可以归纳为：

1. 论文定位是工具平台，不是 benchmark。
2. 能力边界先收窄到 CV 任务。
3. 当前只做既定框架内的推理和私有数据训练，不做跨框架自动找 baseline。
4. 医生只输入一句话，系统要能自动选方法、调参、训练或推理，并收集结果。
5. 先跑通核心闭环，再补实验和论文写作。
6. 2D/3D 分割之外，需要把异常检测和超分任务纳入平台能力表，用多任务实证支撑平台能力。

本目录的实现对应如下：

| 老师要求 | 当前落地 |
| --- | --- |
| 工具平台，不是 benchmark | `run_platform_mvp.py` 生成平台计划并可调用真实 DentalClaw 入口 |
| CV 范围 | `method_registry.json` 的 `allowed_domain` 固定为 `dental_cv` |
| 只支持推理和私有数据训练 | registry 的 `allowed_modes` 限定为 `inference` 与 `private_train` |
| 一句话输入 | `--intent` 接收中文自然语言请求 |
| 自动选方法 | 离线 `method_registry.json` 做方法选择 |
| 先跑通闭环 | 可执行路线复用 `mvp_fullflow/run_mvp_fullflow.py` |
| 多任务扩展 | registry 已登记 2D 分割、3D 分割、异常检测、超分的 adapter 状态 |

## 2. 文件说明

```text
platform_mvp/
  method_registry.json
  private_data_contract.json
  run_platform_mvp.py
  run_platform_mvp.sh
  validate_private2d_package.py
  generate_readiness_matrix.py
  generate_report_pack.py
  README.md
  会议任务落实方案.md
  汇报提纲.md
  Adapter接入规范.md
```

- `method_registry.json`：离线方法表。它是老师说“离线建表”的最小版本，用于解决医疗用户不会选模型、不会找方法的问题。
- `private_data_contract.json`：私有 2D 训练输入契约，定义 images、masks、metadata、split 等最小要求。
- `run_platform_mvp.py`：平台 MVP 编排器。负责解析一句话、查表、生成 workflow，并在可执行路线下调用已有 full-flow 脚本。
- `run_platform_mvp.sh`：运行入口。默认使用普通 `python` 启动平台编排器，委托 DentalClaw 脚本时使用 `/home/yiyang/miniconda3/envs/nnunetv2/bin/python`。
- `validate_private2d_package.py`：私有 2D 输入包预检查脚本，避免在缺标注时错误启动监督训练。
- `generate_readiness_matrix.py`：生成多任务路线状态表，区分已执行、被数据阻塞、有 QC 基础、缺入口脚本等状态。
- `generate_report_pack.py`：生成老师汇报包，把平台 demo、full-flow 证据和私有数据预检查汇总到一份 Markdown。
- `会议任务落实方案.md`：面向老师的任务拆解与工程落实方案。
- `汇报提纲.md`：会议汇报时可直接照着讲的提纲。
- `Adapter接入规范.md`：planned adapter 升级为 executable 的工程门槛和验收标准。

## 3. 快速运行

在项目根目录运行 plan-only 模式：

```bash
sh platform_mvp/run_platform_mvp.sh \
  --intent "用 TDD 全景片做牙齿二值分割推理，并生成病例报告"
```

这个命令不会跑重任务，只生成平台计划：

```text
artifacts/platform_mvp_runs/platform_mvp_YYYYMMDD_HHMMSS/
  platform_plan.json
  platform_summary.md
```

使用已经跑通的 TDD full-flow 结果做快速可执行 demo：

```bash
sh platform_mvp/run_platform_mvp.sh \
  --intent "用 TDD 全景片做牙齿二值分割推理，并生成病例报告" \
  --execute \
  --reuse-inference \
  --reuse-fullflow-run artifacts/mvp_runs/tdd_binary_fullflow_20260709_085641 \
  --run-dir artifacts/platform_mvp_runs/tdd_platform_demo
```

该命令会：

1. 解析一句话意图。
2. 在离线方法表中选中 `tdd_2d_segmentation_infer_report`。
3. 调用 `mvp_fullflow/run_mvp_fullflow.py`。
4. 复用已有推理结果，重建报告和证据摘要。
5. 在 `artifacts/platform_mvp_runs/tdd_platform_demo/` 下生成平台层摘要。

## 4. 输出内容

平台层输出：

```text
platform_plan.json
platform_summary.md
execution_result.json
logs/delegate_stdout.log
logs/delegate_stderr.log
```

委托 full-flow 输出仍在：

```text
artifacts/mvp_runs/tdd_binary_fullflow_20260709_085641/
  manifest.json
  mvp_summary.md
  inference/inference_summary.json
  report_case_100/report.html
  report_case_100/overlay.png
```

其中最适合汇报展示的是：

- `platform_summary.md`：说明一句话如何被解析、如何选方法、走了哪些 workflow。
- `execution_result.json`：机器可读的执行状态和关键产物路径。
- `mvp_summary.md`：真实 full-flow 的数据、QC、指标和报告产物。
- `report_case_100/report.html` 与 `overlay.png`：直观展示结果。

## 5. 当前已接通的路线

当前第一条真实可执行路线是：

```text
TDD 2D tooth binary segmentation inference and report
```

它复用已有结果：

```text
artifacts/mvp_runs/tdd_binary_fullflow_20260709_085641
```

该路线已经验证：

- 输入：TDD 2D 全景片测试集。
- 模型：已有 nnU-Net v2 checkpoint。
- 推理：100 个测试病例。
- 指标：mean Dice 0.9130，mean IoU 0.8549，mean HD95 12.9732。
- 报告：病例 `100` 的 HTML report 和 overlay。

当前新增的第二条可执行路线是：

```text
Dental 2D image super-resolution
```

运行命令：

```bash
sh platform_mvp/run_platform_mvp.sh \
  --intent "对 TDD 全景片做超分辨率增强并输出质量报告" \
  --execute \
  --case-id 100 \
  --run-dir artifacts/platform_mvp_runs/super_resolution_platform_demo
```

该路线使用确定性的 bicubic baseline 做平台 MVP 验证，输出：

- `super_resolution_summary.json`
- `super_resolution_summary.md`
- 原图、低清图、超分图
- 对比图
- PSNR/SSIM

当前病例 `100` 的结果为 mean PSNR 37.9045，mean SSIM 0.9986。该路线用于证明超分任务已经能进入统一平台执行和证据收集流程；论文实验阶段仍应替换或补充学习型超分 baseline。

## 6. 当前未接通但已纳入平台表的路线

为了符合老师要求，`method_registry.json` 已把这些路线纳入平台能力表，但状态标为 `planned_adapter`：

- 私有 2D 牙科影像分割训练。
- ToothFairy3 3D CBCT 分割。
- 牙科异常检测。

这些不是假装已经完成，而是明确告诉老师：平台底座已经支持“登记、选择、拒绝或解释”的机制，后续只需要逐个补 adapter，而不是重写平台结构。

## 7. 私有数据训练预检查

当前仓库中有一个私有 DICOM 数据目录：

```text
data/private01
```

可以用下面命令检查它是否满足私有 2D 分割训练输入契约：

```bash
python platform_mvp/validate_private2d_package.py \
  --data-root data/private01 \
  --mode private_train \
  --out-dir artifacts/platform_mvp_runs/private01_validation
```

当前检查结果是：`data/private01` 有 139 个 DICOM 图像，但没有检测到 mask/label，因此平台决策为 `stop_without_training`。这可以用于汇报时说明：平台会支持私有数据训练，但缺少监督标签时必须停止，不能伪造训练能力或指标。

## 8. 生成多任务 Readiness Matrix

为了回答“2D/3D/异常检测/超分现在分别到哪一步”，可以生成多任务路线状态表：

```bash
python platform_mvp/generate_readiness_matrix.py \
  --out-dir artifacts/platform_mvp_runs/readiness_matrix_latest
```

输出包括：

```text
readiness_matrix.json
readiness_matrix.md
```

该矩阵会把路线分成：

- `executable_verified`：已经真实跑通并有证据。
- `blocked_by_missing_labels`：属于平台范围，但当前输入不满足执行条件。
- `planned_with_qc_basis`：已有 QC 或前置基础，但执行 adapter 未接通。
- `planned_missing_entrypoint`：已纳入能力表，但缺 baseline 或稳定入口。

## 9. 生成老师汇报包

在完成平台 demo 和私有数据预检查后，可以生成一份汇总材料：

```bash
python platform_mvp/generate_report_pack.py \
  --out-dir artifacts/platform_mvp_runs/advisor_report_pack_latest
```

输出包括：

```text
老师汇报材料.md
report_pack_manifest.json
evidence/
  platform_summary.md
  execution_result.json
  fullflow_mvp_summary.md
  private2d_validation_report.md
  readiness_matrix.md
```

`老师汇报材料.md` 是最适合开会时直接展示的总览。

## 10. 汇报时的重点说法

可以这样讲：

> 我根据上次会议把方向从 benchmark 调整为工具平台 MVP。现在先收窄到牙科 CV 任务，做了离线方法表和一句话编排器。用户输入一句话后，平台会解析任务、查表选方法、生成 workflow，并在已接通路线中调用已有 DentalClaw 代码完成真实推理和报告。当前已经跑通 TDD 2D 分割推理报告闭环，并接通了一个 2D 超分 MVP baseline；私有数据训练、3D 分割和异常检测按同一个 registry-adapter 机制继续接入。
