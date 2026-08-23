# DentalClaw 最小全流程 MVP 实验

本目录用于完成一个最小但真实可运行的 DentalClaw 全流程实验。它不重新实现模型、推理或报告逻辑，而是调用当前项目中已经存在的代码入口，将现有数据、QC、模型、推理评估和报告生成串成一个完整闭环。

## 1. 目标

导师当前任务的重点是先验证流程可行，因此本 MVP 选择最稳的固定任务：

> 使用当前项目已经准备好的 TDD 全景片二值牙齿分割数据集和已训练 checkpoint，在测试集上重新推理评估，并为一个病例生成结构化 review report。

该 MVP 证明的是：

- 当前项目已有数据/QC 产物可以被读取和复核。
- 当前项目已有 checkpoint 可以进入推理评估流程。
- 当前项目已有报告脚本可以基于预测 mask 生成病例级报告和 overlay。
- 全部输出可以落到一个统一的运行目录中，便于给导师检查。

该 MVP 暂不证明：

- LLM 能从任意自然语言请求可靠生成恰当工作流。
- detection、classification 等其他任务已经完成平台级验证。
- 报告组件可以作为临床决策支持工具。

## 2. 使用到的已有项目代码

主脚本 `run_mvp_fullflow.py` 会调用以下现有入口：

- 推理与评估：
  `agents/experimentation/skills/tooth_autoinfer_nnunet/scripts/run_inference.py`
- 病例报告：
  `agents/clinical_result/skills/clinical_report/run_report.py`

默认复用以下现有资产：

- 数据集：
  `artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D`
- 数据集 spec：
  `artifacts/results/specs/dataset_spec_501_binary.json`
- 任务 spec：
  `artifacts/results/specs/task_spec_501_binary.json`
- 已训练模型：
  `artifacts/training_runs/trial_501_binary_baseline/best_model/checkpoint_best.pth`
- 数据导出状态：
  `artifacts/datasets/nnUNet/nnUNet_delivery_Dataset501_TDDTeethBinary2D.status.json`
- QC 报告：
  `artifacts/results/reports/datasets_qc/Dataset501_TDDTeethBinary2D.qc.json`

运行时脚本会在本次输出目录生成 `runtime_task_spec.json`。该文件从原始 `task_spec_501_binary.json` 复制而来，并自动写入已有训练记录中的最佳 nnUNet trainer 和 fold=`all`，用于让现有推理入口定位正确的 nnUNet 结果目录。原始 spec 文件不会被修改。

## 3. 运行方式

在项目根目录 `$DENTALCLAW_HOME` 下运行：

```bash
sh mvp_fullflow/run_mvp_fullflow.sh
```

默认会使用：

```bash
$CONDA_HOME/envs/nnunetv2/bin/python
```

如果需要指定 Python 环境：

```bash
PYTHON_BIN=/path/to/python sh mvp_fullflow/run_mvp_fullflow.sh
```

注意：完整 nnUNet 推理会使用 Python multiprocessing。如果在受限沙箱或不允许本地 socket 的环境中运行，可能出现 `PermissionError: [Errno 1] Operation not permitted`。这种情况下应在普通项目 shell 中运行完整命令；如果推理已经完成，只需要重建报告和汇总，可以使用后文的 `--reuse-inference`。

如果需要指定病例 ID，例如病例 `1016`：

```bash
sh mvp_fullflow/run_mvp_fullflow.sh --case-id 1016
```

如果要指定输出目录：

```bash
sh mvp_fullflow/run_mvp_fullflow.sh --run-dir artifacts/mvp_runs/my_mvp_run
```

## 4. 输出内容

默认输出目录形如：

```text
artifacts/mvp_runs/tdd_binary_fullflow_YYYYMMDD_HHMMSS/
```

主要产物包括：

```text
manifest.json
mvp_summary.md
runtime_task_spec.json
evidence/
  nnUNet_delivery_Dataset501_TDDTeethBinary2D.status.json
  nnUNet_delivery_Dataset501_TDDTeethBinary2D.md
  Dataset501_TDDTeethBinary2D.qc.json
  Dataset501_TDDTeethBinary2D.qc.md
  training_run_status.json
  training_summary.md
inference/
  inference_summary.json
  inference_summary.md
  predict_from_raw_data_args.json
  *.png
report_case_100/
  report.md
  report.html
  summary.json
  review_list.json
  overlay.png
  input_image.png
logs/
  inference_stdout.log
  inference_stderr.log
  report_stdout.log
  report_stderr.log
```

其中最适合给导师看的文件是：

- `mvp_summary.md`：中文总结，包含数据、QC、模型、推理指标、报告产物路径和 MVP 边界。
- `manifest.json`：机器可读的运行清单。
- `inference/inference_summary.json`：推理评估指标。
- `report_case_100/report.html`：病例级报告页面。
- `report_case_100/overlay.png`：预测结果叠加图。

## 5. 成功标准

一次 MVP 运行成功，应满足：

1. 命令正常退出。
2. 输出目录下存在 `manifest.json` 和 `mvp_summary.md`。
3. `inference/inference_summary.json` 存在，且包含测试集指标。
4. `report_case_<case_id>/report.md` 和 `report_case_<case_id>/report.html` 存在。
5. `report_case_<case_id>/overlay.png` 存在。

## 6. 快速复用已有推理结果

如果某次运行中已经生成过 `run-dir/inference/inference_summary.json` 和预测 mask，只想重新生成报告和汇总，可以使用：

```bash
sh mvp_fullflow/run_mvp_fullflow.sh \
  --run-dir artifacts/mvp_runs/my_mvp_run \
  --reuse-inference
```

注意：`--reuse-inference` 只复用指定 run 目录中的推理结果，不会自动复用其他历史目录。

## 7. 当前设计边界

这个 MVP 是一个固定任务的全流程 smoke test。它故意避开真实 LLM 编排，因为目前返修的关键问题正是 intent-to-workflow 需要单独做平台级验证。这里先把工程链路跑通，为后续 Reviewer #4 的平台级验证提供最小稳定基础。
