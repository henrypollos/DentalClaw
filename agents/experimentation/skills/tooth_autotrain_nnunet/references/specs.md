# Spec Format

`dataset_spec` maps directly to `schemas/specs.py:DatasetSpec`.

- `root`: dataset root directory
- `imagesTr`, `labelsTr`: training image and label subdirectories
- `imagesVal`, `labelsVal`: validation image and label subdirectories
- `imagesTs`: inference/test image subdirectory
- `extra`: optional metadata such as `nnunet_dataset_id`, `target_backend`, `nnunet_raw`, `nnunet_preprocessed`, `nnunet_results`, `labelsTs`, `image_domain`, or `label_format`

`task_spec` maps directly to `schemas/specs.py:TaskSpec`.

- `task_id`: experiment name
- `modality`: usually `auto`
- `task_type`: `tooth_segmentation`
- `num_classes`: `1` for binary tooth-vs-background, or `32` for FDI tooth labels excluding background
- `class_names`: for example `["teeth"]` for binary, or `11-18, 21-28, 31-38, 41-48` for 32-class

`budget_spec` maps directly to `schemas/specs.py:BudgetSpec`.

- `max_trials`: hyperparameter search rounds
- `max_epochs_per_trial`: epochs per round; default is `100` unless a smaller smoke-test budget is explicitly requested
- `max_parallel`: currently `1`

For raw nnUNet datasets, point `root` directly at `nnUNet_raw/DatasetXXX_NAME`, and set:

- `extra.target_backend`: `nnunetv2_cli`
- `extra.nnunet_dataset_id`: the numeric `DatasetXXX` id
- `extra.nnunet_raw`: `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_raw`
- `extra.nnunet_preprocessed`: `/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_preprocessed`
- `extra.nnunet_results`: `/data/data2/yiyang/DentalClaw/artifacts/models/nnUNet/nnUNet_results`

For TDD binary tooth segmentation with the default `nnUNetTrainer`, use:

- `assets/task_spec_teeth_binary_nnunetv2.template.json`
- `max_trials: 5` to compare up to five fold-based nnUNet runs
