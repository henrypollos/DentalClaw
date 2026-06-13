import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schemas.specs import BudgetSpec, DatasetSpec, TaskSpec
from skills.auto_train_infer_skill import AutoTrainInferenceSkill
from skills.tooth_segmentation_skill import TOOTH_32_FDI_NAMES, ToothSegmentationSkill


def build_dataset_spec(repo_root: Path) -> DatasetSpec:
    return DatasetSpec(
        root=str(repo_root / 'data' / 'pano2d'),
        imagesTr='imagesTr',
        labelsTr='labelsTr',
        imagesVal='imagesVal',
        labelsVal='labelsVal',
        imagesTs='imagesTs',
        extra={
            'image_domain': 'dental_panoramic',
            'label_format': 'png_index_mask',
            'labelsTs': 'labelsVal',
            'preprocess': {
                'normalize_percentiles': [1.0, 99.0],
                'use_clahe': True,
                'clahe_clip_limit': 2.0,
                'clahe_tile_grid_size': 8,
                'median_blur_ksize': 3,
            },
        },
    )


def build_task_spec() -> TaskSpec:
    return TaskSpec(
        task_id='Teeth1to32PanoramicVerify',
        modality='auto',
        task_type='tooth_segmentation',
        num_classes=32,
        class_names=TOOTH_32_FDI_NAMES,
        primary_metric='mean_dice',
        extra={
            'target_backend': 'nnunet_style_2d',
            'run_best_inference': True,
            'preprocess': {
                'normalize_percentiles': [1.0, 99.0],
                'use_clahe': True,
                'clahe_clip_limit': 2.0,
                'clahe_tile_grid_size': 8,
                'median_blur_ksize': 3,
            },
            'training_requirements': {
                'initial_configs': [
                    {
                        'backend': 'builtin_nnunet_style_2d',
                        'model_name': 'verify_128_c8',
                        'base_channels': 8,
                        'depth': 4,
                        'img_size': 128,
                        'batch_size': 2,
                        'lr': 1e-3,
                        'weight_decay': 1e-4,
                        'epochs': 1,
                        'seed': 2026,
                    },
                    {
                        'backend': 'builtin_nnunet_style_2d',
                        'model_name': 'verify_160_c8',
                        'base_channels': 8,
                        'depth': 4,
                        'img_size': 160,
                        'batch_size': 2,
                        'lr': 5e-4,
                        'weight_decay': 5e-5,
                        'epochs': 1,
                        'seed': 2027,
                    },
                ],
                'img_sizes': [128, 160, 192],
                'base_channels': [8, 12, 16],
                'weight_decays': [1e-4, 5e-5],
            },
        },
    )


def run_round(dataset_spec: DatasetSpec, task_spec: TaskSpec, budget_spec: BudgetSpec, workspace: Path):
    orchestrator = AutoTrainInferenceSkill(ToothSegmentationSkill())
    return orchestrator.run(dataset_spec=dataset_spec, task_spec=task_spec, budget_spec=budget_spec, workspace=str(workspace))


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    verify_root = REPO_ROOT / 'workspace' / 'verify_run' / stamp
    dataset_spec = build_dataset_spec(REPO_ROOT)
    task_spec = build_task_spec()
    budget_spec = BudgetSpec(max_trials=2, max_epochs_per_trial=1, max_parallel=1)

    round1 = run_round(dataset_spec, task_spec, budget_spec, verify_root / 'round1')
    round2 = run_round(dataset_spec, task_spec, budget_spec, verify_root / 'round2')

    summary = {
        'verify_root': str(verify_root),
        'round1_best_model_export_path': round1.get('best_model_export_path'),
        'round1_report_dir': str(verify_root / 'round1' / 'report'),
        'round1_inference_dir': str(verify_root / 'round1' / 'best_inference'),
        'round2_best_model_export_path': round2.get('best_model_export_path'),
        'round2_best_model_source_path': round2.get('best_model', {}).get('best_model_path'),
        'round2_memory_path': round2.get('memory_path'),
        'round2_memory_records': round2.get('memory_records'),
        'round2_best_metrics': round2.get('best_model', {}).get('metrics', {}),
        'round2_report_dir': str(verify_root / 'round2' / 'report'),
        'round2_inference_dir': str(verify_root / 'round2' / 'best_inference'),
    }
    (verify_root / 'verify_summary.json').parent.mkdir(parents=True, exist_ok=True)
    (verify_root / 'verify_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
