import json
import os
import subprocess
import sys
from pathlib import Path


class AutoTrainerAgent:
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[3]
        self.workspace = self.repo_root / 'workspace' / 'auto_trainer'
        self.data_root = self.repo_root / 'data' / 'pano2d'
        self.train_script = self.repo_root / 'skills' / 'tooth_autotrain_nnunet' / 'scripts' / 'run_training.py'
        self.python_exe = sys.executable
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _write_specs(self, budget: int, epochs: int):
        dataset_spec = {
            'root': str(self.data_root),
            'imagesTr': 'imagesTr',
            'labelsTr': 'labelsTr',
            'imagesVal': 'imagesVal',
            'labelsVal': 'labelsVal',
            'imagesTs': 'imagesTs',
            'extra': {
                'image_domain': 'dental_panoramic',
                'label_format': 'png_index_mask',
                'preprocess': {
                    'normalize_percentiles': [1.0, 99.0],
                    'use_clahe': True,
                    'clahe_clip_limit': 2.0,
                    'clahe_tile_grid_size': 8,
                    'median_blur_ksize': 3,
                },
            },
        }
        task_spec = {
            'task_id': 'Teeth1to32Panoramic',
            'modality': 'auto',
            'task_type': 'tooth_segmentation',
            'num_classes': 32,
            'class_names': [
                '11', '12', '13', '14', '15', '16', '17', '18',
                '21', '22', '23', '24', '25', '26', '27', '28',
                '31', '32', '33', '34', '35', '36', '37', '38',
                '41', '42', '43', '44', '45', '46', '47', '48',
            ],
            'primary_metric': 'mean_dice',
            'extra': {
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
                    'learning_rates': [1e-3, 5e-4, 2e-4],
                    'img_sizes': [512, 640, 768],
                    'base_channels': [16, 24, 32],
                    'batch_sizes': [2, 1, 1],
                    'depths': [4],
                    'weight_decays': [1e-4, 5e-5],
                },
            },
        }
        budget_spec = {'max_trials': budget, 'max_epochs_per_trial': epochs, 'max_parallel': 1}
        dataset_path = self.workspace / 'dataset_spec.json'
        task_path = self.workspace / 'task_spec.json'
        budget_path = self.workspace / 'budget_spec.json'
        dataset_path.write_text(json.dumps(dataset_spec, indent=2, ensure_ascii=False), encoding='utf-8')
        task_path.write_text(json.dumps(task_spec, indent=2, ensure_ascii=False), encoding='utf-8')
        budget_path.write_text(json.dumps(budget_spec, indent=2, ensure_ascii=False), encoding='utf-8')
        return dataset_path, task_path, budget_path

    def optimize(self, budget=3, epochs=5):
        dataset_path, task_path, budget_path = self._write_specs(budget=budget, epochs=epochs)
        run_dir = self.workspace / 'latest_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_exe,
            str(self.train_script),
            '--dataset-spec', str(dataset_path),
            '--task-spec', str(task_path),
            '--budget-spec', str(budget_path),
            '--workspace', str(run_dir),
        ]
        print('Auto training start. Budget={}, Epochs={}'.format(budget, epochs))
        subprocess.run(cmd, check=True)
        self.report_best(run_dir)

    def report_best(self, run_dir: Path):
        summary_path = run_dir / 'run_summary.json'
        if not summary_path.exists():
            print('No summary generated.')
            return
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        best = summary['best_model']
        metrics = best.get('metrics', {})
        print('Best Tooth Segmentation Model')
        print('=' * 48)
        print('Experiment:   {}'.format(best['exp_id']))
        print('Model:        {}'.format(best['model_name']))
        print('Mean Dice:    {:.4f}'.format(metrics.get('mean_dice', 0.0)))
        print('Mean HD95:    {:.4f}'.format(metrics.get('mean_hd95', 0.0)))
        print('Pixel Acc:    {:.4f}'.format(metrics.get('pixel_accuracy', 0.0)))
        print('Checkpoint:   {}'.format(summary.get('best_model_export_path', best['best_model_path'])))
        print('Memory File:  {}'.format(summary.get('memory_path')))
        print('Memory Recs:  {}'.format(summary.get('memory_records')))
        print('Report Dir:   {}'.format(run_dir / 'report'))
        print('Inference:    {}'.format(run_dir / 'best_inference'))
        print('=' * 48)


if __name__ == '__main__':
    agent = AutoTrainerAgent()
    budget = int(os.environ.get('DENTALCLAW_BUDGET', '3'))
    epochs = int(os.environ.get('DENTALCLAW_EPOCHS', '5'))
    agent.optimize(budget=budget, epochs=epochs)
