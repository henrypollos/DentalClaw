import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schemas.specs import DatasetSpec
from skills.tooth_segmentation_skill import ToothSegmentationSkill, default_teeth32_task_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Compatibility wrapper for one-shot 2D tooth training.')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--backbone', default='nnunet2d_tiny')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_spec = DatasetSpec(root=args.data_root)
    task_spec = default_teeth32_task_spec()
    skill = ToothSegmentationSkill()
    dataset_spec, preprocess_info = skill.preprocess_dataset(dataset_spec, task_spec, args.output_dir)
    skill.set_preprocess_info(preprocess_info)
    skill.analyze_dataset(dataset_spec, task_spec)
    record = skill.run_training(
        dataset_spec=dataset_spec,
        task_spec=task_spec,
        exp_config={
            'backend': 'builtin_nnunet_style_2d',
            'model_name': args.backbone,
            'base_channels': 16 if ('tiny' in args.backbone or '18' in args.backbone) else 24,
            'depth': 4,
            'img_size': args.img_size,
            'batch_size': 2 if args.img_size <= 512 else 1,
            'lr': args.lr,
            'weight_decay': 1e-4,
            'epochs': args.epochs,
            'seed': 2026,
        },
        work_dir=args.output_dir,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
