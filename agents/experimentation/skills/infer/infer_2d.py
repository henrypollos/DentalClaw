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
    parser = argparse.ArgumentParser(description='Compatibility wrapper for 2D tooth inference.')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--backbone', default='nnunet2d_tiny')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    dataset_spec = DatasetSpec(root=str(input_dir.parent), imagesTs=input_dir.name)
    task_spec = default_teeth32_task_spec()
    skill = ToothSegmentationSkill()
    skill.analyze_dataset(dataset_spec, task_spec)
    result = skill.run_inference(
        {
            'model_path': args.model_path,
            'dataset_spec': dataset_spec,
            'task_spec': task_spec,
            'input_dir': args.input_dir,
            'num_classes': task_spec.num_classes,
        },
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
