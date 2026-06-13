import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.tooth_segmentation_skill import ToothSegmentationSkill, load_dataset_spec, load_task_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Run tooth segmentation inference.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dataset-spec', required=True)
    parser.add_argument('--task-spec', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--input-dir', default=None)
    parser.add_argument('--gt-dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_spec = load_dataset_spec(args.dataset_spec)
    task_spec = load_task_spec(args.task_spec)
    skill = ToothSegmentationSkill()
    skill.analyze_dataset(dataset_spec, task_spec)
    result = skill.run_inference({
        'model_path': args.model_path,
        'dataset_spec': dataset_spec,
        'task_spec': task_spec,
        'input_dir': args.input_dir,
        'gt_dir': args.gt_dir,
        'num_classes': task_spec.num_classes,
    }, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
