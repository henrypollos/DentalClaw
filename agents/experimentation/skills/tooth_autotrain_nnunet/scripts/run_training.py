import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

from skills.auto_train_infer_skill import _launch_detached_controller, _parse_cli_args, run_training_workflow

def main():
    args = _parse_cli_args()
    if args.detach:
        print(json.dumps(_launch_detached_controller(args), ensure_ascii=False, indent=2))
        return
    result = run_training_workflow(
        dataset_spec_path=args.dataset_spec,
        task_spec_path=args.task_spec,
        budget_spec_path=args.budget_spec,
        workspace=args.workspace,
    )
    print(json.dumps({
        'workspace': args.workspace,
        'best_model_path': result['best_model']['best_model_path'],
        'best_mean_dice': result['best_model']['metrics'].get('mean_dice'),
        'report_dir': str(Path(args.workspace) / 'report'),
        'inference_dir': str(Path(args.workspace) / 'best_inference'),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
