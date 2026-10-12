"""Inspect detached training without confusing launch success with completion.

CLI refreshes only the requested run's execution_result.json. Historical evidence
is never updated unless that run is explicitly requested.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def inspect_training_status(workspace: Path, launcher_returncode: int = 0) -> dict:
    workspace = Path(workspace).resolve()
    state = _read(workspace / 'run_status.json')
    launcher = _read(workspace / 'launcher_status.json')
    result = {'status': 'submitted', 'training_workspace': str(workspace),
              'controller_status': state.get('status'), 'launcher_status': launcher.get('status'),
              'status_source': str(workspace / 'run_status.json')}
    if state.get('status') in {'failed', 'interrupted'}:
        return dict(result, status=state['status'], reason=state.get('error', 'Controller stopped'))
    if state.get('status') == 'completed':
        summary = _read(workspace / 'run_summary.json')
        model = summary.get('best_model_export_path') or state.get('best_model_path')
        model_path = Path(model) if model else workspace / '__missing_model__'
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        required = [workspace / 'run_summary.json', workspace / 'main_handoff.json',
                    workspace / 'report/summary.json', workspace / 'best_inference/inference_summary.json', model_path]
        missing = [str(p) for p in required if not _nonempty(p)]
        inference_dir = workspace / 'best_inference'
        mask_dir = inference_dir / 'masks' if (inference_dir / 'masks').is_dir() else inference_dir
        masks = [p for p in mask_dir.glob('*.png') if _nonempty(p)]
        if not masks:
            missing.append('nonempty inference masks')
        for mask in masks:
            try:
                with Image.open(mask) as img:
                    img.verify()
            except (OSError, ValueError):
                missing.append('unreadable prediction: ' + str(mask))
        inference = _read(workspace / 'best_inference/inference_summary.json')
        declared_count = inference.get('num_cases') or inference.get('case_count')
        if declared_count is not None and declared_count != len(masks):
            missing.append('prediction count disagrees with inference summary')
        manifest = _read(workspace.parent / 'input_manifest.json')
        expected_test = [x for x in manifest.get('files', []) if x.get('split') == 'test' and x.get('kind') == 'image']
        if expected_test and len(expected_test) != len(masks):
            missing.append('prediction count disagrees with input manifest')
        if not summary or not _read(workspace / 'report/summary.json') or not _read(workspace / 'best_inference/inference_summary.json'):
            missing.append('parseable result summaries')
        if missing:
            return dict(result, status='incomplete_artifacts', missing_artifacts=missing)
        return dict(result, status='completed', best_model_path=str(model_path), prediction_count=len(masks),
                    actual_backend=(summary.get('best_model') or {}).get('config', {}).get('backend'),
                    run_summary=str(workspace / 'run_summary.json'))
    if launcher_returncode != 0 or launcher.get('status') in {'failed', 'interrupted'}:
        return dict(result, status='interrupted' if launcher.get('status') == 'interrupted' else 'failed',
                    reason=launcher.get('error', 'Training launcher failed'))
    pid = launcher.get('pid') or state.get('controller_pid')
    if pid:
        try:
            os.kill(int(pid), 0)
            stat = Path('/proc') / str(pid) / 'stat'
            if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                raise ProcessLookupError()
            return dict(result, status='running', controller_pid=pid)
        except ProcessLookupError:
            return dict(result, status='failed', reason='Controller exited without a verified terminal state')
        except (ValueError, PermissionError):
            return dict(result, status='unknown', reason='Controller liveness could not be established')
    # A running status without a live controller reference is not proof of activity.
    if state or launcher:
        return dict(result, status='unknown', reason='No controller PID or complete terminal evidence')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir / 'execution_result.json'
    result = _read(path)
    if not result.get('training_workspace'):
        raise ValueError('execution_result.json must identify a training_workspace')
    workspace = Path(result['training_workspace'])
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    previous = dict(result)
    result.update(inspect_training_status(workspace, result.get('returncode', 0)))
    with (args.run_dir / 'execution_status_history.jsonl').open('a') as history:
        history.write(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(), 'previous': previous, 'current': result}, ensure_ascii=False) + '\n')
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
