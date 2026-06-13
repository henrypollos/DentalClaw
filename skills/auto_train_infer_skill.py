import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.tooth_segmentation_skill import _dataset_signature, _sanitize_name

ARTIFACT_ROOT = REPO_ROOT / 'artifacts'
TRAINING_REPORT_ROOT = REPO_ROOT / 'artifacts' / 'results' / 'reports' / 'training_runs'
LEGACY_TRAINING_RUN_ROOT = REPO_ROOT / 'artifacts' / 'results' / 'training_runs'
LAUNCHER_STATUS_FILENAME = 'launcher_status.json'
CONTROLLER_STDOUT_FILENAME = 'controller_stdout.log'
CONTROLLER_STDERR_FILENAME = 'controller_stderr.log'


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _pid_is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _launcher_paths(workspace: str) -> Dict[str, Path]:
    workspace_path = Path(workspace)
    return {
        'workspace': workspace_path,
        'status': workspace_path / LAUNCHER_STATUS_FILENAME,
        'stdout': workspace_path / CONTROLLER_STDOUT_FILENAME,
        'stderr': workspace_path / CONTROLLER_STDERR_FILENAME,
    }


def _monitor_paths(workspace: str) -> Dict[str, str]:
    workspace_path = Path(workspace).resolve()
    return {
        'workspace': str(workspace_path),
        'launcher_status_path': str(workspace_path / LAUNCHER_STATUS_FILENAME),
        'run_status_path': str(workspace_path / 'run_status.json'),
        'search_events_path': str(workspace_path / 'search_events.jsonl'),
        'history_path': str(workspace_path / 'history.json'),
        'search_strategy_path': str(workspace_path / 'search_strategy.json'),
        'run_summary_path': str(workspace_path / 'run_summary.json'),
        'main_handoff_path': str(workspace_path / 'main_handoff.md'),
        'controller_stdout_log': str(workspace_path / CONTROLLER_STDOUT_FILENAME),
        'controller_stderr_log': str(workspace_path / CONTROLLER_STDERR_FILENAME),
    }


def _assert_workspace_allowed(workspace: str) -> Path:
    workspace_path = Path(workspace).resolve()
    try:
        workspace_path.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            'Training workspaces must live under {} so that artifacts and status files are centralized.'.format(
                ARTIFACT_ROOT
            )
        ) from exc
    try:
        workspace_path.relative_to(LEGACY_TRAINING_RUN_ROOT.resolve())
    except ValueError:
        return workspace_path
    raise RuntimeError(
        'Refusing to use legacy launcher workspace under {}. '
        'Use a maintained workspace outside the guarded training_runs directory.'.format(
            LEGACY_TRAINING_RUN_ROOT
        )
    )


def _parse_cli_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Run or launch the DentalClaw nnUNet training workflow.')
    parser.add_argument('--dataset-spec', required=True)
    parser.add_argument('--task-spec', required=True)
    parser.add_argument('--budget-spec', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument(
        '--detach',
        action='store_true',
        help='Launch a durable background controller and return immediately with monitor paths.',
    )
    parser.add_argument(
        '--foreground',
        action='store_true',
        help='Run in the current process and wait for completion.',
    )
    parser.add_argument(
        '--worker',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


class AutoTrainInferenceSkill:
    def __init__(self, task_skill):
        self.task_skill = task_skill

    def _utc_now_iso(self):
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _memory_path(self, dataset_spec, task_spec, workspace):
        signature = _dataset_signature(dataset_spec, task_spec)
        name = _sanitize_name(task_spec.task_id)
        return Path(workspace).parent / 'memory_bank' / '{}_{}.json'.format(name, signature)

    def _load_memory(self, dataset_spec, task_spec, workspace):
        memory_path = self._memory_path(dataset_spec, task_spec, workspace)
        if not memory_path.exists():
            return memory_path, []
        data = json.loads(memory_path.read_text(encoding='utf-8'))
        return memory_path, data.get('history', [])

    def _save_memory(self, memory_path, dataset_spec, task_spec, history):
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'task_id': task_spec.task_id,
            'dataset_root': dataset_spec.root,
            'history': history,
            'num_records': len(history),
        }
        memory_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    def _usable_records(self, history):
        usable = []
        for record in history:
            model_path = record.get('best_model_path')
            if model_path and Path(model_path).exists():
                usable.append(record)
        return usable

    def _write_history(self, history, workspace):
        Path(workspace, 'history.json').write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

    def _record_search_reflection(self, history, planned_experiments):
        if not history:
            return
        best_record = self._best_record_or_none(history)
        history[-1]['search_reflection'] = {
            'recorded_at': self._utc_now_iso(),
            'completed_trials': len(history),
            'best_so_far': {
                'exp_id': best_record.get('exp_id'),
                'mean_dice': (best_record.get('metrics') or {}).get('mean_dice'),
                'mean_iou': (best_record.get('metrics') or {}).get('mean_iou'),
            } if best_record else None,
            'next_trials': [self._config_brief(config) for config in planned_experiments],
        }

    def _persist_trial_reasoning(self, workspace, dataset_spec, task_spec, memory_path, memory_history, history):
        self._write_history(history, workspace)
        self._save_memory(memory_path, dataset_spec, task_spec, memory_history + history)

    def _gpu_runtime_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            'captured_at': self._utc_now_iso(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpus': [],
        }
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                parts = [item.strip() for item in line.split(',')]
                if len(parts) != 5:
                    continue
                snapshot['gpus'].append({
                    'index': parts[0],
                    'name': parts[1],
                    'memory_used_mb': parts[2],
                    'memory_total_mb': parts[3],
                    'utilization_gpu_pct': parts[4],
                })
        except Exception:
            try:
                import torch

                if torch.cuda.is_available():
                    snapshot['gpus'] = [
                        {
                            'index': idx,
                            'name': torch.cuda.get_device_name(idx),
                        }
                        for idx in range(torch.cuda.device_count())
                    ]
                else:
                    snapshot['note'] = 'CUDA not available during snapshot capture.'
            except Exception:
                snapshot['note'] = 'GPU snapshot unavailable.'
        return snapshot

    def _config_brief(self, config: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            'backend',
            'trial_name',
            'nnunet_trainer',
            'generated_trainer_name',
            'inherits_from_trainer',
            'trainer_definition_path',
            'configuration',
            'fold',
            'epochs',
            'initial_lr',
            'weight_decay',
            'oversample_foreground_percent',
            'lr_scheduler',
            'lr',
            'img_size',
            'base_channels',
            'batch_size',
            'seed',
            'materialize_trainer_subclass',
            'selected_gpu_index',
            'selected_gpu_name',
            'selection_reason',
        ]
        return {key: config.get(key) for key in keys if key in config}

    def _search_strategy_paths(self, workspace: str) -> Dict[str, Path]:
        workspace_path = Path(workspace)
        return {
            'json': workspace_path / 'search_strategy.json',
            'md': workspace_path / 'search_strategy.md',
            'events': workspace_path / 'search_events.jsonl',
        }

    def _write_search_strategy(
        self,
        workspace: str,
        dataset_spec,
        task_spec,
        budget_spec,
        dataset_info: Dict[str, Any],
        planned_experiments: List[Dict[str, Any]],
        completed_history: List[Dict[str, Any]],
        timing_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        paths = self._search_strategy_paths(workspace)
        payload = {
            'generated_at': self._utc_now_iso(),
            'task_id': task_spec.task_id,
            'dataset_root': dataset_spec.root,
            'recommended_backend': dataset_info.get('recommended_backend'),
            'detected_dimension': dataset_info.get('detected_dimension'),
            'max_trials': budget_spec.max_trials,
            'max_epochs_per_trial': budget_spec.max_epochs_per_trial,
            'planned_experiments': [self._config_brief(config) for config in planned_experiments],
            'completed_experiments': [
                {
                    'exp_id': record.get('exp_id'),
                    'status': record.get('status'),
                    'model_name': record.get('model_name'),
                    'config': self._config_brief(record.get('config', {})),
                    'timing': record.get('timing'),
                    'notes': record.get('notes'),
                    'error': record.get('error'),
                    'metrics': {
                        'mean_dice': (record.get('metrics') or {}).get('mean_dice'),
                        'mean_hd95': (record.get('metrics') or {}).get('mean_hd95'),
                        'mean_iou': (record.get('metrics') or {}).get('mean_iou'),
                        'pixel_accuracy': (record.get('metrics') or {}).get('pixel_accuracy'),
                    },
                }
                for record in completed_history
            ],
            'timing_summary': timing_summary or {},
        }
        paths['json'].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

        lines = [
            '# Hyperparameter Search Strategy',
            '',
            '- Task: {}'.format(task_spec.task_id),
            '- Dataset root: {}'.format(dataset_spec.root),
            '- Recommended backend: {}'.format(dataset_info.get('recommended_backend') or 'unknown'),
            '- Detected dimension: {}'.format(dataset_info.get('detected_dimension') or 'unknown'),
            '- Max trials: {}'.format(budget_spec.max_trials),
            '- Max epochs per trial: {}'.format(budget_spec.max_epochs_per_trial),
        ]
        if timing_summary:
            lines.extend([
                '- Workflow started at: {}'.format(timing_summary.get('started_at') or 'n/a'),
                '- Workflow duration (s): {}'.format(timing_summary.get('total_duration_seconds') or 'n/a'),
            ])
        lines.extend([
            '',
            '## Search Rationale',
            '',
        ])
        if dataset_info.get('recommended_backend') == 'nnUNetv2_cli':
            lines.extend([
                '- The workflow is using nnU-Net CLI with `fold=all` for every trial instead of five-fold cross-validation.',
                '- Trial variation comes from a DentalClaw custom trainer search over optimizer and sampling hyperparameters such as learning rate, weight decay, oversampling, and LR schedule.',
                '- DentalClaw also picks the least-busy visible GPU before each trial and records the chosen device in the command artifacts.',
            ])
        else:
            lines.extend([
                '- The workflow is using the built-in 2D training backend.',
                '- The search varies learning rate, image size, channels, regularization, and seed within the configured budget.',
            ])
        lines.extend(['', '## Planned Trials', ''])
        if planned_experiments:
            for idx, config in enumerate(planned_experiments, start=1):
                brief = self._config_brief(config)
                lines.append(
                    '- Trial {}: {} | reason={}'.format(
                        idx,
                        json.dumps({key: value for key, value in brief.items() if key != 'selection_reason'}, ensure_ascii=False),
                        brief.get('selection_reason', 'n/a'),
                    )
                )
        else:
            lines.append('- None')
        lines.extend(['', '## Attempted Trials', ''])
        if completed_history:
            for record in completed_history:
                metrics = record.get('metrics') or {}
                lines.append(
                    '- {}: status={} | dice={} | iou={} | hd95={} | duration_s={} | reason={} | config={}'.format(
                        record.get('exp_id', 'unknown'),
                        record.get('status', 'unknown'),
                        'n/a' if metrics.get('mean_dice') is None else '{:.4f}'.format(metrics['mean_dice']),
                        'n/a' if metrics.get('mean_iou') is None else '{:.4f}'.format(metrics['mean_iou']),
                        'n/a' if metrics.get('mean_hd95') is None else '{:.4f}'.format(metrics['mean_hd95']),
                        (record.get('timing') or {}).get('duration_seconds', 'n/a'),
                        (record.get('config') or {}).get('selection_reason', 'n/a'),
                        json.dumps(self._config_brief(record.get('config', {})), ensure_ascii=False),
                    )
                )
                if record.get('error'):
                    lines.append('- {} error: {}'.format(record.get('exp_id', 'unknown'), record.get('error')))
        else:
            lines.append('- None yet')
        paths['md'].write_text('\n'.join(lines), encoding='utf-8')
        return {key: str(value) for key, value in paths.items()}

    def _append_search_event(self, workspace: str, payload: Dict[str, Any]) -> None:
        events_path = self._search_strategy_paths(workspace)['events']
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'timestamp': self._utc_now_iso(), **payload}, ensure_ascii=False) + '\n')

    def _write_run_status(self, workspace, payload):
        status_path = Path(workspace) / 'run_status.json'
        merged = {
            'updated_at': self._utc_now_iso(),
            **payload,
        }
        status_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')

    def _build_main_handoff(self, workspace, dataset_spec, task_spec, result):
        workspace_path = Path(workspace)
        best_model = result.get('best_model') or {}
        best_metrics = best_model.get('metrics') or {}
        artifacts = best_model.get('artifacts') or {}
        inference = result.get('inference') or {}
        search_artifacts = result.get('search_strategy') or {}
        report_key = '{}_{}'.format(_sanitize_name(task_spec.task_id), _sanitize_name(workspace_path.name))
        TRAINING_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        artifact_curve_path = None
        training_curve_path = artifacts.get('training_curve_path')
        if training_curve_path and Path(training_curve_path).is_file():
            artifact_curve_path = TRAINING_REPORT_ROOT / '{}_progress.png'.format(report_key)
            shutil.copy2(training_curve_path, artifact_curve_path)
        handoff = {
            'generated_at': self._utc_now_iso(),
            'task_id': task_spec.task_id,
            'dataset_root': dataset_spec.root,
            'workspace': str(workspace_path),
            'workspace_run_summary': str(workspace_path / 'run_summary.json'),
            'report_summary_json': str(workspace_path / 'report' / 'summary.json'),
            'report_summary_md': str(workspace_path / 'report' / 'summary.md'),
            'best_model_path': result.get('best_model_export_path') or best_model.get('best_model_path'),
            'best_experiment_id': best_model.get('exp_id'),
            'trainer': best_model.get('config', {}).get('nnunet_trainer'),
            'requested_epochs': best_model.get('config', {}).get('epochs'),
            'training_curve_path': training_curve_path,
            'artifact_training_curve_path': str(artifact_curve_path) if artifact_curve_path else None,
            'validation_summary_path': artifacts.get('validation_summary_path'),
            'train_command_path': artifacts.get('train_command_path'),
            'preprocess_command_path': artifacts.get('preprocess_command_path'),
            'validation_mean_dice': best_metrics.get('mean_dice'),
            'validation_mean_hd95': best_metrics.get('mean_hd95'),
            'validation_mean_iou': best_metrics.get('mean_iou'),
            'validation_pixel_accuracy': best_metrics.get('pixel_accuracy'),
            'test_inference_summary_json': inference.get('summary_json_path'),
            'test_inference_summary_md': inference.get('summary_md_path'),
            'test_overlay_dir': inference.get('overlay_dir'),
            'test_prediction_dir': inference.get('output_dir'),
            'test_num_evaluated_cases': inference.get('num_evaluated_cases'),
            'test_mean_dice': inference.get('mean_dice'),
            'test_mean_hd95': inference.get('mean_hd95'),
            'test_mean_iou': inference.get('mean_iou'),
            'test_pixel_accuracy': inference.get('pixel_accuracy'),
            'search_strategy_json': search_artifacts.get('json'),
            'search_strategy_md': search_artifacts.get('md'),
            'search_events_path': search_artifacts.get('events'),
            'trial_history_csv': str(workspace_path / 'report' / 'history.csv'),
            'workflow_started_at': (result.get('timing') or {}).get('started_at'),
            'workflow_completed_at': (result.get('timing') or {}).get('completed_at'),
            'workflow_duration_seconds': (result.get('timing') or {}).get('total_duration_seconds'),
            'preprocess_duration_seconds': ((result.get('timing') or {}).get('stage_durations') or {}).get('preprocessing'),
            'dataset_analysis_duration_seconds': ((result.get('timing') or {}).get('stage_durations') or {}).get('analyzing_dataset'),
            'training_duration_seconds': ((result.get('timing') or {}).get('stage_durations') or {}).get('training'),
            'inference_duration_seconds': ((result.get('timing') or {}).get('stage_durations') or {}).get('inference'),
            'gpu_snapshot': (result.get('timing') or {}).get('gpu_snapshot'),
        }

        lines = [
            '# Main Handoff',
            '',
            '- Task: {}'.format(handoff['task_id']),
            '- Dataset root: {}'.format(handoff['dataset_root']),
            '- Workspace: {}'.format(handoff['workspace']),
            '- Best experiment: {}'.format(handoff.get('best_experiment_id') or 'n/a'),
            '- Best model path: {}'.format(handoff.get('best_model_path') or 'n/a'),
            '- Trainer: {}'.format(handoff.get('trainer') or 'n/a'),
            '- Requested epochs: {}'.format(handoff.get('requested_epochs') or 'n/a'),
            '- Workflow started at: {}'.format(handoff.get('workflow_started_at') or 'n/a'),
            '- Workflow completed at: {}'.format(handoff.get('workflow_completed_at') or 'n/a'),
            '- Workflow duration (s): {}'.format(handoff.get('workflow_duration_seconds') or 'n/a'),
            '- Preprocess duration (s): {}'.format(handoff.get('preprocess_duration_seconds') or 'n/a'),
            '- Dataset analysis duration (s): {}'.format(handoff.get('dataset_analysis_duration_seconds') or 'n/a'),
            '- Training duration (s): {}'.format(handoff.get('training_duration_seconds') or 'n/a'),
            '- Inference duration (s): {}'.format(handoff.get('inference_duration_seconds') or 'n/a'),
            '- Training curve: {}'.format(handoff.get('training_curve_path') or 'n/a'),
            '- Training curve copy: {}'.format(handoff.get('artifact_training_curve_path') or 'n/a'),
            '- Validation summary: {}'.format(handoff.get('validation_summary_path') or 'n/a'),
            '- Validation mean Dice: {}'.format('n/a' if handoff.get('validation_mean_dice') is None else '{:.4f}'.format(handoff['validation_mean_dice'])),
            '- Validation mean HD95: {}'.format('n/a' if handoff.get('validation_mean_hd95') is None else '{:.4f}'.format(handoff['validation_mean_hd95'])),
            '- Validation pixel accuracy: {}'.format('n/a' if handoff.get('validation_pixel_accuracy') is None else '{:.4f}'.format(handoff['validation_pixel_accuracy'])),
            '- Validation mean IoU: {}'.format('n/a' if handoff.get('validation_mean_iou') is None else '{:.4f}'.format(handoff['validation_mean_iou'])),
            '- Test inference summary: {}'.format(handoff.get('test_inference_summary_md') or 'n/a'),
            '- Test overlays: {}'.format(handoff.get('test_overlay_dir') or 'n/a'),
            '- Test evaluated cases: {}'.format(handoff.get('test_num_evaluated_cases') or 0),
            '- Test mean Dice: {}'.format('n/a' if handoff.get('test_mean_dice') is None else '{:.4f}'.format(handoff['test_mean_dice'])),
            '- Test mean HD95: {}'.format('n/a' if handoff.get('test_mean_hd95') is None else '{:.4f}'.format(handoff['test_mean_hd95'])),
            '- Test mean IoU: {}'.format('n/a' if handoff.get('test_mean_iou') is None else '{:.4f}'.format(handoff['test_mean_iou'])),
            '- Test pixel accuracy: {}'.format('n/a' if handoff.get('test_pixel_accuracy') is None else '{:.4f}'.format(handoff['test_pixel_accuracy'])),
            '',
            '## Core Files',
            '',
            '- Run summary: {}'.format(handoff['workspace_run_summary']),
            '- Report summary: {}'.format(handoff['report_summary_md']),
            '- Trial history CSV: {}'.format(handoff['trial_history_csv']),
            '- Search strategy: {}'.format(handoff.get('search_strategy_md') or 'n/a'),
            '- Search events: {}'.format(handoff.get('search_events_path') or 'n/a'),
            '- Train command: {}'.format(handoff.get('train_command_path') or 'n/a'),
            '- Preprocess command: {}'.format(handoff.get('preprocess_command_path') or 'n/a'),
        ]
        if handoff.get('gpu_snapshot'):
            lines.extend(['', '## GPU Snapshot', ''])
            for gpu in handoff['gpu_snapshot'].get('gpus', []):
                lines.append(
                    '- GPU {}: {} | mem_used={}MB | mem_total={}MB | util={}%%'.format(
                        gpu.get('index'),
                        gpu.get('name'),
                        gpu.get('memory_used_mb', 'n/a'),
                        gpu.get('memory_total_mb', 'n/a'),
                        gpu.get('utilization_gpu_pct', 'n/a'),
                    )
                )
        if handoff.get('artifact_training_curve_path'):
            lines.extend(['', '## Training Curve', '', '![Training Curve]({})'.format(handoff['artifact_training_curve_path'])])

        workspace_handoff_json = workspace_path / 'main_handoff.json'
        workspace_handoff_md = workspace_path / 'main_handoff.md'
        workspace_handoff_json.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding='utf-8')
        workspace_handoff_md.write_text('\n'.join(lines), encoding='utf-8')

        artifact_handoff_json = TRAINING_REPORT_ROOT / '{}.json'.format(report_key)
        artifact_handoff_md = TRAINING_REPORT_ROOT / '{}.md'.format(report_key)
        artifact_handoff_json.write_text(json.dumps({**handoff, 'workspace_handoff_md': str(workspace_handoff_md)}, indent=2, ensure_ascii=False), encoding='utf-8')
        artifact_handoff_md.write_text('\n'.join(lines), encoding='utf-8')
        return {
            'workspace_json': str(workspace_handoff_json),
            'workspace_md': str(workspace_handoff_md),
            'artifact_json': str(artifact_handoff_json),
            'artifact_md': str(artifact_handoff_md),
        }

    def _timing_summary(self, workflow_started_at, workflow_timer, stage_durations):
        return {
            'started_at': workflow_started_at,
            'stage_durations': stage_durations,
            'total_duration_seconds': round(time.perf_counter() - workflow_timer, 3),
        }

    def _best_record_or_none(self, history):
        completed = [record for record in history if record.get('status') == 'completed']
        if not completed:
            return None
        return self.task_skill.select_best_model(completed)

    def _refresh_search_tracking(
        self,
        workspace,
        dataset_spec,
        task_spec,
        budget_spec,
        dataset_info,
        history,
        memory_path,
        memory_history,
        workflow_started_at,
        workflow_timer,
        stage_durations,
        planned_experiments,
    ):
        self._record_search_reflection(history, planned_experiments)
        self._persist_trial_reasoning(workspace, dataset_spec, task_spec, memory_path, memory_history, history)
        search_strategy = self._write_search_strategy(
            workspace=workspace,
            dataset_spec=dataset_spec,
            task_spec=task_spec,
            budget_spec=budget_spec,
            dataset_info=dataset_info,
            planned_experiments=planned_experiments,
            completed_history=history,
            timing_summary=self._timing_summary(workflow_started_at, workflow_timer, stage_durations),
        )
        best_record = self._best_record_or_none(history)
        self._append_search_event(workspace, {
            'event': 'search_reflection',
            'completed_trials': len(history),
            'best_so_far': {
                'exp_id': best_record.get('exp_id'),
                'mean_dice': (best_record.get('metrics') or {}).get('mean_dice'),
                'mean_iou': (best_record.get('metrics') or {}).get('mean_iou'),
            } if best_record else None,
            'next_trials': [self._config_brief(config) for config in planned_experiments],
        })
        return search_strategy

    def _execute_trial(
        self,
        dataset_spec,
        task_spec,
        exp,
        exp_id,
        work_dir,
        workspace,
        budget_spec,
        history,
    ):
        exp_name = 'exp_{:03d}'.format(exp_id)
        self._write_run_status(workspace, {
            'status': 'running',
            'stage': 'training',
            'dataset_root': dataset_spec.root,
            'task_id': task_spec.task_id,
            'max_trials': budget_spec.max_trials,
            'completed_trials': len(history),
            'current_experiment': {
                'exp_id': exp_name,
                'config': exp,
                'work_dir': work_dir,
            },
        })
        self._append_search_event(workspace, {
            'event': 'trial_started',
            'exp_id': exp_name,
            'config': self._config_brief(exp),
        })
        trial_started_at = self._utc_now_iso()
        trial_timer = time.perf_counter()
        failed = False
        try:
            record = self.task_skill.run_training(dataset_spec, task_spec, exp, work_dir)
        except Exception as exc:
            failed = True
            record = {
                'exp_id': exp_name,
                'task_id': task_spec.task_id,
                'model_name': exp.get('model_name') or exp.get('trial_name') or 'failed_experiment',
                'config': dict(exp),
                'best_model_path': '',
                'metrics': {
                    'mean_dice': None,
                    'mean_hd95': None,
                    'mean_iou': None,
                    'pixel_accuracy': None,
                    'per_class_dice': {},
                    'per_class_hd95': {},
                    'per_class_iou': {},
                },
                'work_dir': work_dir,
                'status': 'failed',
                'notes': str(exc),
                'error': str(exc),
                'artifacts': {
                    'train_command_path': str(Path(work_dir) / 'nnunet_command.json'),
                    'preprocess_command_path': str(Path(work_dir) / 'nnunet_preprocess_command.json'),
                },
            }
        trial_duration = round(time.perf_counter() - trial_timer, 3)
        record['timing'] = {
            'started_at': trial_started_at,
            'completed_at': self._utc_now_iso(),
            'duration_seconds': trial_duration,
        }
        record['gpu_snapshot'] = self._gpu_runtime_snapshot()
        history.append(record)
        self._append_search_event(workspace, {
            'event': 'trial_failed' if failed else 'trial_completed',
            'exp_id': record.get('exp_id'),
            'config': self._config_brief(record.get('config', {})),
            'timing': record.get('timing'),
            'gpu_snapshot': record.get('gpu_snapshot'),
            'error': record.get('error'),
            'metrics': {
                'mean_dice': (record.get('metrics') or {}).get('mean_dice'),
                'mean_hd95': (record.get('metrics') or {}).get('mean_hd95'),
                'mean_iou': (record.get('metrics') or {}).get('mean_iou'),
                'pixel_accuracy': (record.get('metrics') or {}).get('pixel_accuracy'),
            },
        })
        return record, trial_duration

    def run(self, dataset_spec, task_spec, budget_spec, workspace):
        os.makedirs(workspace, exist_ok=True)
        workflow_started_at = self._utc_now_iso()
        workflow_timer = time.perf_counter()
        stage_durations = {
            'preprocessing': 0.0,
            'analyzing_dataset': 0.0,
            'training': 0.0,
            'inference': 0.0,
        }
        self._write_run_status(workspace, {
            'status': 'running',
            'stage': 'initializing',
            'dataset_root': dataset_spec.root,
            'task_id': task_spec.task_id,
            'max_trials': budget_spec.max_trials,
            'completed_trials': 0,
            'started_at': workflow_started_at,
        })
        try:
            memory_path, memory_history = self._load_memory(dataset_spec, task_spec, workspace)
            if hasattr(self.task_skill, 'set_experiment_memory'):
                self.task_skill.set_experiment_memory(memory_history)

            preprocess_info = None
            working_dataset_spec = dataset_spec
            if hasattr(self.task_skill, 'preprocess_dataset'):
                self._write_run_status(workspace, {
                    'status': 'running',
                    'stage': 'preprocessing',
                    'dataset_root': dataset_spec.root,
                    'task_id': task_spec.task_id,
                    'max_trials': budget_spec.max_trials,
                    'completed_trials': 0,
                })
                preprocess_timer = time.perf_counter()
                working_dataset_spec, preprocess_info = self.task_skill.preprocess_dataset(dataset_spec, task_spec, workspace)
                stage_durations['preprocessing'] += round(time.perf_counter() - preprocess_timer, 3)
                if hasattr(self.task_skill, 'set_preprocess_info'):
                    self.task_skill.set_preprocess_info(preprocess_info)

            self._write_run_status(workspace, {
                'status': 'running',
                'stage': 'analyzing_dataset',
                'dataset_root': working_dataset_spec.root,
                'task_id': task_spec.task_id,
                'max_trials': budget_spec.max_trials,
                'completed_trials': 0,
            })
            analyze_timer = time.perf_counter()
            dataset_info = self.task_skill.analyze_dataset(working_dataset_spec, task_spec)
            stage_durations['analyzing_dataset'] += round(time.perf_counter() - analyze_timer, 3)
            history = []
            exp_id = 0

            if not memory_history and hasattr(self.task_skill, 'bootstrap_existing_experiments'):
                bootstrap_records = self.task_skill.bootstrap_existing_experiments(
                    working_dataset_spec,
                    task_spec,
                    workspace,
                )
                for record in bootstrap_records:
                    history.append(record)
                    self._append_search_event(workspace, {
                        'event': 'trial_bootstrapped',
                        'exp_id': record.get('exp_id'),
                        'config': self._config_brief(record.get('config', {})),
                        'timing': record.get('timing'),
                        'metrics': {
                            'mean_dice': (record.get('metrics') or {}).get('mean_dice'),
                            'mean_hd95': (record.get('metrics') or {}).get('mean_hd95'),
                            'mean_iou': (record.get('metrics') or {}).get('mean_iou'),
                            'pixel_accuracy': (record.get('metrics') or {}).get('pixel_accuracy'),
                        },
                        'notes': record.get('notes'),
                    })

            if history:
                remain = budget_spec.max_trials - len(history)
                initial_experiments = self.task_skill.suggest_next_experiments(history, budget_spec)[:remain]
            else:
                initial_experiments = self.task_skill.generate_initial_experiments(working_dataset_spec, task_spec, budget_spec)
            search_strategy = self._write_search_strategy(
                workspace=workspace,
                dataset_spec=working_dataset_spec,
                task_spec=task_spec,
                budget_spec=budget_spec,
                dataset_info=dataset_info,
                planned_experiments=initial_experiments,
                completed_history=history,
                timing_summary={
                    'started_at': workflow_started_at,
                    'stage_durations': stage_durations,
                },
            )
            for exp in initial_experiments:
                exp_id += 1
                work_dir = os.path.join(workspace, 'exp_{:03d}'.format(exp_id))
                record, trial_duration = self._execute_trial(
                    dataset_spec=working_dataset_spec,
                    task_spec=task_spec,
                    exp=exp,
                    exp_id=exp_id,
                    work_dir=work_dir,
                    workspace=workspace,
                    budget_spec=budget_spec,
                    history=history,
                )
                stage_durations['training'] += trial_duration
                remain = budget_spec.max_trials - len(history)
                next_preview = self.task_skill.suggest_next_experiments(history, budget_spec)[:remain]
                search_strategy = self._refresh_search_tracking(
                    workspace=workspace,
                    dataset_spec=working_dataset_spec,
                    task_spec=task_spec,
                    budget_spec=budget_spec,
                    dataset_info=dataset_info,
                    history=history,
                    memory_path=memory_path,
                    memory_history=memory_history,
                    workflow_started_at=workflow_started_at,
                    workflow_timer=workflow_timer,
                    stage_durations=stage_durations,
                    planned_experiments=next_preview,
                )

            while len(history) < budget_spec.max_trials:
                remain = budget_spec.max_trials - len(history)
                next_exps = self.task_skill.suggest_next_experiments(history, budget_spec)[:remain]
                if not next_exps:
                    break
                for exp in next_exps:
                    exp_id += 1
                    work_dir = os.path.join(workspace, 'exp_{:03d}'.format(exp_id))
                    record, trial_duration = self._execute_trial(
                        dataset_spec=working_dataset_spec,
                        task_spec=task_spec,
                        exp=exp,
                        exp_id=exp_id,
                        work_dir=work_dir,
                        workspace=workspace,
                        budget_spec=budget_spec,
                        history=history,
                    )
                    stage_durations['training'] += trial_duration
                    remain_after = budget_spec.max_trials - len(history)
                    next_preview = self.task_skill.suggest_next_experiments(history, budget_spec)[:remain_after]
                    search_strategy = self._refresh_search_tracking(
                        workspace=workspace,
                        dataset_spec=working_dataset_spec,
                        task_spec=task_spec,
                        budget_spec=budget_spec,
                        dataset_info=dataset_info,
                        history=history,
                        memory_path=memory_path,
                        memory_history=memory_history,
                        workflow_started_at=workflow_started_at,
                        workflow_timer=workflow_timer,
                        stage_durations=stage_durations,
                        planned_experiments=next_preview,
                    )
                    if len(history) >= budget_spec.max_trials:
                        break

            search_strategy = self._refresh_search_tracking(
                workspace=workspace,
                dataset_spec=working_dataset_spec,
                task_spec=task_spec,
                budget_spec=budget_spec,
                dataset_info=dataset_info,
                history=history,
                memory_path=memory_path,
                memory_history=memory_history,
                workflow_started_at=workflow_started_at,
                workflow_timer=workflow_timer,
                stage_durations=stage_durations,
                planned_experiments=[],
            )

            self._write_run_status(workspace, {
                'status': 'running',
                'stage': 'selecting_best_model',
                'dataset_root': working_dataset_spec.root,
                'task_id': task_spec.task_id,
                'max_trials': budget_spec.max_trials,
                'completed_trials': len(history),
            })
            current_best = self._best_record_or_none(history)
            if current_best is None:
                raise RuntimeError('All attempted trials failed. See search_events.jsonl and history.json for the per-trial errors.')
            merged_history = memory_history + history
            usable_merged = self._usable_records(merged_history)
            overall_best = self.task_skill.select_best_model(usable_merged or history)

            best_dir = Path(workspace) / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)
            best_src = Path(overall_best['best_model_path'])
            best_export_path = best_src
            if best_src.is_file():
                best_export_path = best_dir / best_src.name
                shutil.copy2(best_src, best_export_path)

            self._write_history(history, workspace)
            self._save_memory(memory_path, dataset_spec, task_spec, merged_history)
            self._write_run_status(workspace, {
                'status': 'running',
                'stage': 'reporting',
                'dataset_root': working_dataset_spec.root,
                'task_id': task_spec.task_id,
                'max_trials': budget_spec.max_trials,
                'completed_trials': len(history),
                'best_model_path': str(best_export_path),
            })
            report = self.task_skill.generate_report(history, os.path.join(workspace, 'report'))
            inference_result = None
            if task_spec.extra.get('run_best_inference', True):
                self._write_run_status(workspace, {
                    'status': 'running',
                    'stage': 'inference',
                    'dataset_root': working_dataset_spec.root,
                    'task_id': task_spec.task_id,
                    'max_trials': budget_spec.max_trials,
                    'completed_trials': len(history),
                    'best_model_path': str(best_export_path),
                })
                best_for_inference = dict(overall_best)
                best_for_inference['best_model_path'] = str(best_export_path)
                inference_timer = time.perf_counter()
                inference_result = self.inference(best_for_inference, working_dataset_spec, task_spec, os.path.join(workspace, 'best_inference'))
                stage_durations['inference'] += round(time.perf_counter() - inference_timer, 3)

            completed_at = self._utc_now_iso()
            total_duration = round(time.perf_counter() - workflow_timer, 3)
            result = {
                'dataset_info': dataset_info,
                'preprocess_info': preprocess_info,
                'history': history,
                'memory_path': str(memory_path),
                'memory_records': len(merged_history),
                'best_model_current_run': current_best,
                'best_model': overall_best,
                'best_model_export_path': str(best_export_path),
                'report': report,
                'inference': inference_result,
                'search_strategy': search_strategy,
                'timing': {
                    'started_at': workflow_started_at,
                    'completed_at': completed_at,
                    'total_duration_seconds': total_duration,
                    'stage_durations': stage_durations,
                    'gpu_snapshot': self._gpu_runtime_snapshot(),
                },
            }
            result['main_handoff'] = self._build_main_handoff(workspace, working_dataset_spec, task_spec, result)
            Path(workspace, 'run_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            self._write_run_status(workspace, {
                'status': 'completed',
                'stage': 'completed',
                'dataset_root': working_dataset_spec.root,
                'task_id': task_spec.task_id,
                'max_trials': budget_spec.max_trials,
                'completed_trials': len(history),
                'best_model_path': str(best_export_path),
                'run_summary_path': str(Path(workspace) / 'run_summary.json'),
                'main_handoff_path': result['main_handoff']['workspace_md'],
                'completed_at': completed_at,
                'total_duration_seconds': total_duration,
                'stage_durations': stage_durations,
            })
            return result
        except Exception as exc:
            self._write_run_status(workspace, {
                'status': 'failed',
                'stage': 'failed',
                'dataset_root': dataset_spec.root,
                'task_id': task_spec.task_id,
                'max_trials': budget_spec.max_trials,
                'error': str(exc),
            })
            raise

    def inference(self, best_record, dataset_spec, task_spec, output_dir, use_tta=True):
        root = Path(dataset_spec.root)
        labels_ts = dataset_spec.extra.get('labelsTs')
        if not labels_ts and (root / 'labelsTs').is_dir():
            labels_ts = 'labelsTs'
        infer_input = {
            'model_path': best_record['best_model_path'],
            'dataset_spec': dataset_spec,
            'task_spec': task_spec,
            'num_classes': task_spec.num_classes,
            'img_size': best_record.get('config', {}).get('img_size', 512),
            'configuration': best_record.get('config', {}).get('configuration'),
            'fold': best_record.get('config', {}).get('fold'),
            'nnunet_trainer': best_record.get('config', {}).get('nnunet_trainer'),
            'nnunet_results_root': (best_record.get('artifacts') or {}).get('nnunet_results_root'),
            'gt_dir': str(root / labels_ts) if labels_ts else None,
            'use_tta': use_tta,
        }
        return self.task_skill.run_inference(infer_input, output_dir)


def run_training_workflow(dataset_spec_path: str, task_spec_path: str, budget_spec_path: str, workspace: str) -> Dict[str, Any]:
    from skills.tooth_segmentation_skill import ToothSegmentationSkill, load_budget_spec, load_dataset_spec, load_task_spec

    dataset_spec = load_dataset_spec(dataset_spec_path)
    task_spec = load_task_spec(task_spec_path)
    budget_spec = load_budget_spec(budget_spec_path)
    orchestrator = AutoTrainInferenceSkill(ToothSegmentationSkill())
    result = orchestrator.run(
        dataset_spec=dataset_spec,
        task_spec=task_spec,
        budget_spec=budget_spec,
        workspace=workspace,
    )
    Path(workspace, 'run_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return result


def _write_launcher_status(workspace: str, payload: Dict[str, Any]) -> None:
    paths = _launcher_paths(workspace)
    _write_json_file(paths['status'], payload)


def _launch_detached_controller(args) -> Dict[str, Any]:
    _assert_workspace_allowed(args.workspace)
    paths = _launcher_paths(args.workspace)
    paths['workspace'].mkdir(parents=True, exist_ok=True)
    if paths['status'].exists():
        existing = _load_json_file(paths['status'])
        existing_pid = existing.get('pid')
        if existing.get('status') in {'starting', 'running'} and _pid_is_running(existing_pid):
            return {
                'status': 'already_running',
                'workspace': args.workspace,
                'pid': existing_pid,
                'launcher_status_path': str(paths['status']),
                'stdout_log': str(paths['stdout']),
                'stderr_log': str(paths['stderr']),
            }

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--dataset-spec', str(Path(args.dataset_spec).resolve()),
        '--task-spec', str(Path(args.task_spec).resolve()),
        '--budget-spec', str(Path(args.budget_spec).resolve()),
        '--workspace', str(Path(args.workspace).resolve()),
        '--worker',
    ]
    stdout_handle = paths['stdout'].open('a', encoding='utf-8')
    stderr_handle = paths['stderr'].open('a', encoding='utf-8')
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env=dict(os.environ),
            text=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    payload = {
        'status': 'starting',
        'pid': process.pid if process is not None else None,
        'workspace': str(Path(args.workspace).resolve()),
        'command': command,
        'started_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'stdout_log': str(paths['stdout']),
        'stderr_log': str(paths['stderr']),
    }
    _write_launcher_status(args.workspace, payload)
    time.sleep(0.5)
    if process is None or process.poll() is not None:
        payload['status'] = 'failed'
        payload['returncode'] = None if process is None else process.returncode
        _write_launcher_status(args.workspace, payload)
        return {
            'status': 'failed_to_launch',
            'workspace': args.workspace,
            'monitor_paths': _monitor_paths(args.workspace),
            'next_action': 'Inspect launcher_status.json and controller logs before retrying. Do not create an ad hoc launcher.',
        }

    payload['status'] = 'running'
    _write_launcher_status(args.workspace, payload)
    return {
        'status': 'launched',
        'workspace': args.workspace,
        'pid': process.pid,
        'monitor_paths': _monitor_paths(args.workspace),
        'supervision_command': '/home/yiyang/miniconda3/envs/nnunetv2/bin/python '
                               '/data/data2/yiyang/DentalClaw/agents/main/skills/supervision-registry/scripts/monitor_training_run.py '
                               '--workspace {}'.format(str(Path(args.workspace).resolve())),
        'next_action': 'Monitor launcher_status.json, run_status.json, search_events.jsonl, and history.json until the workflow reaches completed or failed. Do not replace the workflow with a handwritten launcher.',
    }


def _install_signal_handlers(workspace: str) -> None:
    def _handle_signal(signum, _frame):
        interrupted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        _write_launcher_status(workspace, {
            'status': 'interrupted',
            'pid': os.getpid(),
            'workspace': str(Path(workspace).resolve()),
            'interrupted_at': interrupted_at,
            'signal': signum,
            'stdout_log': str(_launcher_paths(workspace)['stdout']),
            'stderr_log': str(_launcher_paths(workspace)['stderr']),
        })
        _write_json_file(Path(workspace) / 'run_status.json', {
            'updated_at': interrupted_at,
            'status': 'interrupted',
            'stage': 'interrupted',
            'error': 'Training controller received signal {}'.format(signum),
            'controller_pid': os.getpid(),
        })
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _run_worker(args) -> Dict[str, Any]:
    workspace = str(_assert_workspace_allowed(args.workspace))
    _install_signal_handlers(workspace)
    _write_launcher_status(workspace, {
        'status': 'running',
        'pid': os.getpid(),
        'workspace': workspace,
        'started_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'stdout_log': str(_launcher_paths(workspace)['stdout']),
        'stderr_log': str(_launcher_paths(workspace)['stderr']),
    })
    try:
        result = run_training_workflow(
            dataset_spec_path=args.dataset_spec,
            task_spec_path=args.task_spec,
            budget_spec_path=args.budget_spec,
            workspace=workspace,
        )
    except Exception as exc:
        _write_launcher_status(workspace, {
            'status': 'failed',
            'pid': os.getpid(),
            'workspace': workspace,
            'failed_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            'error': str(exc),
            'stdout_log': str(_launcher_paths(workspace)['stdout']),
            'stderr_log': str(_launcher_paths(workspace)['stderr']),
        })
        raise

    _write_launcher_status(workspace, {
        'status': 'completed',
        'pid': os.getpid(),
        'workspace': workspace,
        'completed_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'run_summary_path': str(Path(workspace) / 'run_summary.json'),
        'main_handoff_path': str(Path(workspace) / 'main_handoff.md'),
        'stdout_log': str(_launcher_paths(workspace)['stdout']),
        'stderr_log': str(_launcher_paths(workspace)['stderr']),
    })
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_cli_args(argv)
    detach = args.detach or (not args.foreground and not args.worker)
    if args.worker:
        result = _run_worker(args)
        print(json.dumps({
            'workspace': args.workspace,
            'best_model_path': result['best_model']['best_model_path'],
            'best_mean_dice': result['best_model']['metrics'].get('mean_dice'),
            'report_dir': str(Path(args.workspace) / 'report'),
            'inference_dir': str(Path(args.workspace) / 'best_inference'),
        }, ensure_ascii=False, indent=2))
        return 0
    if detach:
        print(json.dumps(_launch_detached_controller(args), ensure_ascii=False, indent=2))
        return 0

    result = _run_worker(args)
    print(json.dumps({
        'workspace': args.workspace,
        'best_model_path': result['best_model']['best_model_path'],
        'best_mean_dice': result['best_model']['metrics'].get('mean_dice'),
        'report_dir': str(Path(args.workspace) / 'report'),
        'inference_dir': str(Path(args.workspace) / 'best_inference'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
