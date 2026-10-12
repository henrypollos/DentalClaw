import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from platform_mvp import run_platform_mvp as platform
from platform_mvp.tdd_model_binding import resolve_model_binding, verify_model_binding


class RuntimeEntrypointTests(unittest.TestCase):
    def test_launcher_supports_specialized_parsers(self):
        env = dict(os.environ, RUNNER_PYTHON=sys.executable, DENTALCLAW_PYTHON=sys.executable)
        for flag in ('--next-version', '--data-services'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run(
                    ['sh', str(platform.THIS_DIR / 'run_platform_mvp.sh'), flag, '--help'],
                    cwd=cwd, env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--out-dir', result.stdout)

    def test_launcher_works_outside_repo_without_conda_home(self):
        env = dict(os.environ)
        for key in ('CONDA_HOME', 'CONDA_PREFIX', 'RUNNER_PYTHON', 'DENTALCLAW_PYTHON'):
            env.pop(key, None)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(['sh', str(platform.THIS_DIR / 'run_platform_mvp.sh'), '--help'],
                                    cwd=cwd, env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--offline-planner', result.stdout)

    def test_missing_explicit_interpreter_fails_with_actionable_message(self):
        env = dict(os.environ, RUNNER_PYTHON='/absent/dentalclaw/python', DENTALCLAW_PYTHON=sys.executable)
        result = subprocess.run(['sh', str(platform.THIS_DIR / 'run_platform_mvp.sh'), '--help'],
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Set RUNNER_PYTHON and DENTALCLAW_PYTHON', result.stderr)

    def test_offline_planner_never_calls_agent_or_search(self):
        with tempfile.TemporaryDirectory() as cwd, patch.object(platform, '_agent_decide') as agent, \
                patch.object(platform, '_do_web_search') as search:
            plan = platform.build_plan(intent='Run binary tooth segmentation inference on TDD.',
                                       registry_path=platform.DEFAULT_REGISTRY, case_id='100',
                                       run_dir=Path(cwd), reuse_fullflow_run=None, agent_enabled=False)
        agent.assert_not_called(); search.assert_not_called()
        self.assertEqual(plan['agent_call']['attempt_count'], 0)
        self.assertEqual(plan['decision_source'], 'offline_registry_requested')
        self.assertFalse(plan['web_search']['performed'])

    def test_ownership_with_modality_digits_is_preserved(self):
        parsed = platform.parse_intent('Train a binary tooth segmentation model on my own 2D raster data.')
        self.assertEqual(parsed['dataset_source'], 'user_supplied')
        self.assertEqual(parsed['dataset'], 'Private2D')
        self.assertEqual(parsed['mode'], 'private_train')

    def test_case_only_and_reuse_cannot_share_a_cohort_summary(self):
        with patch.object(sys, 'argv', ['platform', '--intent', 'TDD segmentation',
                                      '--tdd-case-only', '--reuse-inference']), \
                patch('sys.stderr'), self.assertRaises(SystemExit) as error:
            platform.parse_args()
        self.assertEqual(error.exception.code, 2)


class ModelBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / 'run/exp_001/Trainer__Plans__2d'
        fold = self.model / 'fold_all'; fold.mkdir(parents=True)
        self.recorded = fold / 'checkpoint_best.pth'; self.recorded.write_bytes(b'trusted fixture, no model loading')
        for name in ('plans.json', 'dataset.json'):
            (self.model / name).write_text('{}')
        self.export = self.root / 'export.pth'; self.export.write_bytes(self.recorded.read_bytes())
        self.summary = self.root / 'run/report/summary.json'; self.summary.parent.mkdir()
        self.summary.write_text(json.dumps({'best_experiment': {
            'best_model_path': str(self.recorded),
            'config': {'fold': 'all', 'configuration': '2d', 'nnunet_trainer': 'Trainer'}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def bind(self):
        return resolve_model_binding(self.export, self.root / 'run')

    def test_matching_export_binds_best_checkpoint_and_snapshot(self):
        binding = self.bind(); verify_model_binding(binding, self.export)
        self.assertEqual(binding['checkpoint_name'], 'checkpoint_best.pth')
        self.assertEqual(binding['model_folder'], str(self.model))

    def test_different_export_cannot_silently_use_snapshot_weights(self):
        self.export.write_bytes(b'different weights')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.bind()

    def test_missing_model_metadata_is_rejected(self):
        (self.model / 'plans.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.bind()

    def test_bound_metadata_changed_before_launch_is_rejected(self):
        binding = self.bind()
        (self.model / 'dataset.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_model_binding(binding, self.export)


if __name__ == '__main__':
    unittest.main()
