import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from platform_mvp import run_platform_mvp as platform
from platform_mvp.evaluation_adapter import normalize_bundle
from platform_mvp.route_preflight import preflight


class SharedRoutePreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'raster_package'
        for split, value in [('Tr', 30), ('Val', 70), ('Ts', 110)]:
            for folder in ['images', 'labels']:
                (self.data / (folder + split)).mkdir(parents=True)
            Image.new('L', (16, 16), value).save(self.data / f'images{split}/a.png')
            Image.new('L', (16, 16), 1).save(self.data / f'labels{split}/a.png')
        self.registry = platform.load_registry(platform.DEFAULT_REGISTRY)

    def tearDown(self):
        self.tmp.cleanup()

    def private_plan(self):
        return {'executable': True,
                'selected_method': {'id': 'private_2d_segmentation_train', 'status': 'executable'},
                'intent': {'dataset': 'Private2D', 'task_family': 'segmentation', 'modality': '2d', 'mode': 'private_train'},
                'execution': {}}

    def private_args(self, root=None):
        return Namespace(private_data_root=str(root or self.data), execute=True, dentalclaw_python='python')

    def tdd_fixture(self):
        # Model execution is mocked; all preflight dependencies must be local
        # fixtures rather than artifacts that happen to exist on the author's host.
        from mvp_fullflow import run_mvp_fullflow as fullflow
        training = self.root / 'training_run'
        training.mkdir()
        delivery = self.root / 'delivery.json'
        qc = self.root / 'qc.json'
        delivery.write_text('{}')
        qc.write_text('{}')
        for name, value in [('DEFAULT_TRAINING_RUN', training),
                            ('DEFAULT_DELIVERY_STATUS', delivery),
                            ('DEFAULT_QC_JSON', qc)]:
            patcher = patch.object(fullflow, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        plan = self.private_plan()
        plan['selected_method']['id'] = 'tdd_2d_segmentation_infer_report'
        plan['intent'].update(dataset='TDD', mode='inference')
        plan['execution']['requested_case_id'] = '100'
        (self.data / 'imagesTs/a.png').rename(self.data / 'imagesTs/100_0000.png')
        (self.data / 'labelsTs/a.png').rename(self.data / 'labelsTs/100.png')
        dataset_spec = self.root / 'dataset_spec.json'
        dataset_spec.write_text(json.dumps({'root': str(self.data), 'extra': {'target_backend': 'nnunetv2_cli'}}))
        task_spec = self.root / 'task_spec.json'
        task_spec.write_text(json.dumps({'extra': {'nnunet_configuration': '2d'}}))
        checkpoint = self.root / 'checkpoint.pth'
        checkpoint.write_bytes(b'development fixture: never loaded')
        method = next(m for m in self.registry['methods'] if m['id'] == plan['selected_method']['id'])
        method['existing_assets'].update(dataset_root=str(self.data), dataset_spec=str(dataset_spec),
                                         task_spec=str(task_spec), checkpoint=str(checkpoint))
        registry_path = self.root / 'registry.json'
        registry_path.write_text(json.dumps(self.registry))
        plan['registry_path'] = str(registry_path)
        args = Namespace(private_data_root=None, registry=registry_path, case_id='100', execute=False,
                         reuse_fullflow_run=self.root / 'absent_run', reuse_inference=False, dentalclaw_python='python')
        return plan, args, checkpoint

    def assert_blocked_without_launch(self, plan, args, reason):
        with patch.object(platform, '_run_subprocess') as launcher, patch.object(platform.subprocess, 'run') as raw_launcher:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'run')
        launcher.assert_not_called(); raw_launcher.assert_not_called()
        self.assertEqual(result['status'], 'blocked_incompatible')
        self.assertIn(reason, result['reason_codes'])
        self.assertFalse(plan['execution']['will_execute'])
        self.assertFalse(plan['execution']['execution_permission'])
        self.assertEqual(json.loads((self.root / 'run/preflight.json').read_text()), plan['preflight'])
        return result

    def test_input_changed_after_positive_plan_is_rechecked(self):
        plan = self.private_plan()
        plan['preflight'] = preflight(plan, self.registry, self.data)
        self.assertEqual(plan['preflight']['status'], 'route_ready')
        (self.data / 'labelsTr/a.png').unlink()
        self.assert_blocked_without_launch(plan, self.private_args(), 'missing_masks')
        prediction = normalize_bundle({'id': 'DEV', 'run_status': 'ok', 'plan': plan})
        self.assertEqual(prediction['action'], 'block')
        self.assertFalse(prediction['execution_permission'])

    def test_missing_root_is_clarification_in_executor_and_evaluation(self):
        plan = self.private_plan(); args = self.private_args(); args.private_data_root = None
        with patch.object(platform, '_run_subprocess') as launcher:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'run')
        launcher.assert_not_called()
        self.assertEqual(result['status'], 'awaiting_clarification')
        pred = normalize_bundle({'id': 'DEV', 'run_status': 'ok', 'plan': plan})
        self.assertEqual(pred['action'], 'clarify'); self.assertEqual(pred['clarification_fields'], ['data_root'])

    def test_unverified_cli_package_cannot_enter_builtin_route(self):
        (self.data / 'dataset.json').write_text('{}')
        self.assert_blocked_without_launch(self.private_plan(), self.private_args(), 'backend_not_verified')

    def test_directory_name_alone_does_not_select_nnunet_backend(self):
        dest = self.root / 'DatasetUserRaster'
        self.data.rename(dest)
        result = preflight(self.private_plan(), self.registry, dest)
        self.assertEqual(result['status'], 'route_ready')
        self.assertEqual(result['backend'], 'builtin_nnunet_style_2d')

    def test_no_validation_split_is_blocked_before_training(self):
        shutil.rmtree(self.data / 'imagesVal'); shutil.rmtree(self.data / 'labelsVal')
        self.assert_blocked_without_launch(self.private_plan(), self.private_args(), 'validation_split_required')

    def test_tta_flag_cannot_redirect_plain_route_to_other_executor(self):
        plan = self.private_plan(); plan['intent']['tta'] = True
        self.assert_blocked_without_launch(plan, self.private_args(), 'route_contract_mismatch')

    def test_unavailable_registry_fails_closed(self):
        args = self.private_args(); args.registry = self.root / 'missing_registry.json'
        self.assert_blocked_without_launch(self.private_plan(), args, 'preflight_unavailable')

    def test_candidate_not_executable_cannot_launch(self):
        plan = self.private_plan(); plan['executable'] = False
        self.assert_blocked_without_launch(plan, self.private_args(), 'route_unavailable')

    def test_tdd_asset_removed_after_positive_plan_is_rechecked(self):
        plan, args, checkpoint = self.tdd_fixture()
        plan['preflight'] = preflight(plan, self.registry)
        self.assertEqual(plan['preflight']['status'], 'route_ready')
        checkpoint.unlink(); args.execute = True
        self.assert_blocked_without_launch(plan, args, 'required_asset_missing')

    def test_tdd_commands_bind_the_checked_assets_without_launching(self):
        plan, args, _ = self.tdd_fixture()
        with patch.object(platform, '_run_subprocess') as launcher:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'run')
        launcher.assert_not_called(); self.assertEqual(result['status'], 'command_generated')
        for option, key in [('--dataset-root', 'dataset_root'), ('--dataset-spec', 'dataset_spec'),
                            ('--task-spec', 'task_spec'), ('--model-path', 'checkpoint')]:
            self.assertEqual(result['command'][result['command'].index(option) + 1], plan['preflight']['validated_assets'][key])
        self.assertFalse(plan['execution']['will_execute'])

    def test_reused_predictions_are_copied_without_writing_historical_run(self):
        plan, args, _ = self.tdd_fixture()
        source = self.root / 'historical'; (source / 'inference').mkdir(parents=True)
        assets = next(m for m in self.registry['methods'] if m['id'] == plan['selected_method']['id'])['existing_assets']
        (source / 'manifest.json').write_text(json.dumps({
            **{k: assets[k] for k in ('dataset_root', 'dataset_spec', 'task_spec')},
            'model_path': assets['checkpoint']}))
        (source / 'inference/inference_summary.json').write_text('{}')
        Image.new('L', (16, 16), 1).save(source / 'inference/100.png')
        before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
        args.execute = True; args.reuse_inference = True; args.reuse_fullflow_run = source
        with patch.object(platform, '_run_subprocess', return_value=Namespace(returncode=0)) as launch:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'new_run')
        self.assertEqual(result['status'], 'incomplete_artifacts')
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--run-dir') + 1], str(self.root / 'new_run/delegate_tdd_fullflow'))
        after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual((self.root / 'new_run/delegate_tdd_fullflow/inference/100.png').read_bytes(), before['inference/100.png'])

    def test_invalid_reuse_source_does_not_start_delegate(self):
        plan, args, _ = self.tdd_fixture()
        args.execute = True; args.reuse_inference = True
        with patch.object(platform, '_run_subprocess') as launch:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'new_run')
        launch.assert_not_called()
        self.assertEqual(result['reason_codes'], ['reuse_source_invalid'])
        self.assertFalse(plan['execution']['will_execute'])

    def test_case_only_check_limits_the_fresh_tdd_delegate(self):
        plan, args, _ = self.tdd_fixture(); args.tdd_case_only = True
        with patch.object(platform, '_run_subprocess') as launch:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'new_run')
        launch.assert_not_called()
        self.assertIn('--inference-case-only', result['command'])

    def test_tdd_cannot_substitute_named_data_for_explicit_user_package(self):
        plan, args, _ = self.tdd_fixture(); args.private_data_root = str(self.root / 'other_package')
        self.assert_blocked_without_launch(plan, args, 'dataset_identity_mismatch')

    def test_requested_tdd_case_must_exist(self):
        plan, args, _ = self.tdd_fixture(); args.case_id = 'not_present'
        self.assert_blocked_without_launch(plan, args, 'requested_case_missing')

    def test_dataset_spec_cannot_point_at_a_different_root(self):
        plan, args, _ = self.tdd_fixture()
        (self.root / 'dataset_spec.json').write_text(json.dumps({'root': str(self.root / 'other'), 'extra': {'target_backend': 'nnunetv2_cli'}}))
        self.assert_blocked_without_launch(plan, args, 'asset_contract_mismatch')

    def test_native_planner_embeds_shared_gate_and_evaluator_reads_it(self):
        with patch.object(platform, '_do_web_search', return_value=[]), patch.object(platform, '_agent_decide', return_value=None):
            plan = platform.build_plan(intent='Train a segmentation model on my private 2D images.',
                                       registry_path=platform.DEFAULT_REGISTRY, case_id='100',
                                       run_dir=self.root / 'plan', reuse_fullflow_run=None, data_root=self.data)
        self.assertEqual(plan['preflight'], preflight(plan, self.registry, self.data))
        pred = normalize_bundle({'id': 'DEV', 'run_status': 'ok', 'plan': plan})
        self.assertEqual(pred['action'], 'route_ready')
        self.assertEqual(pred['backend'], 'builtin_nnunet_style_2d')
        self.assertFalse(plan['execution']['will_execute'])

    def test_cli_entry_saves_blocked_plan_without_starting_training(self):
        (self.data / 'labelsTr/a.png').unlink()
        args = Namespace(intent='Train a segmentation model on my private 2D images.',
                         registry=platform.DEFAULT_REGISTRY, case_id='100', run_dir=self.root / 'cli_run',
                         reuse_fullflow_run=None, lang='en', private_data_root=str(self.data), execute=True,
                         allow_external=False, dentalclaw_python='python')
        with patch.object(platform, 'parse_args', return_value=args), patch.object(platform, '_do_web_search', return_value=[]), patch.object(platform, '_agent_decide', return_value=None), patch.object(platform, '_run_subprocess') as launcher:
            self.assertEqual(platform.main(), 0)
        launcher.assert_not_called()
        plan = json.loads((args.run_dir / 'platform_plan.json').read_text())
        self.assertEqual(plan['preflight']['status'], 'block')
        self.assertFalse(plan['execution']['will_execute'])
        result = json.loads((args.run_dir / 'execution_result.json').read_text())
        self.assertEqual(result['status'], 'blocked_incompatible')
