import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from platform_mvp.validate_private2d_package import validate_package, DEFAULT_CONTRACT
from platform_mvp.training_status import inspect_training_status
from platform_mvp import run_platform_mvp as platform


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split in ('Tr', 'Val', 'Ts'):
            (self.root / ('images' + split)).mkdir()
            (self.root / ('labels' + split)).mkdir()
            Image.new('L', (16, 16), {'Tr': 30, 'Val': 70, 'Ts': 110}[split]).save(self.root / ('images' + split) / 'a.png')
            Image.new('L', (16, 16), 1).save(self.root / ('labels' + split) / 'a.png')

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self):
        return validate_package(self.root, DEFAULT_CONTRACT, 'private_train')

    def test_valid_reused_case_names_are_not_leakage(self):
        result = self.validate()
        self.assertTrue(result['can_execute_requested_mode'])
        self.assertEqual(result['manifest']['patient_split_check'], 'not_established')

    def test_missing_partial_corrupt_shape_and_class(self):
        label = self.root / 'labelsTr/a.png'
        label.unlink()
        self.assertIn('missing_masks', self.validate()['reason_codes'])
        label.write_bytes(b'broken')
        self.assertIn('unreadable_file', self.validate()['reason_codes'])
        Image.new('L', (8, 8), 1).save(label)
        self.assertIn('image_mask_shape_mismatch', self.validate()['reason_codes'])
        Image.new('L', (16, 16), 255).save(label)
        self.assertIn('invalid_label_values', self.validate()['reason_codes'])
        Image.new('L', (16, 16), 1).save(label)
        Image.new('L', (16, 16), 66).save(self.root / 'imagesTr/b.png')
        self.assertIn('incomplete_pairing', self.validate()['reason_codes'])

    def test_pixels_and_patient_leakage(self):
        (self.root / 'imagesTs/a.png').write_bytes((self.root / 'imagesTr/a.png').read_bytes())
        self.assertIn('cross_split_image_duplicate', self.validate()['reason_codes'])
        (self.root / 'metadata.json').write_text(json.dumps({'patient_ids': {'train/a': 'P1', 'test/a': 'P1'}}))
        self.assertIn('cross_split_patient_duplicate', self.validate()['reason_codes'])

    def test_volume_and_duplicate_ids_blocked(self):
        (self.root / 'imagesTr/a_0000.nii.gz').write_bytes(b'not a volume')
        result = self.validate()
        self.assertIn('unsupported_2d_format', result['reason_codes'])
        self.assertIn('duplicate_case_id', result['reason_codes'])

    def test_wrong_filename_pairing_blocked(self):
        (self.root / 'imagesTr/a.png').rename(self.root / 'imagesTr/a_0000.png')
        self.assertIn('backend_filename_mismatch', self.validate()['reason_codes'])

    def test_validator_rejection_never_starts_training(self):
        (self.root / 'labelsTr/a.png').unlink()
        plan = {'executable': True, 'selected_method': {'id': 'private_2d_segmentation_train', 'status': 'executable'}, 'intent': {'dataset': 'Private2D', 'modality': '2d', 'task_family': 'segmentation', 'mode': 'private_train'}, 'execution': {}}
        args = Namespace(private_data_root=str(self.root), execute=True)
        with patch.object(platform, '_run_subprocess') as launch:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'run')
        self.assertEqual(result['status'], 'blocked_incompatible')
        launch.assert_not_called()

    def test_legal_package_launch_is_not_completion_and_specs_match(self):
        plan = {'executable': True, 'selected_method': {'id': 'private_2d_segmentation_train', 'status': 'executable'}, 'intent': {'dataset': 'Private2D', 'modality': '2d', 'task_family': 'segmentation', 'mode': 'private_train'}, 'execution': {}}
        args = Namespace(private_data_root=str(self.root), execute=True, dentalclaw_python='python')
        with patch.object(platform, '_run_subprocess', return_value=Namespace(returncode=0)) as launch:
            result = platform.execute_selected_route(plan=plan, args=args, run_dir=self.root / 'run')
        self.assertEqual(result['status'], 'submitted')
        launch.assert_called_once()
        spec = json.loads((self.root / 'run/private2d_dataset_spec.json').read_text())
        self.assertEqual(spec['root'], str(self.root))
        self.assertEqual(spec['imagesVal'], 'imagesVal')
        self.assertTrue(spec['extra']['input_manifest_id'])

    def test_ownership_preserved(self):
        for text in ['本院全景片分割训练', '请对本院CBCT进行分割推理']:
            self.assertEqual(platform.parse_intent(text)['dataset_source'], 'user_supplied')
        self.assertEqual(platform.parse_intent('全景片分割')['dataset'], 'unknown')
        self.assertEqual(platform.parse_intent('TDD segmentation inference')['dataset'], 'TDD')

    def test_no_private_fallback_to_tdd(self):
        plan = {'selected_method': {'id': 'dental_2d_super_resolution'}, 'intent': {'dataset': 'Private2D'}}
        result = platform.execute_selected_route(plan=plan, args=Namespace(), run_dir=self.root / 'run')
        self.assertEqual(result['status'], 'blocked_incompatible')

    def test_status_failure_interruption_and_incomplete_outputs(self):
        w = self.root / 'status'; w.mkdir()
        self.assertEqual(inspect_training_status(w)['status'], 'submitted')
        for status in ('failed', 'interrupted'):
            (w / 'run_status.json').write_text(json.dumps({'status': status}))
            self.assertEqual(inspect_training_status(w)['status'], status)
        (w / 'run_status.json').write_text(json.dumps({'status': 'completed'}))
        self.assertEqual(inspect_training_status(w)['status'], 'incomplete_artifacts')
        (w / 'run_status.json').unlink()
        (w / 'launcher_status.json').write_text(json.dumps({'status': 'failed'}))
        self.assertEqual(inspect_training_status(w)['status'], 'failed')

    def test_completion_requires_readable_predictions_and_matching_count(self):
        w = self.root / 'complete'; w.mkdir()
        for folder in ('report', 'best_inference/masks'):
            (w / folder).mkdir(parents=True)
        model = w / 'model.pth'; model.write_bytes(b'file existence only; never loaded in status reader')
        for name, payload in [('run_status.json', {'status': 'completed'}),
                              ('run_summary.json', {'best_model_export_path': str(model)}),
                              ('main_handoff.json', {'status': 'completed'}),
                              ('report/summary.json', {'num_trials': 1}),
                              ('best_inference/inference_summary.json', {'num_cases': 1})]:
            (w / name).write_text(json.dumps(payload))
        mask = w / 'best_inference/masks/a.png'; mask.write_bytes(b'invalid png')
        self.assertEqual(inspect_training_status(w)['status'], 'incomplete_artifacts')
        Image.new('L', (16, 16), 0).save(mask)
        self.assertEqual(inspect_training_status(w)['status'], 'completed')
        (w / 'best_inference/inference_summary.json').write_text(json.dumps({'num_cases': 2}))
        self.assertEqual(inspect_training_status(w)['status'], 'incomplete_artifacts')


if __name__ == '__main__':
    unittest.main()
