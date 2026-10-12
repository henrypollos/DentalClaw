"""Regression checks for user intent and the binary route's target boundary."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from platform_mvp import run_platform_mvp as platform
from platform_mvp.route_preflight import preflight


class NaturalGoalRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = platform.load_registry(platform.DEFAULT_REGISTRY)

    def tearDown(self):
        self.tmp.cleanup()

    def test_negative_training_phrases_do_not_become_training_requests(self):
        for ending in ['不要训练', '不用重新训练模型', '无需重新训练',
                       'without retraining', 'do not train', "don't retrain", 'no need to train']:
            with self.subTest(ending=ending):
                parsed = platform.parse_intent('TDD tooth segmentation prediction; '+ending)
                self.assertEqual(parsed['mode'], 'inference')

    def test_a_negative_task_does_not_override_an_affirmative_task(self):
        for text in ['本院全景片训练分类模型，不做分割。',
                     '不要做分割，用本院影像训练分类模型。',
                     'Train a classification model on private images; do not segment them.']:
            with self.subTest(text=text):
                parsed = platform.parse_intent(text)
                self.assertEqual(parsed['task_family'], 'classification')
                self.assertEqual(parsed['mode'], 'private_train')
                self.assertIsNone(platform.select_method(parsed, self.registry)[0])

    def test_negative_training_in_one_clause_does_not_cancel_later_training(self):
        parsed = platform.parse_intent('不用重新训练 TDD 模型，而是用本院影像训练二值分割模型。')
        self.assertEqual(parsed['dataset'], 'Private2D')
        self.assertEqual(parsed['mode'], 'private_train')

    def test_training_metadata_is_not_an_instruction_to_train(self):
        parsed = platform.parse_intent('Use the model trained on TDD to segment the test radiographs.')
        self.assertEqual(parsed['mode'], 'inference')

    def test_ownership_paraphrases_still_identify_user_supplied_input(self):
        for text in ['Train segmentation on our hospital panoramic radiographs.',
                     'Train segmentation on my own dental images.',
                     '我有一些全景片，准备训练分割模型。']:
            with self.subTest(text=text):
                parsed = platform.parse_intent(text)
                self.assertEqual(parsed['dataset'], 'Private2D')
                self.assertEqual(parsed['mode'], 'private_train')

    def test_panorama_without_ownership_or_dataset_name_remains_unknown(self):
        parsed = platform.parse_intent('Train a binary segmentation model on panoramic radiographs.')
        self.assertEqual(parsed['dataset'], 'unknown')

    def test_learn_a_model_from_annotations_expresses_training(self):
        parsed = platform.parse_intent('从本院像素标注学出牙齿分割模型。')
        self.assertEqual(parsed['mode'], 'private_train')

    def test_category_assignment_goal_is_classification(self):
        parsed = platform.parse_intent('Learn from our hospital radiographs to assign each image to one of three categories.')
        self.assertEqual(parsed['dataset'], 'Private2D')
        self.assertEqual(parsed['task_family'], 'classification')

    def test_instance_segmentation_does_not_match_binary_routes(self):
        for text in ['本院影像训练分割模型，每颗牙齿分别生成独立掩膜并给出牙位编号。',
                     'Use TDD for instance segmentation and tooth numbering.']:
            with self.subTest(text=text):
                parsed = platform.parse_intent(text)
                self.assertEqual(parsed['segmentation_target'], 'instance')
                self.assertIsNone(platform.select_method(parsed, self.registry)[0])

    def test_multiclass_segmentation_does_not_match_binary_routes(self):
        for text in ['本院影像训练分割模型，把牙齿、牙龈、骨组织标成三个类别。',
                     'Train multiclass segmentation on private dental images.']:
            with self.subTest(text=text):
                parsed = platform.parse_intent(text)
                self.assertEqual(parsed['segmentation_target'], 'multiclass')
                self.assertIsNone(platform.select_method(parsed, self.registry)[0])

    def test_binary_task_can_still_select_its_verified_template(self):
        parsed = platform.parse_intent('用本院影像训练二值分割模型，区分牙齿和背景。')
        self.assertEqual(platform.select_method(parsed, self.registry)[0]['id'], 'private_2d_segmentation_train')

    def test_positive_detection_still_wins_over_mask_conversion(self):
        parsed = platform.parse_intent('本院影像把 segmentation mask 转为检测框，然后训练 detector。')
        self.assertEqual(parsed['task_family'], 'detection')

    def test_forced_binary_candidate_is_blocked_for_other_segmentation_targets(self):
        for target in ['instance', 'multiclass']:
            for rid,dataset,mode in [('private_2d_segmentation_train','Private2D','private_train'),
                                     ('tdd_2d_segmentation_infer_report','TDD','inference')]:
                with self.subTest(target=target,rid=rid):
                    plan = {'executable':True,'selected_method':{'id':rid,'status':'executable'},
                            'intent':{'dataset':dataset,'task_family':'segmentation','modality':'2d',
                                      'mode':mode,'segmentation_target':target}}
                    gate = preflight(plan, self.registry)
                    self.assertEqual(gate['status'], 'block')
                    self.assertFalse(gate['execution_permission'])

    def test_executor_rechecks_target_before_subprocess_launch(self):
        registry_path = self.root/'registry.json'
        registry_path.write_text(json.dumps(self.registry))
        plan = {'executable':True,'registry_path':str(registry_path),
                'selected_method':{'id':'private_2d_segmentation_train','status':'executable'},
                'intent':{'dataset':'Private2D','task_family':'segmentation','modality':'2d',
                          'mode':'private_train','segmentation_target':'instance'},'execution':{}}
        args = Namespace(execute=True, private_data_root=None, registry=registry_path, dentalclaw_python='python')
        with patch.object(platform,'_run_subprocess') as launch, patch.object(platform.subprocess,'run') as raw_launch:
            result = platform.execute_selected_route(plan=plan,args=args,run_dir=self.root/'run')
        launch.assert_not_called()
        raw_launch.assert_not_called()
        self.assertEqual(result['status'], 'blocked_incompatible')
        self.assertFalse(plan['execution']['execution_permission'])


if __name__ == '__main__':
    unittest.main()
