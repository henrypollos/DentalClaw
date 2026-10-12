import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from platform_mvp.evaluation_adapter import normalize_bundle, preflight
from platform_mvp import run_platform_mvp as platform


def bundle():
    return {'id':'DEV','run_status':'ok','plan':{'intent':{'dataset':'Private2D','task_family':'segmentation','modality':'2d','mode':'private_train'},'selected_method':{'id':'private_2d_segmentation_train','status':'executable'},'executable':True},'preflight':{}}


class EvaluationAdapterTests(unittest.TestCase):
    def test_vague_dental_request_clarifies_without_lookup_or_execution(self):
        for text in ['帮我研究牙齿情况。', 'Help me investigate dental images.']:
            with self.subTest(text=text), patch.object(platform, '_do_web_search') as search, patch.object(platform, '_agent_decide') as agent:
                plan = platform.build_plan(intent=text, registry_path=platform.REPO_ROOT / 'platform_mvp/method_registry.json', case_id='100', run_dir=Path('/tmp/not-created'), reuse_fullflow_run=None)
            search.assert_not_called(); agent.assert_not_called()
            self.assertFalse(plan['execution']['will_execute'])
            pred = normalize_bundle({'id':'DEV', 'run_status':'ok', 'plan':plan})
            self.assertEqual(pred['action'], 'clarify')
            self.assertEqual(pred['clarification_fields'], ['dataset_id', 'task'])
            self.assertFalse(pred['execution_permission'])

    def test_unrelated_request_still_rejected_without_lookup(self):
        with patch.object(platform, '_do_web_search') as search, patch.object(platform, '_agent_decide') as agent:
            plan = platform.build_plan(intent='Tell me the weather tomorrow.', registry_path=platform.REPO_ROOT / 'platform_mvp/method_registry.json', case_id='100', run_dir=Path('/tmp/not-created'), reuse_fullflow_run=None)
        search.assert_not_called(); agent.assert_not_called()
        self.assertFalse(plan['supported']); self.assertFalse(plan['executable'])
        self.assertIsNone(plan['selected_method'])

    def test_executable_without_gate_not_ready(self):
        p=normalize_bundle(bundle());self.assertEqual(p['run_status'],'invalid');self.assertIsNone(p['execution_permission'])

    def test_gate_block_overrides_candidate_not_reference(self):
        b=bundle();b['preflight']={'candidate_route_id':'private_2d_segmentation_train','status':'block','execution_permission':False,'reason_codes':['incomplete_pairing']}
        p=normalize_bundle(b);self.assertEqual(p['action'],'block');self.assertIsNone(p['route_id']);self.assertIsNone(p['actual_execution_started'])

    def test_ready_retains_context(self):
        b=bundle();b['preflight']={'candidate_route_id':'private_2d_segmentation_train','status':'route_ready','execution_permission':True,'dataset_id':'manifest:123','backend':'builtin_nnunet_style_2d'}
        p=normalize_bundle(b);self.assertEqual(p['route_id'],'private_2d_segmentation_train');self.assertEqual(p['dataset_id'],'manifest:123')

    def test_proposal_over_existing_route_not_corrected(self):
        b=bundle();b['plan'].update(executable=False,registry_lookup_performed=True,registry_match_id='private_2d_segmentation_train',selected_method={'id':'agent_external_suggestion'})
        self.assertEqual(normalize_bundle(b)['run_status'],'invalid')

    def test_no_match_requires_structured_evidence(self):
        b=bundle();b['plan'].update(executable=False,selected_method=None)
        self.assertEqual(normalize_bundle(b)['run_status'],'invalid')
        b['plan'].update(registry_lookup_performed=True,registry_match_id=None)
        self.assertEqual(normalize_bundle(b)['action'],'no_registered_route')

    def test_clarification_is_not_rejection(self):
        b=bundle();b['plan']['intent'].update(dataset='unknown',task_family='unknown');b['plan']['selected_method']=None;b['plan']['executable']=False
        self.assertEqual(normalize_bundle(b)['run_status'],'invalid')
        b['plan']['selected_method']={'id':'platform_clarification'}
        self.assertEqual(normalize_bundle(b)['clarification_fields'],['dataset_id','task'])

    def test_preflight_wrong_modality_blocks_before_reading(self):
        b=bundle();b['plan']['intent']['modality']='3d'
        with patch('platform_mvp.route_preflight.validate_package') as qc:
            result=preflight(b['plan'],{'methods':[{'id':'private_2d_segmentation_train','status':'executable'}]},Path('/nonexistent'))
        self.assertEqual(result['status'],'block');qc.assert_not_called()

    def test_backend_asset_absence_not_success(self):
        b=bundle();b['plan']['selected_method']={'id':'tdd_2d_segmentation_infer_report','status':'executable'};b['plan']['intent'].update(dataset='TDD',mode='inference')
        registry={'methods':[{'id':'tdd_2d_segmentation_infer_report','status':'executable','existing_assets':{'dataset_root':'not-existing'}}]}
        self.assertEqual(preflight(b['plan'],registry)['reason_codes'],['required_asset_missing'])
