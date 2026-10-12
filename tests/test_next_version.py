import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from platform_mvp.next_version import runner as n
from platform_mvp.data_services import service as s
from platform_mvp.data_services.build_binding import build
from platform_mvp.validate_private2d_package import validate_package,DEFAULT_CONTRACT


def decision(route=None,kind='use_registry',fields=None,**params):
    p={'case_ids':[],'artifact_kinds':[],'table':False,'selection_required':False};p.update(params)
    return dict(decision=kind,route_id=route,clarification_fields=fields or [],parameters=p,confidence=0.8,reasoning='development fixture')


class NextVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.data=self.root/'source';self.data.mkdir()
        for i,suf in enumerate(['Tr','Val','Ts']):
            for folder in ['images','labels']:(self.data/(folder+suf)).mkdir()
            Image.new('L',(8,8),20+i*30).save(self.data/f'images{suf}/a.png')
            Image.new('L',(8,8),1).save(self.data/f'labels{suf}/a.png')
        qc=validate_package(self.data,DEFAULT_CONTRACT,'private_train')
        self.context={'data_root':str(self.data),'data_card':{'dataset_id':'manifest:'+qc['manifest_id'],'manifest_id':qc['manifest_id']}}
        summary=self.data/'summary.json';s.write_json(summary,{'metrics_per_case':{'a':{'mean_dice':0.4}}})
        self.manifest=self.data/'binding.json';s.write_json(self.manifest,build(summary,'d','r',self.data/'imagesTs',self.data/'labelsTs',self.data/'imagesTs'))
        self.reg=s.read_json(n.REGISTRY)
    def tearDown(self):self.tmp.cleanup()
    def plan(self,d,c=None):return n.build_plan('development request',c if c is not None else self.context,d,self.reg)
    def test_four_choices_are_not_keyword_preselected(self):
        with patch.object(n,'preflight',return_value={'status':'route_ready'}) as gate:
            for route in [m['id'] for m in self.reg['methods']]:
                c=dict(self.context);c['selected_case_ids']=['a']
                if route == 'tdd_2d_segmentation_infer_report':
                    c['data_card'] = {'dataset_id': 'Dataset501_TDDTeethBinary2D'}
                p=self.plan(decision(route),c)
                self.assertEqual(p['selected_method']['id'],route)
        self.assertEqual(gate.call_count,4)
    def test_schema_disallows_unknown_route_and_code_paths(self):
        for d in [decision('shell'),decision('binary2d_package_qc',case_ids=['../x'])]:
            with self.assertRaises(ValueError):n.validate_decision(d)
        d=decision('binary2d_package_qc');d['parameters']['command']='echo bad'
        with self.assertRaises(ValueError):n.validate_decision(d)
    def test_context_rejects_reference_answers(self):
        with self.assertRaises(ValueError):n.validate_context({'expected_action':'route_ready'})
    def test_prompt_excludes_local_paths_and_artifacts(self):
        text=n.prompt_for('show existing results',{'result_manifest':str(self.manifest)})
        self.assertNotIn(str(self.root),text);self.assertIn('available_case_ids',text)
    def test_training_gate_and_explicit_launch_consent(self):
        p=self.plan(decision('private_2d_segmentation_train'))
        self.assertEqual(n.normalize(p)['action'],'route_ready')
        with patch('platform_mvp.run_platform_mvp.execute_selected_route') as run:
            r=n.dispatch(p,self.reg,self.context,self.root/'out')
        run.assert_not_called();self.assertFalse(r['actual_execution_started'])
    def test_qc_bad_data_audit_stays_executable_no_training(self):
        (self.data/'labelsTr/a.png').unlink()
        c={'data_root':str(self.data)}
        p=self.plan(decision('binary2d_package_qc'),c)
        self.assertEqual(n.normalize(p)['action'],'route_ready')
        with patch('platform_mvp.run_platform_mvp.execute_selected_route') as legacy:
            result=n.dispatch(p,self.reg,c,self.root/'qc')
        legacy.assert_not_called();self.assertEqual(result['status'],'completed')
        self.assertFalse(s.read_json(self.root/'qc/artifacts/qc_report.json')['quality_passed'])
    def test_result_delivery_does_not_infer(self):
        c={'result_manifest':str(self.manifest)};p=self.plan(decision('existing_result_delivery'),c)
        with patch('platform_mvp.run_platform_mvp.execute_selected_route') as legacy:
            result=n.dispatch(p,self.reg,c,self.root/'delivery')
        legacy.assert_not_called();self.assertEqual(result['status'],'completed')
    def test_missing_inputs_and_no_default_case(self):
        self.assertEqual(n.normalize(self.plan(decision('binary2d_package_qc'),{}))['clarification_fields'],['data_root'])
        self.assertEqual(n.normalize(self.plan(decision('existing_result_delivery'),{}))['clarification_fields'],['result_manifest'])
        c={'data_card':{'dataset_id':'Dataset501_TDDTeethBinary2D'}}
        self.assertEqual(n.normalize(self.plan(decision('tdd_2d_segmentation_infer_report'),c))['clarification_fields'],['requested_case_ids'])
    def test_explicit_dataset_identity_mismatch(self):
        c={'data_root':str(self.data),'data_card':{'dataset_id':'Dataset501_TDDTeethBinary2D'},'selected_case_ids':['a']}
        self.assertEqual(n.normalize(self.plan(decision('tdd_2d_segmentation_infer_report'),c))['action'],'block')
    def test_known_incompatible_dataset_blocks_before_case_clarification(self):
        c = {'data_card': {'dataset_id': 'another_dataset'}}
        pred = n.normalize(self.plan(decision('tdd_2d_segmentation_infer_report'), c))
        self.assertEqual(pred['action'], 'block')
        self.assertIn('route_contract_mismatch', pred['reason_codes'])

    def test_coverage_replay_blocks_partial_delivery_before_dispatch(self):
        from platform_mvp.next_version.coverage_contract import REQUIREMENTS
        context = self.root / 'context.json'
        response = self.root / 'decision.json'
        out = self.root / 'run'
        s.write_json(context, {'result_manifest': str(self.manifest)})
        d = decision('existing_result_delivery', case_ids=['a'], artifact_kinds=['mask'])
        d['requirements'] = dict.fromkeys(REQUIREMENTS, False)
        d['requirements'].update(ground_truth_masks=True, prediction_masks=True)
        s.write_json(response, d)
        with patch.object(n, 'call_planner') as online, patch.object(n, 'execute_service') as execute:
            code = n.main(['--intent', 'Deliver ground truth and predictions', '--context', str(context),
                           '--decision-json', str(response), '--coverage-check', '--execute',
                           '--out-dir', str(out)])
        self.assertEqual(code, 0)
        online.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(s.read_json(out/'prediction.json')['action'], 'no_registered_route')
        self.assertFalse(s.read_json(out/'execution_result.json')['actual_execution_started'])
        self.assertTrue(s.read_json(out/'coverage_audit.json')['capability_violations'])
        self.assertTrue(s.read_json(out/'runtime_version.json')['coverage_check'])

    def test_clarify_and_no_route_do_not_dispatch(self):
        for d in [decision(None,'clarify',['task']),decision(None,'no_registered_route')]:
            p=self.plan(d,{})
            self.assertEqual(n.normalize(p)['action'],d['decision'])
            self.assertFalse(n.dispatch(p,self.reg,{},self.root/'out')['actual_execution_started'])
    def test_online_failure_never_becomes_ready(self):
        p=self.plan(decision('binary2d_package_qc'))
        pred=n.normalize(p,{'status':'timeout'})
        self.assertIsNone(pred['action']);self.assertIsNone(pred['execution_permission'])
    def test_user_case_selection_conflict_blocks(self):
        c={'result_manifest':str(self.manifest),'selected_case_ids':['a']}
        self.assertEqual(n.normalize(self.plan(decision('existing_result_delivery',case_ids=['other']),c))['action'],'block')

if __name__=='__main__':unittest.main()
