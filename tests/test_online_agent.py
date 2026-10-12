import copy
import io
import json
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from platform_mvp.online_agent import call_planner, DEFAULT_SETTINGS, validate_settings
from platform_mvp.agent_data_card import card_from_validation
from platform_mvp.evaluation_adapter import normalize_bundle
from platform_mvp import run_platform_mvp as platform


class Response(io.BytesIO):
    status = 200


class OnlineAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)

    def tearDown(self):
        self.tmp.cleanup()

    def body(self, value=None, finish='stop'):
        value = value or {'decision':'use_registry', 'confidence':0.9, 'reasoning':'Existing route fits.'}
        return json.dumps({'model':'deepseek-v4-pro', 'id':'mock-response', 'usage':{'total_tokens':10},
                           'choices':[{'finish_reason':finish, 'message':{'content':json.dumps(value)}}]}).encode()

    def call(self, **kwargs):
        return call_planner('JSON decision for aggregate input facts.', settings=self.settings,
                            api_key=kwargs.pop('api_key', 'test-secret'),
                            registry_available=kwargs.pop('registry_available', True), audit_dir=self.root/'audit', **kwargs)

    def test_valid_response_saved_without_credentials_or_headers(self):
        with patch('urllib.request.urlopen', return_value=Response(self.body())) as transport:
            result = self.call()
        self.assertEqual(result['status'],'ok'); self.assertEqual(transport.call_count,1)
        content=''.join(p.read_text() for p in (self.root/'audit').iterdir())
        self.assertNotIn('test-secret',content); self.assertNotIn('Authorization',content)
        self.assertEqual(result['returned_model'],'deepseek-v4-pro')
        self.assertTrue((self.root/'audit/raw_response.txt').exists())

    def test_http_401_is_recorded_once_without_retry(self):
        exc=urllib.error.HTTPError('https://api.deepseek.com',401,'Unauthorized',{},io.BytesIO(b'Invalid test-secret'))
        with patch('urllib.request.urlopen',side_effect=exc) as transport:
            result=self.call()
        self.assertEqual(transport.call_count,1); self.assertEqual(result['http_status'],401)
        self.assertEqual(result['status'],'http_error'); self.assertIsNone(result['decision'])
        self.assertNotIn('test-secret',(self.root/'audit/raw_response.txt').read_text())

    def test_missing_credentials_do_not_call_network(self):
        with patch('urllib.request.urlopen') as transport:
            result=self.call(api_key=None)
        transport.assert_not_called(); self.assertEqual(result['attempt_count'],0)
        self.assertEqual(result['status'],'missing_credentials')

    def test_timeout_is_not_success_or_automatic_fallback(self):
        with patch('urllib.request.urlopen',side_effect=socket.timeout()) as transport:
            result=self.call()
        self.assertEqual(transport.call_count,1); self.assertEqual(result['status'],'timeout')
        self.assertFalse(result['fallback_performed_by_transport'])

    def test_use_registry_without_match_is_invalid(self):
        with patch('urllib.request.urlopen',return_value=Response(self.body())):
            result=self.call(registry_available=False)
        self.assertEqual(result['status'],'invalid_schema'); self.assertIsNone(result['decision'])

    def test_markdown_wrapped_json_is_not_repaired(self):
        body=json.loads(self.body());body['choices'][0]['message']['content']='```json\n{}\n```'
        with patch('urllib.request.urlopen',return_value=Response(json.dumps(body).encode())):
            result=self.call()
        self.assertEqual(result['status'],'invalid_response')

    def test_truncated_json_is_not_accepted(self):
        with patch('urllib.request.urlopen',return_value=Response(self.body(finish='length'))):
            result=self.call()
        self.assertEqual(result['status'],'incomplete_response'); self.assertIsNone(result['decision'])

    def test_credentials_or_provider_switch_cannot_enter_settings_snapshot(self):
        for values in [{'api_key':'secret'}, {'endpoint':'https://untrusted.example/chat'}, {'max_attempts':2}]:
            s=copy.deepcopy(self.settings); s.update(values)
            with self.subTest(values=list(values)),self.assertRaises(ValueError):validate_settings(s)

    def test_card_contains_aggregate_facts_without_patient_or_file_records(self):
        report={'manifest_id':'hash','split_counts':{'train':{'image_count':1,'paired_case_count':1}},
                'manifest':{'patient_split_check':'checked_supplied_ids','files':[
                    {'kind':'image','split':'train','case_id':'sensitive-case','path':'private.png','shape':[16,16],'patient_id':'sensitive-patient'},
                    {'kind':'mask','split':'train','case_id':'sensitive-case','path':'private-label.png','shape':[8,8],'label_values':[0,1]}]}}
        card=card_from_validation(report); text=json.dumps(card)
        self.assertEqual(card['image_mask_shape_mismatch_count'],1)
        self.assertEqual(card['label_values_observed'],[0,1])
        for field in ['sensitive-case','private.png','sensitive-patient','expected','route_id']:
            self.assertNotIn(field,text)

    def test_data_card_reaches_native_agent_prompt(self):
        record={}
        with patch.object(platform,'_load_deepseek_api_key',return_value=None),patch('platform_mvp.online_agent.call_planner',return_value={'status':'http_error','attempt_count':1,'http_status':401,'decision':None}) as call:
            decision=platform._agent_decide(intent='Train on my private 2D masks.', parsed=platform.parse_intent('Train segmentation on private 2D masks.'),
                                            search_results=[], registry_match={'id':'private_2d_segmentation_train','display_name':'Private','status':'executable'}, registry_reasons=[],
                                            data_card={'data_card_id':'DEV-CARD','image_mask_shape_mismatch_count':1},online_settings=self.settings,audit_dir=self.root/'audit',audit_record=record)
        self.assertIsNone(decision);self.assertIn('DEV-CARD',call.call_args.args[0])
        self.assertEqual(record['http_status'],401)

    def test_online_failure_with_successful_registry_fallback_still_counts_as_error(self):
        plan={'online_call_required':True,'agent_call':{'status':'http_error'},'executable':True,
              'intent':{'dataset':'Private2D'},'selected_method':{'id':'private_2d_segmentation_train','status':'executable'},
              'preflight':{'status':'route_ready','candidate_route_id':'private_2d_segmentation_train','execution_permission':True}}
        pred=normalize_bundle({'id':'DEV','run_status':'ok','plan':plan})
        self.assertEqual(pred['run_status'],'error'); self.assertIsNone(pred['execution_permission'])

    def test_conflicting_gate_cannot_override_native_production_gate(self):
        plan={'preflight':{'status':'block','execution_permission':False}}
        pred=normalize_bundle({'id':'DEV','run_status':'ok','plan':plan,'preflight':{'status':'route_ready','execution_permission':True}})
        self.assertEqual(pred['run_status'],'invalid')

    def test_early_clarification_does_not_pretend_to_call_online_agent(self):
        with patch.object(platform,'_agent_decide') as agent:
            plan=platform.build_plan(intent='帮我研究牙齿情况。',registry_path=platform.DEFAULT_REGISTRY,
                                     case_id='100',run_dir=self.root/'run',reuse_fullflow_run=None,online_settings=self.settings,search_enabled=False)
        agent.assert_not_called(); self.assertEqual(plan['agent_call']['attempt_count'],0)
        self.assertEqual(plan['agent_call']['status'],'not_called_early_clarification')

    def test_custom_provider_uses_separate_explicit_configuration(self):
        self.settings.update(provider='openai_compatible',endpoint='https://llm.example/v1/chat/completions',
                             thinking=None,credential_env='DENTALCLAW_PLANNER_API_KEY')
        with patch('urllib.request.urlopen',return_value=Response(self.body())) as transport:
            result=self.call(api_key='custom-test-key')
        request=transport.call_args.args[0]
        self.assertEqual(request.full_url,self.settings['endpoint'])
        self.assertNotIn('thinking',json.loads(request.data))
        self.assertEqual(result['status'],'ok')

    def test_custom_provider_does_not_reuse_deepseek_profile(self):
        settings=copy.deepcopy(self.settings)
        settings.update(provider='openai_compatible',endpoint='https://llm.example/v1/chat/completions',
                        thinking=None,credential_env='DENTALCLAW_PLANNER_API_KEY')
        with patch.dict('os.environ',{},clear=True),patch.object(platform,'_load_deepseek_api_key') as old_key,patch('platform_mvp.online_agent.call_planner',return_value={'status':'missing_credentials','decision':None}) as call:
            platform._agent_decide(intent='Private segmentation',parsed={'dataset':'Private2D'},search_results=[],
                                   registry_match=None,registry_reasons=[],online_settings=settings)
        old_key.assert_not_called();self.assertIsNone(call.call_args.kwargs['api_key'])

    def test_card_observes_dataset_json_in_input_directory(self):
        report={'data_root':str(self.root),'manifest_id':'hash','split_counts':{},
                'manifest':{'files':[],'patient_split_check':'not_established'}}
        self.assertFalse(card_from_validation(report)['dataset_json_present'])
        (self.root/'dataset.json').write_text('{}')
        self.assertTrue(card_from_validation(report)['dataset_json_present'])
        report.pop('data_root')
        self.assertIsNone(card_from_validation(report)['dataset_json_present'])

    def test_native_input_block_overrides_misleading_agent_readiness_in_reason(self):
        data=self.root/'empty_package';data.mkdir()
        response={'status':'ok','attempt_count':1,'decision':{'decision':'use_registry','confidence':0.9,
                   'reasoning':'All inputs are valid and execution can proceed.'}}
        with patch('platform_mvp.online_agent.call_planner',return_value=response):
            plan=platform.build_plan(intent='Train a segmentation model on my private 2D images.',
                                     registry_path=platform.DEFAULT_REGISTRY,case_id='100',run_dir=self.root/'run',
                                     reuse_fullflow_run=None,data_root=data,data_card={'data_card_id':'DEV'},
                                     online_settings=self.settings,search_enabled=False)
        self.assertEqual(plan['preflight']['status'],'block')
        self.assertFalse(plan['execution']['execution_permission'])
        self.assertNotIn('All inputs are valid',plan['reason'])
        self.assertIn('All inputs are valid',plan['selected_method']['agent_note'])

    def test_valid_package_reason_uses_gate_instead_of_invented_agent_requirements(self):
        data=self.root/'valid_package'
        for split,value in [('Tr',30),('Val',70),('Ts',110)]:
            for folder in ['images','labels']:(data/(folder+split)).mkdir(parents=True)
            Image.new('L',(16,16),value).save(data/f'images{split}/a.png')
            Image.new('L',(16,16),1).save(data/f'labels{split}/a.png')
        response={'status':'ok','attempt_count':1,'decision':{'decision':'use_registry','confidence':0.9,
                   'reasoning':'A dataset.json is required before execution.'}}
        with patch('platform_mvp.online_agent.call_planner',return_value=response):
            plan=platform.build_plan(intent='Train a segmentation model on my private 2D images.',
                                     registry_path=platform.DEFAULT_REGISTRY,case_id='100',run_dir=self.root/'run',
                                     reuse_fullflow_run=None,data_root=data,data_card={'data_card_id':'DEV'},
                                     online_settings=self.settings,search_enabled=False)
        self.assertTrue(plan['execution']['execution_permission'])
        self.assertFalse(plan['execution']['will_execute'])
        self.assertEqual(plan['reason_source'],'deterministic_preflight')
        self.assertNotIn('dataset.json',plan['reason'])
        self.assertIn('dataset.json',plan['selected_method']['agent_note'])


if __name__=='__main__':unittest.main()
