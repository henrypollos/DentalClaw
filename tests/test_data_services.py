import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace
from PIL import Image
from platform_mvp.data_services import service as s
from platform_mvp.data_services.build_binding import build
from platform_mvp.data_services.runner import main, identify_task
from platform_mvp.route_preflight import preflight
from platform_mvp.evaluation_adapter import normalize_bundle


class DataServicesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source';self.source.mkdir()
        self.registry=s.read_json(s.REGISTRY)
        self.data=self.source/'data';self.data.mkdir()
        for folder in ['images','masks']:(self.data/folder).mkdir()
        Image.new('L',(8,8),40).save(self.data/'images/a.png')
        Image.new('L',(8,8),1).save(self.data/'masks/a.png')
        self.summary=self.source/'summary.json'
        s.write_json(self.summary,dict(metrics_per_case={'a':{'mean_dice':0.5,'mean_iou':0.25}},mean_dice=0.5))
        self.manifest=self.source/'binding.json'
        s.write_json(self.manifest,build(self.summary,'dataset-A','run-A',self.data/'images',self.data/'masks',self.data/'images'))

    def tearDown(self):self.tmp.cleanup()

    def plan(self,kind='results',**kwargs):
        rid='existing_result_delivery' if kind=='results' else 'binary2d_package_qc'
        method=next(m for m in self.registry['methods'] if m['id']==rid)
        return dict(executable=True,selected_method=method,
                    intent=dict(task_family=method['task_family'],mode=method['allowed_modes'][0],modality='2d'),
                    execution=dict(result_manifest=str(self.manifest),**kwargs))

    def test_missing_inputs_and_selection_clarify(self):
        p=self.plan();p['execution']={}
        self.assertEqual(preflight(p,self.registry)['clarification_fields'],['result_manifest'])
        p=self.plan(selection_required=True)
        self.assertEqual(preflight(p,self.registry)['clarification_fields'],['requested_case_ids'])
        self.assertEqual(preflight(self.plan('qc'),self.registry)['clarification_fields'],['data_root'])

    def test_wrong_identity_and_missing_case_block(self):
        for kw,reason in [({'expected_dataset_id':'wrong'},'dataset_identity_mismatch'),({'expected_run_id':'wrong'},'run_identity_mismatch'),({'requested_case_ids':['absent']},'requested_case_missing_or_duplicate')]:
            self.assertIn(reason,preflight(self.plan(**kw),self.registry)['reason_codes'])

    def test_rechecks_changed_source_without_delivery(self):
        p=self.plan();self.assertEqual(preflight(p,self.registry)['status'],'route_ready')
        self.summary.write_text('{}')
        r=s.execute(p,self.registry,self.root/'out')
        self.assertEqual(r['status'],'block');self.assertFalse((self.root/'out').exists())

    def test_copy_index_table_are_traceable_no_model_launch(self):
        p=self.plan(table=True,requested_case_ids=['a'])
        with patch('subprocess.run',side_effect=AssertionError('no subprocess')):
            r=s.execute(p,self.registry,self.root/'out')
        self.assertEqual(r['status'],'completed')
        d=s.read_json(self.root/'out/delivery_manifest.json')
        self.assertEqual(d['dataset_id'],'dataset-A');self.assertFalse(d['metrics_recomputed'])
        for a in d['artifacts']:self.assertEqual(s.digest(self.root/'out'/a['path']),a['sha256'])
        self.assertIn('0.5',(self.root/'out/per_case_metrics.csv').read_text())
        self.assertIn('a / overlay',(self.root/'out/report.html').read_text())

    def test_invalid_hash_duplicate_artifact_and_missing_file(self):
        orig=s.read_json(self.manifest)
        for variant,reason in [(dict(orig,artifacts=orig['artifacts']*2),'duplicate_case_artifact'),(dict(orig,artifacts=[dict(orig['artifacts'][0],sha256='bad')]),'artifact_hash_mismatch'),(dict(orig,artifacts=[dict(orig['artifacts'][0],path='/nonexistent/never')]),'result_artifact_missing')]:
            s.write_json(self.manifest,variant)
            self.assertIn(reason,preflight(self.plan(),self.registry)['reason_codes'])

    def test_qc_success_does_not_grant_training_permission(self):
        r=s.execute(self.plan('qc'),self.registry,self.root/'out',self.data)
        self.assertEqual(r['status'],'completed')
        q=s.read_json(self.root/'out/qc_report.json');self.assertTrue(q['quality_passed']);self.assertFalse(q['training_execution_permission'])

    def test_qc_detects_bad_label_pair_shape_and_unreadable(self):
        Image.new('L',(7,8),255).save(self.data/'masks/a.png')
        Image.new('L',(8,8),30).save(self.data/'images/missing.png')
        (self.data/'images/broken.png').write_bytes(b'not an image')
        p=self.plan('qc');g=preflight(p,self.registry,self.data)
        self.assertEqual(g['status'],'route_ready')
        r=s.execute(p,self.registry,self.root/'out',self.data)
        self.assertEqual(r['status'],'completed')
        q=s.read_json(self.root/'out/qc_report.json');self.assertFalse(q['quality_passed'])
        self.assertTrue({'invalid_label_values','image_mask_shape_mismatch','incomplete_pairing','unreadable_file'}<=set(q['reason_codes']))
        self.assertEqual(normalize_bundle(dict(id='DEV',run_status='ok',plan=p))['action'],'route_ready')

    def test_qc_empty_and_missing_root_block(self):
        empty=self.root/'empty';empty.mkdir()
        self.assertEqual(preflight(self.plan('qc'),self.registry,empty)['status'],'block')
        self.assertEqual(preflight(self.plan('qc'),self.registry,self.root/'absent')['status'],'block')

    def test_output_cannot_overwrite_or_enter_source(self):
        for out in [self.root/'source/new',self.source]:
            with self.assertRaises(s.ContractError):s.execute(self.plan(),self.registry,out)

    def test_qc_cli_and_normalized_prediction(self):
        out=self.root/'run'
        code=main(['--intent','请检查图像掩膜配对和标签值','--data-root',str(self.data),'--out-dir',str(out),'--execute'])
        self.assertEqual(code,0)
        p=s.read_json(out/'prediction.json');self.assertEqual(p['route_id'],'binary2d_package_qc');self.assertTrue(p['actual_execution_started'])

    def test_plan_only_and_unsupported_analysis_never_execute(self):
        for i,text in enumerate(['请领取已有结果文件','请用已有结果计算 bootstrap 置信区间']):
            out=self.root/f'run{i}'
            self.assertEqual(main(['--intent',text,'--result-manifest',str(self.manifest),'--out-dir',str(out)]),0)
            p=s.read_json(out/'prediction.json');self.assertIsNone(p['actual_execution_started']);self.assertFalse((out/'artifacts').exists())
            self.assertEqual(p['action'],'route_ready' if i==0 else 'no_registered_route')

    def test_unknown_task_clarifies(self):
        out=self.root/'run'
        main(['--intent','帮我做研究','--out-dir',str(out)])
        self.assertEqual(s.read_json(out/'prediction.json')['action'],'clarify')

    def test_table_missing_metrics_blocks(self):
        s.write_json(self.summary,{'mean_dice':0.5})
        d=s.read_json(self.manifest);d['artifacts'][0]['sha256']=s.digest(self.summary);s.write_json(self.manifest,d)
        self.assertIn('per_case_metrics_missing',preflight(self.plan(table=True),self.registry)['reason_codes'])

    def test_summary_only_binding_delivers_without_fake_images(self):
        s.write_json(self.manifest,build(self.summary,'dataset-A','run-A'))
        out=self.root/'run'
        main(['--intent','领取已有结果文件，不需要另外制作表格','--result-manifest',str(self.manifest),'--out-dir',str(out),'--execute'])
        self.assertEqual(s.read_json(out/'prediction.json')['action'],'route_ready')
        self.assertFalse((out/'artifacts/per_case_metrics.csv').exists())

    def test_requested_table_is_delivered(self):
        out=self.root/'run'
        main(['--intent','把已有指标整理成表格','--result-manifest',str(self.manifest),'--out-dir',str(out),'--execute'])
        self.assertTrue((out/'artifacts/per_case_metrics.csv').is_file())

    def test_explicit_user_selection_no_default_case(self):
        out=self.root/'run'
        main(['--intent','查看指定病例已有叠加图','--result-manifest',str(self.manifest),'--out-dir',str(out),'--execute'])
        p=s.read_json(out/'prediction.json');self.assertEqual(p['action'],'clarify');self.assertFalse(p['actual_execution_started'])

if __name__=='__main__':unittest.main()
