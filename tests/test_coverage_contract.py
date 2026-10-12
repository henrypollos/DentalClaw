import unittest
from platform_mvp.next_version.coverage_contract import REQUIREMENTS,apply_coverage_contract,validate_decision
class CoverageTests(unittest.TestCase):
 def decision(self,route,**req):
  return dict(decision='use_registry',route_id=route,clarification_fields=[],parameters=dict(case_ids=['example'],artifact_kinds=['mask'],table=False,selection_required=False),confidence=.9,reasoning='proposed supported route',requirements=dict.fromkeys(REQUIREMENTS,False)|req)
 def test_two_mask_roles_cannot_be_collapsed_to_one_kind(self):
  d,a=apply_coverage_contract(self.decision('existing_result_delivery',ground_truth_masks=True,prediction_masks=True))
  self.assertEqual(d['decision'],'no_registered_route');self.assertIsNone(d['route_id']);self.assertTrue(a['capability_violations'])
 def test_prediction_delivery_and_optional_truth_remain_supported(self):
  d,a=apply_coverage_contract(self.decision('existing_result_delivery',prediction_masks=True))
  self.assertEqual(d['decision'],'use_registry');self.assertFalse(a['capability_violations'])
 def test_package_export_not_confused_with_inventory_for_any_route(self):
  for route in ['binary2d_package_qc','existing_result_delivery','private_2d_segmentation_train','tdd_2d_segmentation_infer_report']:
   d,a=apply_coverage_contract(self.decision(route,source_package_files=True))
   self.assertEqual(d['decision'],'no_registered_route')
 def test_conditional_training_and_qc_require_complete_workflow(self):
  for route in ['binary2d_package_qc','private_2d_segmentation_train']:
   d,a=apply_coverage_contract(self.decision(route,qc_report=True,smoke_training=True))
   self.assertEqual(d['decision'],'no_registered_route')
 def test_simple_qc_and_simple_training_not_blocked(self):
  for route,req in [('binary2d_package_qc',{'qc_report':True}),('private_2d_segmentation_train',{'smoke_training':True})]:
   self.assertEqual(apply_coverage_contract(self.decision(route,**req))[0]['decision'],'use_registry')
 def test_builtin_inference_is_not_additional_analysis(self):
  self.assertEqual(apply_coverage_contract(self.decision('tdd_2d_segmentation_infer_report',prediction_masks=True))[0]['decision'],'use_registry')
 def test_requirements_schema_fails_closed(self):
  d=self.decision('existing_result_delivery');del d['requirements']['qc_report']
  with self.assertRaises(ValueError):validate_decision(d)
  d=self.decision('existing_result_delivery');d['requirements']['qc_report']='false'
  with self.assertRaises(ValueError):validate_decision(d)
 def test_raw_proposal_unchanged_and_auditable(self):
  d=self.decision('existing_result_delivery',ground_truth_masks=True,prediction_masks=True)
  corrected,a=apply_coverage_contract(d)
  self.assertEqual(d['decision'],'use_registry');self.assertEqual(a['proposed_route_id'],d['route_id']);self.assertNotIn('requirements',corrected)
if __name__=='__main__':unittest.main()
