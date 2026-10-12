"""Validate required deliverables against the unchanged four-route contracts.

This is not semantic extraction from keywords or evaluation IDs: requirements are
explicit planner output and are retained in the audit alongside the raw decision.
"""
from copy import deepcopy
from platform_mvp.next_version import runner as base

REQUIREMENTS={'ground_truth_masks','prediction_masks','source_package_files','qc_report','smoke_training','additional_analysis'}

def validate_decision(value):
    if not isinstance(value,dict) or 'requirements' not in value:
        raise ValueError('required_deliverables_missing')
    requirements=value['requirements']
    if not isinstance(requirements,dict) or set(requirements)!=REQUIREMENTS or any(type(v) is not bool for v in requirements.values()):
        raise ValueError('required_deliverables_schema')
    base.validate_decision({k:v for k,v in value.items() if k!='requirements'})
    return value

def apply_coverage_contract(value):
    validate_decision(value)
    d=deepcopy({k:v for k,v in value.items() if k!='requirements'})
    r=value['requirements'];rid=d['route_id'];reasons=[]
    if d['decision']=='use_registry':
        if r['source_package_files']:reasons.append('source_package_export_not_registered')
        if r['additional_analysis']:reasons.append('additional_analysis_not_registered')
        if r['qc_report'] and rid!='binary2d_package_qc':reasons.append('standalone_qc_report_not_in_selected_route')
        if r['smoke_training'] and rid!='private_2d_segmentation_train':reasons.append('training_not_in_selected_route')
        if r['ground_truth_masks'] and r['prediction_masks']:
            reasons.append('distinct_ground_truth_and_prediction_delivery_not_registered')
        elif r['ground_truth_masks'] and rid!='existing_result_delivery':
            reasons.append('ground_truth_file_delivery_not_in_selected_route')
    proposed=d['decision']
    if reasons:
        d.update(decision='no_registered_route',route_id=None,clarification_fields=[],
                 reasoning='Required deliverables exceed the selected workflow: '+', '.join(reasons))
    return d,dict(requirements=deepcopy(r),proposed_decision=proposed,proposed_route_id=rid,
                  capability_violations=reasons,final_decision=d['decision'],final_route_id=d['route_id'],
                  text_or_request_id_matching=False)
