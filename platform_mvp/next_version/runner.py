"""Unified four-route planner, explicit context and fail-closed execution."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from argparse import Namespace
from platform_mvp.online_agent import call_planner, validate_settings
from platform_mvp.route_preflight import preflight
from platform_mvp.evaluation_adapter import normalize_bundle
from platform_mvp.data_services.service import read_json, write_json, within, ROUTES, execute as execute_service

HERE=Path(__file__).resolve().parent
REGISTRY=HERE/'registry.json'
FIELDS={'task','dataset_id','data_root','result_manifest','requested_case_ids'}
CARD_KEYS={'data_card_id','dataset_id','manifest_id','image_type','annotation_description','label_values_observed','split_summary','image_mask_shape_mismatch_count','unreadable_file_count','dataset_json_present','patient_split_check','known_missing_information'}
CONTEXT_KEYS={'data_card','data_root','result_manifest','selected_case_ids','selection_required','expected_run_id','tdd_case_only','dentalclaw_python'}


def validate_decision(value):
    required={'decision','route_id','clarification_fields','parameters','confidence','reasoning'}
    if not isinstance(value,dict) or set(value)!=required:raise ValueError('decision_schema')
    if value['decision'] not in {'use_registry','clarify','no_registered_route'}:raise ValueError('decision')
    routes={m['id'] for m in read_json(REGISTRY)['methods']}
    if value['decision']=='use_registry':
        if value['route_id'] not in routes:raise ValueError('unknown_route')
    elif value['route_id'] is not None:raise ValueError('unexpected_route')
    fields=value['clarification_fields']
    if not isinstance(fields,list) or any(not isinstance(x,str) or x not in FIELDS for x in fields):raise ValueError('clarification_fields')
    if (value['decision']=='clarify') != bool(fields):raise ValueError('clarification_mismatch')
    p=value['parameters']
    if not isinstance(p,dict) or set(p)!={'case_ids','artifact_kinds','table','selection_required'}:raise ValueError('parameters')
    for k in ['table','selection_required']:
        if type(p[k]) is not bool:raise ValueError(k)
    if not isinstance(p['case_ids'],list) or any(not isinstance(x,str) or not x or any(c in x for c in '/\\\x00') for x in p['case_ids']):raise ValueError('case_ids')
    if len(set(p['case_ids']))!=len(p['case_ids']):raise ValueError('duplicate_case_ids')
    if not isinstance(p['artifact_kinds'],list) or any(x not in {'summary','image','mask','overlay'} for x in p['artifact_kinds']):raise ValueError('artifact_kinds')
    c=value['confidence']
    if type(c) not in (int,float) or not 0<=c<=1 or not isinstance(value['reasoning'],str):raise ValueError('reasoning_confidence')
    return value


def validate_context(context):
    if not isinstance(context,dict) or set(context)-CONTEXT_KEYS:raise ValueError('context_schema')
    if not isinstance(context.get('data_card',{}),dict):raise ValueError('data_card')
    if set(context.get('data_card',{}))-CARD_KEYS:raise ValueError('data_card_fields_not_allowlisted')
    for key in ['data_root','result_manifest','expected_run_id','dentalclaw_python']:
        if key in context and context[key] is not None and not isinstance(context[key],str):raise ValueError(key)
    for key in ['selection_required','tdd_case_only']:
        if key in context and type(context[key]) is not bool:raise ValueError(key)
    ids=context.get('selected_case_ids',[])
    if not isinstance(ids,list) or any(not isinstance(x,str) or not x or '/' in x or '\\' in x for x in ids):raise ValueError('selected_case_ids')
    return context


def prompt_for(request,context,coverage_check=False):
    facts={'data_card':context.get('data_card',{}),'data_root_bound':bool(context.get('data_root')),
           'selected_case_ids':context.get('selected_case_ids',[]),'selection_required':context.get('selection_required',False),
           'result_manifest_bound':bool(context.get('result_manifest')),'tdd_case_only':context.get('tdd_case_only',False)}
    # Only aggregate identities/availability leave the machine, never images or local paths.
    if context.get('result_manifest'):
        try:
            b=read_json(context['result_manifest'])
            facts['existing_results']={'dataset_id':b.get('dataset_id'),'run_id':b.get('run_id'),
                'artifact_kinds':sorted({a['kind'] for a in b['artifacts']}),
                'available_case_ids':sorted({a['case_id'] for a in b['artifacts'] if a.get('case_id')})}
        except (OSError,ValueError,TypeError,KeyError):facts['result_manifest_readable']=False
    prompt_name = 'planner_prompt_coverage.txt' if coverage_check else 'planner_prompt.txt'
    prompt = (HERE/prompt_name).read_text()
    if not context.get('data_card',{}).get('dataset_id'):
        prompt += (HERE/'missing_identity_rule.txt').read_text()
    return prompt+'\nINPUT_JSON:\n'+json.dumps({'request':request,'facts':facts},ensure_ascii=False)


def build_plan(request,context,decision,registry):
    d=validate_decision(decision);c=validate_context(context);card=c.get('data_card',{})
    rid=d['route_id'];method=next((m for m in registry['methods'] if m['id']==rid),None)
    dataset=card.get('dataset_id')
    native_dataset='TDD' if dataset=='Dataset501_TDDTeethBinary2D' else 'Private2D' if dataset else 'unknown'
    task=method['task_family'] if method else 'unknown'
    mode=method['allowed_modes'][0] if method else 'unknown'
    chosen=c.get('selected_case_ids') or d['parameters']['case_ids']
    plan=dict(intent=dict(raw_intent=request,dataset=native_dataset,task_family=task,modality='2d',mode=mode,
                          segmentation_target='binary'),selected_method=method,executable=bool(method),
              registry_lookup_performed=True,registry_match_id=rid,planner_decision=d,
              execution=dict(result_manifest=c.get('result_manifest'),requested_case_ids=chosen,
                             requested_case_id=chosen[0] if len(chosen)==1 else None,
                             selection_required=c.get('selection_required',False) or d['parameters']['selection_required'],
                             artifact_kinds=d['parameters']['artifact_kinds'] or None,table=d['parameters']['table'],
                             expected_dataset_id=dataset,expected_run_id=c.get('expected_run_id')))
    if not method:return plan
    def gate(status,reason=None,fields=None):
        return dict(status=status,candidate_route_id=rid,execution_permission=False,dataset_id=None,backend=None,
                    reason_codes=[reason] if reason else [],clarification_fields=fields or [])
    if c.get('selected_case_ids') and d['parameters']['case_ids'] and c['selected_case_ids']!=d['parameters']['case_ids']:
        plan['preflight']=gate('block','conflicting_case_selection');return plan
    if rid not in ROUTES and native_dataset=='unknown':
        plan['preflight']=gate('clarify',fields=['dataset_id']);return plan
    # Ported from the refrozen E2 gate: case selection cannot fix a known dataset mismatch.
    if rid=='tdd_2d_segmentation_infer_report' and native_dataset=='Private2D':
        plan['preflight']=gate('block','route_contract_mismatch');return plan
    if rid=='tdd_2d_segmentation_infer_report' and not chosen:
        plan['preflight']=gate('clarify',fields=['requested_case_ids']);return plan
    if rid=='tdd_2d_segmentation_infer_report' and len(chosen)>1:
        plan['preflight']=gate('block','single_report_case_required');return plan
    if rid=='tdd_2d_segmentation_infer_report' and c.get('result_manifest') and not c.get('data_root'):
        plan['preflight']=gate('clarify',fields=['data_root']);return plan
    # Never accept a claimed named dataset for a user package with a different identity.
    try:
        g=preflight(plan,registry,Path(c['data_root']) if c.get('data_root') else None)
        expected=('manifest:'+card['manifest_id']) if rid=='binary2d_package_qc' and card.get('manifest_id') else dataset if rid!='binary2d_package_qc' else None
        if expected and g.get('dataset_id') and g['dataset_id']!=expected:
            g=gate('block','dataset_identity_mismatch')
        plan['preflight']=g
    except (OSError,ValueError,TypeError,KeyError):plan['preflight']=gate('block','preflight_unavailable')
    return plan


def normalize(plan,call=None):
    p=normalize_bundle(dict(id='development',run_status='ok',plan=plan))
    p['normalization_version']='four-routes-release-port-20261012'
    d=plan.get('planner_decision',{})
    if d.get('decision') in {'clarify','no_registered_route'}:
        p.update(run_status='ok',action=d['decision'],execution_permission=False,route_id=None,candidate_route_id=None,
                 normalization_errors=[],reason_codes=['task_not_registered'] if d['decision']=='no_registered_route' else [],
                 clarification_fields=d['clarification_fields'])
    if call and call.get('status')!='ok':
        p.update(run_status='error',action=None,execution_permission=None,route_id=None,normalization_errors=['online_planner_failed'])
    return p


def dispatch(plan,registry,context,out,allow_model_execution=False):
    if plan.get('preflight',{}).get('status')!='route_ready':return dict(status='not_started',actual_execution_started=False)
    rid=plan['selected_method']['id']
    if rid in ROUTES:
        return execute_service(plan,registry,Path(out)/'artifacts',Path(context['data_root']) if context.get('data_root') else None)
    if not allow_model_execution:return dict(status='model_execution_not_enabled',actual_execution_started=False)
    from platform_mvp import run_platform_mvp as legacy
    args=Namespace(private_data_root=context.get('data_root'),registry=REGISTRY,case_id=plan['execution'].get('requested_case_id'),
                   execute=True,dentalclaw_python=context.get('dentalclaw_python') or legacy.DEFAULT_DENTALCLAW_PYTHON,
                   reuse_fullflow_run=None,reuse_inference=False,tdd_case_only=context.get('tdd_case_only',False))
    result=legacy.execute_selected_route(plan=plan,args=args,run_dir=Path(out)/'model_execution')
    # A returned command proves subprocess launch was attempted, not completion of asynchronous training.
    result['actual_execution_started']=bool(result.get('command'))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intent',required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--settings',type=Path,default=HERE/'settings.json')
    p.add_argument('--decision-json',type=Path,help='Explicit offline replay for engineering tests only; never fallback')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-model-execution',action='store_true')
    p.add_argument('--coverage-check',action='store_true',help='Post-evaluation mandatory-deliverable check; changes the planner schema')
    a=p.parse_args(argv);c=validate_context(read_json(a.context));out=a.out_dir.resolve()
    validator = validate_decision
    if a.coverage_check:
        from platform_mvp.next_version.coverage_contract import validate_decision as validator, apply_coverage_contract
    sources=[Path(x).resolve() for x in [c.get('data_root')] if x]
    if c.get('result_manifest'):
        sources.append(Path(c['result_manifest']).resolve().parent)
        try:sources += [(Path(c['result_manifest']).resolve().parent / x['path']).resolve().parent for x in read_json(c['result_manifest'])['artifacts']]
        except (OSError,ValueError,KeyError,TypeError):pass
    if out.exists() or any(within(out,s) or within(s,out) for s in sources):p.error('output must be fresh and outside inputs')
    out.mkdir(parents=True,exist_ok=False);registry=read_json(REGISTRY);call=None
    if a.decision_json:
        d=validator(read_json(a.decision_json));write_json(out/'offline_replay.json',{'source':str(a.decision_json),'not_online_validation':True})
    else:
        settings=read_json(a.settings);validate_settings(settings);key=os.environ.get(settings['credential_env'])
        if key is None and settings['provider']=='deepseek':
            from platform_mvp.run_platform_mvp import _load_deepseek_api_key
            key=_load_deepseek_api_key()
        call=call_planner(prompt_for(a.intent,c,a.coverage_check),settings=settings,api_key=key,registry_available=True,
                          audit_dir=out/'online_agent',decision_validator=validator)
        d=call.get('decision')
        if d is None:
            write_json(out/'prediction.json',dict(run_status='error',action=None,execution_permission=None,actual_execution_started=False,planner_status=call['status']))
            print(json.dumps({'status':'planner_failed','planner_status':call['status']}));return 1
    write_json(out/'runtime_version.json',{'version':'release-port-20261012','routing_base':'refrozen-v8.1','coverage_check':a.coverage_check,'frozen_evaluation_reproduction':False})
    if a.coverage_check:
        d,audit=apply_coverage_contract(d)
        write_json(out/'coverage_audit.json',audit)
    plan=build_plan(a.intent,c,d,registry);result=None
    if a.execute:result=dispatch(plan,registry,c,out,a.allow_model_execution)
    pred=normalize(plan,call);pred['actual_execution_started']=result.get('actual_execution_started') if result else None
    write_json(out/'platform_plan.json',plan);write_json(out/'prediction.json',pred)
    if result:write_json(out/'execution_result.json',result)
    print(json.dumps({'action':pred['action'],'route_id':pred['route_id'],'execution_status':result['status'] if result else 'not_started'}))
    return 1 if pred['run_status']!='ok' or result and result['status']=='failed' else 0

if __name__=='__main__':raise SystemExit(main())
