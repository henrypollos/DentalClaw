"""Opt-in deterministic development entry point, no LLM/network/model calls."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from platform_mvp.data_services.service import REGISTRY, execute, service_preflight, read_json, write_json, ContractError, within
from platform_mvp.evaluation_adapter import normalize_bundle


def identify_task(text):
    # Narrow, inspectable surface rules. Not evidence of general intent understanding.
    t=text.lower()
    # Ignore explicit prohibitions before rejecting unsupported affirmative work.
    t=re.sub(r'(?:不需要|不要|不用|不再|无需)[^，。；;]{0,12}(?:训练|推理|重新计算|制作表格|跑模型)', '', t)
    t=re.sub(r'\b(?:do not|without|no need to)\s+(?:recompute|retrain|train|run inference)\b', '', t)
    if any(x in t for x in ['bootstrap','置信区间','异常类型','空掩膜','内部孔洞','孔洞筛查','孔洞','边界 f1','校准','训练','重新计算','最差','最优','阈值扫描','排序','后处理','分层','contour','train ','recompute','calibrat']):
        return 'unsupported'
    qc=any(x in t for x in ['配对','数据质量','标签值','尺寸一致','不可读','qc','pairing','data quality'])
    results=any(x in t for x in ['已有','已保存','已经保存','叠加图','指标文件','结果文件','已有结果','existing result','saved result','overlay'])
    if qc and results:return 'ambiguous'
    if qc:return 'binary2d_qc'
    if results:return 'result_delivery'
    return 'ambiguous'


def build_plan(args):
    registry=read_json(args.registry)
    detected=identify_task(args.intent)
    task=detected if args.task=='auto' else args.task
    if detected=='unsupported':task='unsupported'
    artifact_kinds=args.artifact_kinds
    if artifact_kinds is None and task=='result_delivery':
        if '叠加图' in args.intent or 'overlay' in args.intent.lower():
            artifact_kinds=['image','overlay']
            if '指标' in args.intent or 'metrics' in args.intent.lower(): artifact_kinds.append('summary')
        elif any(x in args.intent for x in ['指标文件','结果文件','整体结果']):
            artifact_kinds=['summary']
    format_text=re.sub(r'(?:不需要|不要|不用|不再|无需)[^，。；;]*', '', args.intent).lower()
    wants_table=args.table or any(x in format_text for x in ['逐例表','指标表','整理成表','表格','csv'])
    mode={'binary2d_qc':'qc','result_delivery':'deliver'}.get(task,'unknown')
    rid={'binary2d_qc':'binary2d_package_qc','result_delivery':'existing_result_delivery'}.get(task)
    method=next((m for m in registry['methods'] if m['id']==rid),None)
    # The binding, not language keywords, establishes concrete data identity.
    plan=dict(intent=dict(raw_intent=args.intent,dataset='bound_input',task_family=task,modality='2d',mode=mode),
              selected_method=method,executable=bool(method),registry_lookup_performed=True,registry_match_id=rid,
              execution=dict(result_manifest=str(args.result_manifest) if args.result_manifest else None,
                             requested_case_ids=args.case_ids,selection_required=args.selection_required or any(x in args.intent for x in ['指定','已选定']),
                             artifact_kinds=artifact_kinds,table=wants_table,
                             expected_dataset_id=args.expected_dataset_id,expected_run_id=args.expected_run_id))
    if task=='ambiguous':
        plan['intent']['task_family']='unknown';plan['selected_method']={'id':'platform_clarification'}
    if method:plan['preflight']=service_preflight(plan,registry,args.data_root)
    return plan,registry


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intent',required=True)
    p.add_argument('--task',choices=['auto','binary2d_qc','result_delivery'],default='auto')
    p.add_argument('--registry',type=Path,default=REGISTRY)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--result-manifest',type=Path)
    p.add_argument('--expected-dataset-id');p.add_argument('--expected-run-id')
    p.add_argument('--case-ids',nargs='+');p.add_argument('--selection-required',action='store_true')
    p.add_argument('--artifact-kinds',nargs='+',choices=['summary','image','mask','overlay'])
    p.add_argument('--table',action='store_true',help='Format existing scalar metrics, never recompute')
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    # Fresh audit destination must not mutate a QC source or result source directory.
    target=a.out_dir.resolve()
    sources=[x.resolve() for x in [a.data_root] if x]
    if a.result_manifest:
        sources.append(a.result_manifest.resolve().parent)
        try:
            for x in read_json(a.result_manifest).get('artifacts',[]):
                sources.append((a.result_manifest.resolve().parent/x['path']).resolve().parent)
        except (OSError,ValueError,TypeError,KeyError,AttributeError):pass
    if target.exists() or any(target==s or within(target, s) or within(s, target) for s in sources):
        p.error('out-dir must be fresh and outside input directories')
    plan,reg=build_plan(a)
    target.mkdir(parents=True,exist_ok=False)
    result=None
    if a.execute and plan.get('preflight',{}).get('status')=='route_ready':
        result=execute(plan,reg,target/'artifacts',a.data_root)
    prediction=normalize_bundle(dict(id='development',run_status='ok',plan=plan))
    prediction['actual_execution_started']=result['actual_execution_started'] if result else (False if a.execute else None)
    write_json(target/'platform_plan.json',plan);write_json(target/'prediction.json',prediction)
    if result:write_json(target/'execution_result.json',result)
    print(json.dumps(dict(action=prediction['action'],execution_status=result['status'] if result else 'not_started',out_dir=str(target))))
    return 1 if result and result['status']=='failed' or prediction['run_status']!='ok' else 0


if __name__=='__main__':raise SystemExit(main())
