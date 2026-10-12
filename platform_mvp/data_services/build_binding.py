"""Create an explicit development binding from an existing inference summary."""
import argparse
from pathlib import Path
from platform_mvp.data_services.service import read_json, write_json, digest, ContractError, case_key, within


def build(summary, dataset_id, run_id, images=None, masks=None, overlays=None, case_ids=None):
    summary=Path(summary).resolve();doc=read_json(summary)
    metrics=doc.get('metrics_per_case')
    if not isinstance(metrics,dict) or not metrics:raise ContractError('per_case_metrics_missing')
    canonical={case_key(str(k)):k for k in metrics}
    if len(canonical)!=len(metrics):raise ContractError('ambiguous_metric_case_id')
    chosen=case_ids or sorted(canonical)
    if any(c not in canonical or '/' in c or '\\' in c for c in chosen):raise ContractError('invalid_case_selection')
    artifacts=[dict(kind='summary',path=str(summary),sha256=digest(summary))]
    for kind,folder in [('image',images),('mask',masks),('overlay',overlays)]:
        if folder is None:continue
        for cid in chosen:
            candidates=[Path(folder)/(cid+'.png')]
            if kind=='image':candidates.append(Path(folder)/(cid+'_0000.png'))
            present=[p for p in candidates if p.is_file()]
            if len(present)!=1:raise ContractError('missing_or_ambiguous_'+kind+':'+cid)
            p=present[0].resolve();artifacts.append(dict(kind=kind,case_id=cid,path=str(p),sha256=digest(p)))
    return dict(schema_version=1,dataset_id=dataset_id,run_id=run_id,artifacts=artifacts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--dataset-id',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--images',type=Path);p.add_argument('--masks',type=Path);p.add_argument('--overlays',type=Path);p.add_argument('--case-ids',nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=build(a.summary,a.dataset_id,a.run_id,a.images,a.masks,a.overlays,a.case_ids)
    roots=[x.resolve() for x in [a.summary.parent,a.images,a.masks,a.overlays] if x]
    if a.output.exists() or any(within(a.output.resolve(), x) for x in roots):p.error('output must be new and outside source directories')
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)

if __name__=='__main__':main()
