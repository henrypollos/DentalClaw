"""Read-only result delivery and binary-raster QC. No model or network calls."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name('registry.json')
ROUTES = {'existing_result_delivery', 'binary2d_package_qc'}
KINDS = {'summary', 'image', 'mask', 'overlay'}


class ContractError(ValueError):
    pass


def case_key(value):
    return value[:-4] if value.endswith('.png') else value


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    def reject(value):
        raise ContractError('nonfinite_json')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def binding(path, execution):
    """Explicit file identities are required; no inference of dataset/run identity."""
    p = Path(path).resolve()
    doc = read_json(p)
    if not isinstance(doc, dict) or doc.get('schema_version') != 1:
        raise ContractError('invalid_result_manifest')
    if not all(isinstance(doc.get(k), str) and doc[k] for k in ('dataset_id', 'run_id')):
        raise ContractError('result_identity_missing')
    if execution.get('expected_dataset_id') and doc['dataset_id'] != execution['expected_dataset_id']:
        raise ContractError('dataset_identity_mismatch')
    if execution.get('expected_run_id') and doc['run_id'] != execution['expected_run_id']:
        raise ContractError('run_identity_mismatch')
    artifacts = doc.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractError('result_artifacts_missing')
    seen = set(); checked = []
    for a in artifacts:
        if not isinstance(a, dict) or a.get('kind') not in KINDS:
            raise ContractError('invalid_artifact_kind')
        cid = a.get('case_id')
        if a['kind'] != 'summary' and (not isinstance(cid, str) or not cid or any(c in cid for c in '/\\\x00')):
            raise ContractError('invalid_case_id')
        key = (a['kind'], cid)
        if key in seen:
            raise ContractError('duplicate_case_artifact')
        seen.add(key)
        if not isinstance(a.get('path'), str) or not isinstance(a.get('sha256'), str):
            raise ContractError('artifact_binding_incomplete')
        source = (p.parent / a['path']).resolve()
        if not source.is_file():
            raise ContractError('result_artifact_missing')
        if digest(source) != a['sha256']:
            raise ContractError('artifact_hash_mismatch')
        checked.append(dict(a, path=str(source)))
    kinds = set(execution.get('artifact_kinds') or {a['kind'] for a in checked})
    if not kinds <= KINDS or not kinds:
        raise ContractError('invalid_requested_artifact_kinds')
    cases = sorted({a['case_id'] for a in checked if a.get('case_id') is not None})
    requested = execution.get('requested_case_ids') or cases
    if len(requested) != len(set(requested)) or any(c not in cases for c in requested):
        raise ContractError('requested_case_missing_or_duplicate')
    for kind in kinds:
        if kind == 'summary':
            if not any(a['kind'] == kind for a in checked):
                raise ContractError('requested_artifact_missing')
        elif not requested or any((kind, cid) not in seen for cid in requested):
            raise ContractError('requested_artifact_missing')
    selected = [a for a in checked if a['kind'] in kinds and (a['kind'] == 'summary' or a['case_id'] in requested)]
    if execution.get('table'):
        summaries = [a for a in checked if a['kind'] == 'summary']
        if len(summaries) != 1:
            raise ContractError('metrics_summary_required')
        metrics = read_json(summaries[0]['path']).get('metrics_per_case')
        if not isinstance(metrics, dict) or not metrics or any(not isinstance(v, dict) for v in metrics.values()):
            raise ContractError('per_case_metrics_missing')
        # File-based case keys (e.g. synthetic_Ts_0.png) remain unchanged.
        metric_cases = {k: case_key(str(k)) for k in metrics}
        if len(set(metric_cases.values())) != len(metric_cases):
            raise ContractError('ambiguous_metric_case_id')
        if requested and any(c not in metric_cases.values() for c in requested):
            raise ContractError('selected_case_metrics_missing')
    return dict(doc, artifacts=selected, selected_case_ids=requested,
                manifest_path=str(p), manifest_sha256=digest(p))


def service_preflight(plan, registry, data_root=None):
    selected = plan.get('selected_method') or {}; rid = selected.get('id')
    base = dict(candidate_route_id=rid, version='data-services-dev-20261008',
                execution_permission=False, dataset_id=None, backend='readonly_python',
                reason_codes=[], clarification_fields=[])
    def outcome(status, code=None, fields=None, **extra):
        return dict(base, status=status, execution_permission=status == 'route_ready',
                    reason_codes=[code] if code else [], clarification_fields=fields or [], **extra)
    method = next((m for m in registry.get('methods', []) if m.get('id') == rid), None)
    if not method or method.get('status') != 'executable' or not plan.get('executable'):
        return outcome('block', 'route_unavailable')
    intent = plan.get('intent', {}); ex = plan.get('execution', {})
    if intent.get('task_family') != method['task_family'] or intent.get('mode') not in method['allowed_modes'] or intent.get('modality') != '2d':
        return outcome('block', 'route_contract_mismatch')
    try:
        if rid == 'existing_result_delivery':
            if not ex.get('result_manifest'):
                return outcome('clarify', fields=['result_manifest'])
            if ex.get('selection_required') and not ex.get('requested_case_ids'):
                return outcome('clarify', fields=['requested_case_ids'])
            b = binding(ex['result_manifest'], ex)
            base['dataset_id'] = b['dataset_id']
            return outcome('route_ready', result_binding=b)
        if data_root is None:
            return outcome('clarify', fields=['data_root'])
        root = Path(data_root).resolve()
        if not root.is_dir():
            return outcome('block', 'data_root_missing')
        from platform_mvp.validate_private2d_package import validate_package, DEFAULT_CONTRACT
        qc = validate_package(root, DEFAULT_CONTRACT, 'inference_or_qc')
        if not qc['image_count'] and not qc['mask_count']:
            return outcome('block', 'no_inspectable_files')
        # QC can successfully discover defects. It never authorizes training.
        for split, counts in qc['split_counts'].items():
            if counts['unmatched_image_count']:
                qc['issues'].append(dict(code='incomplete_pairing', detail=split))
        qc['reason_codes'] = sorted({x['code'] for x in qc['issues']})
        base['dataset_id'] = 'manifest:' + qc['manifest_id']
        return outcome('route_ready', qc_report=qc, data_root=str(root))
    except ContractError as exc:
        return outcome('block', str(exc))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return outcome('block', 'input_unreadable_or_invalid', error_type=type(exc).__name__)


def execute(plan, registry, output, data_root=None):
    """Revalidate immediately before output; never overwrite or write into inputs."""
    gate = service_preflight(plan, registry, data_root)
    plan['preflight'] = gate
    if gate['status'] != 'route_ready':
        return dict(status=gate['status'], actual_execution_started=False, preflight=gate)
    out = Path(output).resolve()
    if out.exists():
        raise ContractError('output_must_be_fresh')
    source_roots = [Path(gate['data_root'])] if 'data_root' in gate else [Path(gate['result_binding']['manifest_path']).parent]
    if 'result_binding' in gate:
        source_roots += [Path(a['path']).parent for a in gate['result_binding']['artifacts']]
    if any(out == p or within(out, p) or within(p, out) for p in source_roots):
        raise ContractError('output_overlaps_source')
    out.mkdir(parents=True, exist_ok=False)
    try:
        if 'qc_report' in gate:
            qc = gate['qc_report']
            report = dict(schema_version=1, status='completed', audit_completed=True,
                          quality_passed=not qc['issues'], training_execution_permission=False,
                          data_root=gate['data_root'], manifest_id=qc['manifest_id'],
                          split_counts=qc['split_counts'], issues=qc['issues'],
                          reason_codes=qc['reason_codes'], patient_split_check=qc['manifest']['patient_split_check'])
            write_json(out/'qc_report.json', report)
            write_json(out/'input_manifest.json', qc['manifest'])
            with (out/'qc_issues.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=['code', 'detail']); w.writeheader(); w.writerows(qc['issues'])
            (out/'report.html').write_text('<meta charset="utf-8"><h1>Binary 2D data QC</h1><pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>', encoding='utf-8')
        else:
            b = gate['result_binding']; delivered = []
            for i, a in enumerate(b['artifacts']):
                suffix = Path(a['path']).suffix
                target = out / f'{i:04d}_{a["kind"]}{suffix}'
                shutil.copyfile(a['path'], target)
                if digest(target) != a['sha256']:
                    raise ContractError('source_changed_during_delivery')
                delivered.append(dict(a, source_path=a['path'], path=target.name))
            if plan['execution'].get('table'):
                doc = binding(plan['execution']['result_manifest'], dict(plan['execution'], artifact_kinds=['summary']))
                summary = next(a for a in doc['artifacts'] if a['kind']=='summary')
                metrics = read_json(summary['path'])['metrics_per_case']
                if b['selected_case_ids']:
                    metrics = {k:v for k,v in metrics.items() if case_key(k) in b['selected_case_ids']}
                fields = sorted({k for values in metrics.values() for k,v in values.items() if v is None or isinstance(v, (int,float,bool))})
                with (out/'per_case_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    w=csv.DictWriter(f,fieldnames=['case_id']+fields);w.writeheader()
                    for cid, values in metrics.items(): w.writerow(dict(case_id=cid, **{k:values.get(k) for k in fields}))
            report=dict(schema_version=1,status='completed',dataset_id=b['dataset_id'],run_id=b['run_id'],
                        source_manifest=b['manifest_path'],source_manifest_sha256=b['manifest_sha256'],
                        selected_case_ids=b['selected_case_ids'],artifacts=delivered,new_model_inference=False,
                        metrics_recomputed=False,table='per_case_metrics.csv' if plan['execution'].get('table') else None)
            write_json(out/'delivery_manifest.json',report)
            lines=['<meta charset="utf-8"><h1>Existing results</h1>', '<p>'+html.escape(b['dataset_id']+' / '+b['run_id'])+'</p>']
            for a in delivered:
                label=html.escape(str(a.get('case_id') or 'overall')+' / '+a['kind'])
                lines.append(f'<p>{label}: <a href="{a["path"]}">{a["path"]}</a></p>')
                if a['kind'] in {'image','overlay'}: lines.append(f'<img width="480" src="{a["path"]}" alt="{label}">')
            if report['table']: lines.append('<a href="per_case_metrics.csv">Existing per-case metrics</a>')
            (out/'report.html').write_text('\n'.join(lines),encoding='utf-8')
        return dict(status='completed',actual_execution_started=True,output_dir=str(out),preflight=gate,
                    training_started=False,inference_started=False)
    except Exception as exc:
        write_json(out/'failure.json',dict(status='failed',error_type=type(exc).__name__,reason=str(exc)))
        return dict(status='failed',actual_execution_started=True,preflight=gate,reason=str(exc))
