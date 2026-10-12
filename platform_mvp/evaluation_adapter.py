"""Read-only preflight and normalization for decision evaluation.

Never reads reference answers, starts a model, repairs input, or grades proposal
prose. A bundle records the native plan separately from the preflight decision.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from platform_mvp.route_preflight import preflight

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2026-10-02-eval-adapter-v2-shared-gate'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def normalize_bundle(bundle: dict) -> dict:
    """Normalize native machine fields; no reference labels or free-text grading."""
    plan = bundle.get('plan') or {}
    intent = plan.get('intent') or {}
    selected = plan.get('selected_method') or {}
    gate = bundle.get('preflight') or plan.get('preflight') or {}
    result = {'id': bundle['id'], 'run_status': 'ok', 'action': None, 'route_id': None,
              'dataset_id': gate.get('dataset_id'), 'task': intent.get('task_family'),
              'modality': intent.get('modality'), 'mode': intent.get('mode'),
              'backend': gate.get('backend'), 'execution_permission': None,
              'reason_codes': [], 'clarification_fields': [],
              'actual_execution_started': None, 'normalization_version': VERSION,
              'candidate_route_id': selected.get('id'), 'normalization_errors': []}
    if bundle.get('run_status') != 'ok':
        result['run_status'] = bundle.get('run_status', 'error')
        return result
    if plan.get('online_call_required') and (plan.get('agent_call') or {}).get('status') != 'ok':
        result.update(run_status='error', normalization_errors=['required_online_call_failed'])
        return result
    if bundle.get('preflight') and plan.get('preflight') and bundle['preflight'] != plan['preflight']:
        result.update(run_status='invalid', normalization_errors=['preflight_conflicts_with_native_plan'])
        return result
    if selected.get('id') == 'platform_clarification':
        fields = []
        if intent.get('dataset') == 'unknown': fields.append('dataset_id')
        if intent.get('task_family') == 'unknown': fields.append('task')
        if not fields:
            result.update(run_status='invalid', normalization_errors=['clarification_fields_not_structured'])
        else:
            result.update(action='clarify', execution_permission=False, clarification_fields=fields)
        return result
    if plan.get('executable') is True and selected.get('status') == 'executable':
        if gate.get('candidate_route_id') != selected.get('id'):
            result.update(run_status='invalid', normalization_errors=['preflight_route_mismatch'])
        elif gate.get('status') in {'route_ready', 'clarify', 'block'}:
            action = gate['status']
            result.update(action=action, execution_permission=gate.get('execution_permission'),
                          reason_codes=gate.get('reason_codes', []), clarification_fields=gate.get('clarification_fields', []))
            if action == 'route_ready': result['route_id'] = selected['id']
        else:
            result.update(run_status='invalid', normalization_errors=['preflight_not_established'])
        return result
    # Rejection/proposal proves "no route" only with explicit structured lookup evidence.
    if (plan.get('registry_lookup_performed') is True and plan.get('registry_match_id') is None
            and intent.get('dataset') != 'unknown' and intent.get('task_family') != 'unknown'
            and plan.get('executable') is False):
        result.update(action='no_registered_route', execution_permission=False, reason_codes=['task_not_registered'])
        return result
    result.update(run_status='invalid', normalization_errors=['no_unambiguous_primary_action'])
    return result


def write_bundle_and_prediction(bundle: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'raw_bundle.json'
    raw.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + '\n')
    pred = normalize_bundle(bundle)
    pred.update(raw_artifact=str(raw.resolve()), raw_sha256=sha256(raw),
                normalizer_sha256=sha256(__file__))
    (directory / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False, indent=2) + '\n')
    return pred
