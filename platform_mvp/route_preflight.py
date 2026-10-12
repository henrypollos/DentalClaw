"""Shared read-only checks for planning, execution, and evaluation.

No reference answers, network, subprocesses, model loading, or input repair.
"""
from __future__ import annotations
import json
from pathlib import Path
from platform_mvp.validate_private2d_package import validate_package, DEFAULT_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_VERSION = '2026-10-03-shared-route-gate-v2-binary-target'
GATED_ROUTES = frozenset({'tdd_2d_segmentation_infer_report', 'private_2d_segmentation_train'})


def preflight(plan: dict, registry: dict, data_root: Path | None = None) -> dict:
    """Check a selected candidate, not the correctness of the research intent."""
    selected = plan.get('selected_method') or {}
    rid = selected.get('id')
    # Opt-in development registry only. The frozen v5 snapshot is unchanged.
    from platform_mvp.data_services.service import ROUTES, service_preflight
    if rid in ROUTES:
        return service_preflight(plan, registry, data_root)
    intent = plan.get('intent') or {}
    base = {'version': PREFLIGHT_VERSION, 'candidate_route_id': rid, 'execution_permission': False,
            'dataset_id': None, 'backend': None, 'reason_codes': [], 'clarification_fields': []}
    if rid not in GATED_ROUTES:
        return dict(base, status='not_applicable')
    method = next((m for m in registry.get('methods', []) if m['id'] == rid), None)
    if (not method or method.get('status') != 'executable'
            or selected.get('status') != 'executable' or plan.get('executable') is not True):
        return dict(base, status='block', reason_codes=['route_unavailable'])
    if (intent.get('modality') != '2d' or intent.get('task_family') != 'segmentation'
            or intent.get('tta') or intent.get('ensemble')
            or intent.get('segmentation_target') in {'instance', 'multiclass'}):
        return dict(base, status='block', reason_codes=['route_contract_mismatch'])
    if rid == 'private_2d_segmentation_train':
        if intent.get('dataset') != 'Private2D' or intent.get('mode') != 'private_train':
            return dict(base, status='block', reason_codes=['route_contract_mismatch'])
        if data_root is None:
            return dict(base, status='clarify', clarification_fields=['data_root'])
        root = Path(data_root).resolve()
        # Only the ordinary-raster TinyUNet executor has been verified for this route.
        if (root / 'dataset.json').exists():
            return dict(base, status='block', reason_codes=['backend_not_verified'])
        qc = validate_package(root, DEFAULT_CONTRACT, 'private_train')
        extra = []
        if not qc['split_counts'].get('val', {}).get('paired_case_count'):
            extra.append('validation_split_required')
        if not qc['split_counts'].get('test', {}).get('image_count'):
            extra.append('inference_split_required')
        base.update(dataset_id='manifest:' + qc['manifest_id'], backend='builtin_nnunet_style_2d')
    else:
        if intent.get('dataset') not in {'TDD', 'Dataset501_TDDTeethBinary2D'} or intent.get('mode') != 'inference':
            return dict(base, status='block', reason_codes=['route_contract_mismatch'])
        assets = method.get('existing_assets', {})
        if not assets.get('dataset_root'):
            return dict(base, status='block', reason_codes=['required_asset_missing'], missing_assets=['dataset_root'])
        root = (ROOT / assets['dataset_root']).resolve()
        base.update(dataset_id='Dataset501_TDDTeethBinary2D', backend='nnunetv2')
        if data_root is not None and Path(data_root).resolve() != root:
            return dict(base, status='block', reason_codes=['dataset_identity_mismatch'])
        needed = ['checkpoint', 'dataset_spec', 'task_spec']
        missing = [k for k in needed if not assets.get(k) or not (ROOT / assets[k]).is_file()]
        if missing:
            return dict(base, status='block', reason_codes=['required_asset_missing'], missing_assets=missing)
        # These are the additional fixed dependencies of the full-flow adapter.
        from mvp_fullflow import run_mvp_fullflow as fullflow
        dependencies = {
            'training_run': fullflow.DEFAULT_TRAINING_RUN,
            'delivery_status': fullflow.DEFAULT_DELIVERY_STATUS,
            'qc_json': fullflow.DEFAULT_QC_JSON,
            'inference_script': fullflow.DEFAULT_INFERENCE_SCRIPT,
            'report_script': fullflow.DEFAULT_REPORT_SCRIPT,
        }
        missing = [k for k, p in dependencies.items() if not (p.is_dir() if k == 'training_run' else p.is_file())]
        if missing:
            return dict(base, status='block', reason_codes=['required_asset_missing'], missing_assets=missing)
        try:
            spec = json.loads((ROOT / assets['dataset_spec']).read_text())
            task = json.loads((ROOT / assets['task_spec']).read_text())
            if (Path(spec['root']).resolve() != root
                    or spec.get('extra', {}).get('target_backend') != 'nnunetv2_cli'
                    or task.get('extra', {}).get('nnunet_configuration') != '2d'):
                return dict(base, status='block', reason_codes=['asset_contract_mismatch'])
        except (OSError, ValueError, TypeError, KeyError):
            return dict(base, status='block', reason_codes=['asset_contract_mismatch'])
        case_id = plan.get('execution', {}).get('requested_case_id')
        if case_id is not None and (not (root / 'imagesTs' / f'{case_id}_0000.png').is_file()
                                    or not (root / 'labelsTs' / f'{case_id}.png').is_file()):
            return dict(base, status='block', reason_codes=['requested_case_missing'])
        base['validated_assets'] = {k: str((ROOT / assets[k]).resolve()) for k in ['dataset_root', *needed]}
        qc = validate_package(root, DEFAULT_CONTRACT, 'inference_or_qc')
        extra = []
        test = qc['split_counts'].get('test', {})
        if not test.get('image_count') or test.get('paired_case_count') != test.get('image_count'):
            extra.append('test_evaluation_pairs_required')
    reasons = sorted(set(qc['reason_codes'] + extra))
    return dict(base, status='block' if reasons else 'route_ready',
                execution_permission=not reasons, reason_codes=reasons,
                input_manifest_id=qc['manifest_id'], qc_report=qc)
