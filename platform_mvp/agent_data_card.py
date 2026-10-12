"""Build aggregate input facts without transmitting image or patient records."""
import hashlib
import json
from pathlib import Path


def card_from_validation(report, *, dataset_id=None, card_id=None):
    manifest = report['manifest']
    records = manifest['files']
    images = {(r['split'], r['case_id']): r for r in records if r['kind'] == 'image'}
    masks = {(r['split'], r['case_id']): r for r in records if r['kind'] == 'mask'}
    mismatches = sum(images[k].get('shape') != masks[k].get('shape') for k in images.keys() & masks.keys())
    values = sorted({v for r in masks.values() for v in r.get('label_values', [])})
    card = {'data_card_id': card_id or 'manifest:' + report['manifest_id'],
            'dataset_id': dataset_id or 'manifest:' + report['manifest_id'],
            'manifest_id': report['manifest_id'], 'image_type': '2d_raster',
            'annotation_description': 'pixel masks' if masks else 'no pixel masks observed',
            'label_values_observed': values, 'split_summary': report['split_counts'],
            'image_mask_shape_mismatch_count': mismatches,
            'unreadable_file_count': sum('shape' not in r for r in records),
            'dataset_json_present': (Path(report['data_root'])/'dataset.json').is_file() if report.get('data_root') else None,
            'patient_split_check': manifest['patient_split_check'],
            'known_missing_information': [],
            'source_note': 'Input facts only; cohort provenance is recorded locally. No independent clinical validation is asserted.'}
    return card


def missing_card(*, card_id, dataset_known=False):
    return {'data_card_id': card_id, 'dataset_id': None, 'manifest_id': None,
            'image_type': None, 'annotation_description': None,
            'known_missing_information': ['data_root'] if dataset_known else ['dataset_id', 'data_root']}


def card_digest(card):
    return hashlib.sha256(json.dumps(card, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
