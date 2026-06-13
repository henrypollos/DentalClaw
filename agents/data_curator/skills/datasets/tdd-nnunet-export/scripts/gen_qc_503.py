#!/usr/bin/env python3
"""Generate QC report for Dataset503."""

import sys
from pathlib import Path

# Setup paths correctly
SCRIPT_DIR = Path(__file__).resolve().parent
# scripts -> datasets -> skills -> data_curator -> agents -> DentalClaw = 5 parents
REPO_ROOT = SCRIPT_DIR.parents[5]
TDD_SCRIPTS = REPO_ROOT / "agents" / "data_curator" / "skills" / "datasets" / "tdd-curation" / "scripts"
CORE_LIB = REPO_ROOT / "agents" / "data_curator" / "skills" / "core" / "_lib"

sys.path.insert(0, str(TDD_SCRIPTS))
sys.path.insert(0, str(CORE_LIB))

from tdd_common import load_cases
from dataset_qc import run_dataset_qc, write_report_bundle

dataset_root = Path('/data/data2/yiyang/DentalClaw/data/TDD')
nnunet_raw = Path('/data/data2/yiyang/DentalClaw/artifacts/datasets/nnUNet/nnUNet_raw/Dataset503_TDDTeeth32Class2D')
dataset_id = 503
dataset_name = 'Dataset503_TDDTeeth32Class2D'

# Get all train and test case IDs
cases_tr = [f.stem.replace('_0000', '') for f in (nnunet_raw / 'imagesTr').glob('*.png')]
cases_ts = [f.stem for f in (nnunet_raw / 'imagesTs').glob('*.png')] if (nnunet_raw / 'imagesTs').exists() else []
all_case_ids = sorted(set(cases_tr + cases_ts))

print(f"Processing {len(all_case_ids)} cases ({len(cases_tr)} train, {len(cases_ts)} test)")

# Build qc_cases from source data
qc_cases = []
source_cases = list(load_cases(dataset_root))
source_by_id = {c['case_id']: c for c in source_cases}

for idx, case_id in enumerate(all_case_ids):
    case = source_by_id.get(case_id)
    if case is None:
        print(f"Warning: Missing case {case_id}")
        continue
    
    bbox_annotations = []
    for obj in (case.get('bbox_item') or {}).get('Label', {}).get('objects', []):
        bbox_annotations.append({
            'label': str(obj.get('title', '')).strip(),
            'bbox_xyxy': obj.get('bounding box') or [],
        })
    
    polygon_annotations = []
    for obj in (case.get('polygon_item') or {}).get('Label', {}).get('objects', []):
        polygon_annotations.append({
            'label': str(obj.get('title', '')).strip(),
            'polygons': obj.get('polygons') or [],
        })
    
    raster_labels = []
    if case.get('teeth_mask') is not None:
        raster_labels.append({'role': 'teeth_mask', 'path': case['teeth_mask']})
    if case.get('maxillomandibular_mask') is not None:
        raster_labels.append({'role': 'maxillomandibular_mask', 'path': case['maxillomandibular_mask']})
    
    qc_cases.append({
        'case_id': case_id,
        'images': [{'role': 'panoramic_image', 'path': case['radiograph']}],
        'raster_labels': raster_labels,
        'bbox_annotations': bbox_annotations,
        'polygon_annotations': polygon_annotations,
        'metadata': {
            'export_split': 'train' if case_id in cases_tr else 'test',
            'export_task': 'teeth_32class',
        },
    })
    
    if (idx + 1) % 200 == 0:
        print(f"Processed {idx + 1}/{len(all_case_ids)} cases")

split_payload = {'splits': {'train': cases_tr, 'val': [], 'test': cases_ts}}

print("Running QC...")
qc_report = run_dataset_qc(
    dataset_root=dataset_root.resolve(),
    dataset_name=dataset_name,
    dataset_mode='tdd_source_export',
    cases=qc_cases,
    split_payload=split_payload,
    metadata={
        'source_dataset_root': str(dataset_root.resolve()),
        'export_dataset_root': str(nnunet_raw),
        'export_task': 'teeth_32class',
        'nnunet_dataset_id': dataset_id,
    },
)

report_paths = write_report_bundle(qc_report, report_key=f'Dataset{dataset_id}_TDDTeeth32Class2D')
print(f'\nQC report JSON: {report_paths["report_json"]}')
print(f'QC report MD: {report_paths["report_md"]}')
print(f'Status: {qc_report["summary"]["dataset_status"]}')
print(f'Cases: {qc_report["summary"]["case_count"]}, Ready: {qc_report["summary"]["ready_case_count"]}, Blocked: {qc_report["summary"]["blocked_case_count"]}')
