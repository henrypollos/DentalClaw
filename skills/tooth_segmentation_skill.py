import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset

from schemas.records import ExperimentRecord
from schemas.specs import BudgetSpec, DatasetSpec, TaskSpec
from skills.base_skill import BaseSkill


TOOTH_32_FDI_NAMES = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]

IMAGE_SUFFIXES_2D = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
IMAGE_SUFFIXES_3D = {'.nii', '.nii.gz', '.mha', '.mhd', '.nrrd', '.npy', '.npz'}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NNUNET_RAW_ROOT = REPO_ROOT / 'artifacts' / 'datasets' / 'nnUNet' / 'nnUNet_raw'
DEFAULT_NNUNET_PREPROCESSED_ROOT = REPO_ROOT / 'artifacts' / 'datasets' / 'nnUNet' / 'nnUNet_preprocessed'
DEFAULT_NNUNET_RESULTS_ROOT = REPO_ROOT / 'artifacts' / 'models' / 'nnUNet' / 'nnUNet_results'
DEFAULT_NNUNET_TRAINER_SOURCE_ROOT = REPO_ROOT.parent / 'JoD' / 'nnUNet' / 'nnunetv2' / 'training' / 'nnUNetTrainer'
NNUNETV2_BIN_DIR = Path('/home/yiyang/miniconda3/envs/nnunetv2/bin')
NNUNET_EPOCH_TRAINERS = {
    1: 'nnUNetTrainer_1epoch',
    5: 'nnUNetTrainer_5epochs',
    10: 'nnUNetTrainer_10epochs',
    20: 'nnUNetTrainer_20epochs',
    50: 'nnUNetTrainer_50epochs',
    100: 'nnUNetTrainer_100epochs',
    250: 'nnUNetTrainer_250epochs',
    500: 'nnUNetTrainer_500epochs',
    750: 'nnUNetTrainer_750epochs',
    2000: 'nnUNetTrainer_2000epochs',
    4000: 'nnUNetTrainer_4000epochs',
    8000: 'nnUNetTrainer_8000epochs',
}
NNUNET_DYNAMIC_EPOCH_TRAINER = 'nnUNetTrainer_DentalClawEpochs'
NNUNET_DYNAMIC_EPOCH_ENV = 'DENTALCLAW_NNUNET_EPOCHS'
DENTALCLAW_NNUNET_INITIAL_LR_ENV = 'DENTALCLAW_NNUNET_INITIAL_LR'
DENTALCLAW_NNUNET_WEIGHT_DECAY_ENV = 'DENTALCLAW_NNUNET_WEIGHT_DECAY'
DENTALCLAW_NNUNET_OVERSAMPLE_ENV = 'DENTALCLAW_NNUNET_OVERSAMPLE_FOREGROUND_PERCENT'
DENTALCLAW_NNUNET_LR_SCHEDULER_ENV = 'DENTALCLAW_NNUNET_LR_SCHEDULER'
DENTALCLAW_SELECTED_GPU_ENV = 'DENTALCLAW_SELECTED_GPU'


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(_serializable(payload), handle, indent=2, ensure_ascii=False)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _coerce_dataclass_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed_fields = {item.name for item in fields(cls)}
    coerced = {key: value for key, value in payload.items() if key in allowed_fields}
    unknown = {key: value for key, value in payload.items() if key not in allowed_fields}
    if 'extra' in allowed_fields:
        merged_extra = dict(coerced.get('extra') or {})
        if cls is DatasetSpec and 'dataset_name' in unknown:
            merged_extra.setdefault('dataset_name', unknown.pop('dataset_name'))
        merged_extra.update(unknown)
        coerced['extra'] = merged_extra
    return coerced


def load_dataset_spec(path: str) -> DatasetSpec:
    return DatasetSpec(**_coerce_dataclass_payload(DatasetSpec, _load_json(path)))


def load_task_spec(path: str) -> TaskSpec:
    return TaskSpec(**_coerce_dataclass_payload(TaskSpec, _load_json(path)))


def load_budget_spec(path: str) -> BudgetSpec:
    return BudgetSpec(**_coerce_dataclass_payload(BudgetSpec, _load_json(path)))


def default_teeth32_task_spec(task_id: str = 'Teeth1to32Panoramic') -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        modality='auto',
        task_type='tooth_segmentation',
        num_classes=32,
        class_names=TOOTH_32_FDI_NAMES,
        primary_metric='mean_dice',
        extra={
            'target_backend': 'nnunet_style_2d',
            'run_best_inference': True,
            'training_requirements': {
                'learning_rates': [1e-3, 5e-4, 2e-4],
                'img_sizes': [512, 640, 768],
                'base_channels': [16, 24, 32],
                'batch_sizes': [2, 1, 1],
                'depths': [4],
                'weight_decays': [1e-4, 5e-5],
            },
        },
    )


def _collect_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_file()])


def _suffix_of(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith('.nii.gz'):
        return '.nii.gz'
    return path.suffix.lower()


def _stem_without_suffix(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith('.nii.gz'):
        return path.name[:-7]
    return path.stem


def _case_key_from_path(path: Path) -> str:
    stem = _stem_without_suffix(path)
    return re.sub(r'_\d{4}$', '', stem)


def _detect_dimension(files: Iterable[Path]) -> str:
    suffixes = {_suffix_of(path) for path in files}
    if suffixes & IMAGE_SUFFIXES_3D:
        return '3d'
    if suffixes & IMAGE_SUFFIXES_2D:
        return '2d'
    return 'unknown'


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return 'n/a'
    return '{:.4f}'.format(float(value))


def _dataset_signature(dataset_spec: DatasetSpec, task_spec: TaskSpec) -> str:
    raw = json.dumps(
        {
            'root': dataset_spec.root,
            'imagesTr': dataset_spec.imagesTr,
            'labelsTr': dataset_spec.labelsTr,
            'imagesVal': dataset_spec.imagesVal,
            'labelsVal': dataset_spec.labelsVal,
            'imagesTs': dataset_spec.imagesTs,
            'task_id': task_spec.task_id,
            'num_classes': task_spec.num_classes,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def _sanitize_name(name: str) -> str:
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in {'_', '-'}:
            cleaned.append(ch)
        else:
            cleaned.append('_')
    return ''.join(cleaned).strip('_') or 'task'


def _is_nnunet_raw_dataset(root: Path) -> bool:
    return (root / 'dataset.json').is_file() and (root / 'imagesTr').is_dir() and (root / 'labelsTr').is_dir()


def _infer_nnunet_dataset_id(root: Path, dataset_spec: DatasetSpec) -> Optional[int]:
    value = dataset_spec.extra.get('nnunet_dataset_id')
    if value is not None:
        return int(value)
    match = re.match(r'Dataset(\d{3})_', root.name)
    if match:
        return int(match.group(1))
    return None


def _resolve_nnunet_roots(root: Path, dataset_spec: DatasetSpec) -> Dict[str, Path]:
    raw_root = Path(dataset_spec.extra.get('nnunet_raw') or dataset_spec.extra.get('nnUNet_raw') or (
        root.parent if root.parent.name == 'nnUNet_raw' else DEFAULT_NNUNET_RAW_ROOT
    )).resolve()
    preprocessed_root = Path(
        dataset_spec.extra.get('nnunet_preprocessed')
        or dataset_spec.extra.get('nnUNet_preprocessed')
        or DEFAULT_NNUNET_PREPROCESSED_ROOT
    ).resolve()
    results_root = Path(
        dataset_spec.extra.get('nnunet_results')
        or dataset_spec.extra.get('nnUNet_results')
        or DEFAULT_NNUNET_RESULTS_ROOT
    ).resolve()
    return {
        'nnUNet_raw': raw_root,
        'nnUNet_preprocessed': preprocessed_root,
        'nnUNet_results': results_root,
    }


def _nnunet_cli_targets(command_name: str) -> List[str]:
    mapping = {
        'train': ['nnUNetv2_train', 'nnUNet_train'],
        'predict': ['nnUNetv2_predict', 'nnUNet_predict'],
        'plan_and_preprocess': ['nnUNetv2_plan_and_preprocess'],
    }
    return mapping[command_name]


def _resolve_nnunet_cli(command_name: str) -> Optional[str]:
    for candidate in _nnunet_cli_targets(command_name):
        found = shutil.which(candidate)
        if found:
            return found
        env_candidate = NNUNETV2_BIN_DIR / candidate
        if env_candidate.is_file():
            return str(env_candidate)
    return None


def _default_nnunet_configuration(detected_dimension: str) -> str:
    return '2d' if detected_dimension == '2d' else '3d_fullres'


def _resolve_nnunet_trainer(exp_config: Dict[str, Any], task_spec: TaskSpec) -> Tuple[str, Dict[str, str], Optional[int]]:
    explicit_trainer = exp_config.get('nnunet_trainer') or task_spec.extra.get('nnunet_trainer')
    requested_epochs = exp_config.get('epochs')
    if requested_epochs is not None:
        requested_epochs = int(requested_epochs)
        if requested_epochs <= 0:
            raise RuntimeError('nnUNet 训练轮数必须是正整数。')

    if explicit_trainer:
        return str(explicit_trainer), {}, requested_epochs
    if requested_epochs is None:
        return 'nnUNetTrainer', {}, None
    if requested_epochs in NNUNET_EPOCH_TRAINERS:
        return NNUNET_EPOCH_TRAINERS[requested_epochs], {}, requested_epochs
    return NNUNET_DYNAMIC_EPOCH_TRAINER, {NNUNET_DYNAMIC_EPOCH_ENV: str(requested_epochs)}, requested_epochs


def _resolve_nnunet_trainer_env(exp_config: Dict[str, Any]) -> Dict[str, str]:
    if exp_config.get('materialize_trainer_subclass'):
        return {}
    mapping = {
        'epochs': NNUNET_DYNAMIC_EPOCH_ENV,
        'initial_lr': DENTALCLAW_NNUNET_INITIAL_LR_ENV,
        'weight_decay': DENTALCLAW_NNUNET_WEIGHT_DECAY_ENV,
        'oversample_foreground_percent': DENTALCLAW_NNUNET_OVERSAMPLE_ENV,
        'lr_scheduler': DENTALCLAW_NNUNET_LR_SCHEDULER_ENV,
    }
    trainer_env: Dict[str, str] = {}
    for key, env_name in mapping.items():
        value = exp_config.get(key)
        if value is not None:
            trainer_env[env_name] = str(value)
    extra_env = exp_config.get('trainer_env_overrides') or {}
    for key, value in extra_env.items():
        env_name = mapping.get(str(key), str(key))
        trainer_env[env_name] = str(value)
    return trainer_env


def _sanitize_python_identifier(value: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z_]+', '_', str(value))
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if not cleaned:
        cleaned = 'generated'
    if cleaned[0].isdigit():
        cleaned = 'T_{}'.format(cleaned)
    return cleaned


def _build_generated_trainer_name(exp_config: Dict[str, Any], work_dir: str) -> str:
    existing = exp_config.get('generated_trainer_name')
    if existing:
        return _sanitize_python_identifier(str(existing))
    trial_name = exp_config.get('trial_name') or Path(work_dir).name
    digest = hashlib.sha1(str(Path(work_dir).resolve()).encode('utf-8')).hexdigest()[:8]
    return 'nnUNetTrainer_DentalClawAdaptive_{}_{}'.format(
        _sanitize_python_identifier(str(trial_name)),
        digest,
    )


def _materialize_nnunet_trainer_subclass(
    exp_config: Dict[str, Any],
    task_spec: TaskSpec,
    work_dir: str,
) -> Optional[Dict[str, Any]]:
    if not exp_config.get('materialize_trainer_subclass'):
        return None

    trainer_root = Path(
        task_spec.extra.get('nnunet_trainer_source_root')
        or DEFAULT_NNUNET_TRAINER_SOURCE_ROOT
    ).resolve()
    trainer_root.mkdir(parents=True, exist_ok=True)

    parent_trainer = _sanitize_python_identifier(
        str(
            exp_config.get('inherits_from_trainer')
            or exp_config.get('nnunet_trainer')
            or task_spec.extra.get('nnunet_followup_parent_trainer')
            or 'nnUNetTrainer'
        )
    )
    trainer_name = _build_generated_trainer_name(exp_config, work_dir)
    trainer_path = trainer_root / '{}.py'.format(trainer_name)

    initial_lr = float(exp_config.get('initial_lr', 1e-2))
    weight_decay = float(exp_config.get('weight_decay', 3e-5))
    oversample = float(exp_config.get('oversample_foreground_percent', 0.33))
    epochs = int(
        exp_config.get('epochs')
        or task_spec.extra.get('nnunet_search_epochs')
        or 100
    )
    scheduler = str(exp_config.get('lr_scheduler') or 'poly').strip().lower()
    if scheduler not in {'poly', 'cosine'}:
        raise RuntimeError('Unsupported trainer scheduler {!r}'.format(scheduler))

    trainer_source = """import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.{parent_trainer} import {parent_trainer}


class {trainer_name}({parent_trainer}):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = {initial_lr!r}
        self.weight_decay = {weight_decay!r}
        self.oversample_foreground_percent = {oversample!r}
        self.num_epochs = {epochs!r}
        self.lr_scheduler_name = {scheduler!r}
        self.print_to_log_file(
            'DentalClaw adaptive trainer overrides:',
            {{
                'parent_trainer': {parent_trainer!r},
                'initial_lr': self.initial_lr,
                'weight_decay': self.weight_decay,
                'oversample_foreground_percent': self.oversample_foreground_percent,
                'num_epochs': self.num_epochs,
                'lr_scheduler': self.lr_scheduler_name,
            }},
            add_timestamp=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        if self.lr_scheduler_name == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        else:
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
""".format(
        parent_trainer=parent_trainer,
        trainer_name=trainer_name,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        oversample=oversample,
        epochs=epochs,
        scheduler=scheduler,
    )
    trainer_path.write_text(trainer_source, encoding='utf-8')

    trainer_metadata = {
        'trainer_name': trainer_name,
        'trainer_definition_path': str(trainer_path),
        'inherits_from_trainer': parent_trainer,
        'initial_lr': initial_lr,
        'weight_decay': weight_decay,
        'oversample_foreground_percent': oversample,
        'epochs': epochs,
        'lr_scheduler': scheduler,
    }
    _write_json(Path(work_dir) / 'generated_trainer.json', trainer_metadata)
    return trainer_metadata


def _parse_int_token(value: str) -> int:
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


def _query_gpu_inventory() -> List[Dict[str, Any]]:
    gpu_result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus: List[Dict[str, Any]] = []
    for line in gpu_result.stdout.splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) != 6:
            continue
        gpus.append({
            'index': int(parts[0]),
            'uuid': parts[1],
            'name': parts[2],
            'memory_used_mb': _parse_int_token(parts[3]),
            'memory_total_mb': _parse_int_token(parts[4]),
            'utilization_gpu_pct': _parse_int_token(parts[5]),
            'active_compute_processes': 0,
            'active_compute_memory_mb': 0,
        })

    app_result = subprocess.run(
        [
            'nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    apps_by_uuid: Dict[str, List[Dict[str, Any]]] = {}
    for line in (app_result.stdout or '').splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) != 4:
            continue
        apps_by_uuid.setdefault(parts[0], []).append({
            'pid': parts[1],
            'process_name': parts[2],
            'used_memory_mb': _parse_int_token(parts[3]),
        })

    for gpu in gpus:
        apps = apps_by_uuid.get(gpu['uuid'], [])
        gpu['active_compute_processes'] = len(apps)
        gpu['active_compute_memory_mb'] = sum(item['used_memory_mb'] for item in apps)
    return gpus


def _select_available_gpu() -> Optional[Dict[str, Any]]:
    try:
        inventory = _query_gpu_inventory()
    except Exception:
        return None
    if not inventory:
        return None
    return sorted(
        inventory,
        key=lambda item: (
            item['active_compute_processes'] > 0,
            item['active_compute_processes'],
            item['memory_used_mb'],
            item['utilization_gpu_pct'],
            item['index'],
        ),
    )[0]


def _parse_nnunet_summary(summary_path: Path) -> Dict[str, Any]:
    payload = _load_json(str(summary_path))
    foreground_mean = payload.get('foreground_mean', {})
    mean_section = payload.get('mean', {})
    per_class_dice = {}
    per_class_iou = {}
    for key, value in mean_section.items():
        if not isinstance(value, dict):
            continue
        if value.get('Dice') is not None:
            per_class_dice[str(key)] = float(value['Dice'])
        if value.get('IoU') is not None:
            per_class_iou[str(key)] = float(value['IoU'])
    return {
        'mean_dice': float(foreground_mean.get('Dice', 0.0) or 0.0),
        'mean_hd95': None,
        'pixel_accuracy': None,
        'mean_iou': float(foreground_mean.get('IoU', 0.0) or 0.0),
        'per_class_dice': per_class_dice,
        'per_class_hd95': {},
        'per_class_iou': per_class_iou,
        'metric_source': 'nnunet_validation_summary',
        'metric_summary_path': str(summary_path),
    }


def _config_identity(config: Dict[str, Any]) -> str:
    backend = str(config.get('backend') or '')
    if backend == 'nnUNetv2_cli':
        trainer_identity = (
            config.get('trainer_identity')
            or config.get('generated_trainer_name')
            or config.get('inherits_from_trainer')
            or config.get('nnunet_trainer')
        )
        keys = [
            'backend',
            'trainer_identity',
            'configuration',
            'fold',
            'epochs',
            'initial_lr',
            'weight_decay',
            'oversample_foreground_percent',
            'lr_scheduler',
        ]
        payload = {key: config.get(key) for key in keys if key != 'trainer_identity'}
        payload['trainer_identity'] = trainer_identity
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    else:
        keys = [
            'backend',
            'model_name',
            'base_channels',
            'depth',
            'img_size',
            'batch_size',
            'lr',
            'weight_decay',
            'epochs',
            'seed',
        ]
    return json.dumps({key: config.get(key) for key in keys}, sort_keys=True, ensure_ascii=False)


def _metric_context(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return 'no validation metrics were available yet'
    dice = metrics.get('mean_dice')
    iou = metrics.get('mean_iou')
    if dice is None and iou is None:
        return 'no validation metrics were available yet'
    parts = []
    if dice is not None:
        parts.append('Dice={:.4f}'.format(float(dice)))
    if iou is not None:
        parts.append('IoU={:.4f}'.format(float(iou)))
    return ', '.join(parts)


def _replace_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _wants_nnunet_cli(dataset_spec: DatasetSpec, task_spec: TaskSpec, detected_dimension: str, root: Optional[Path] = None) -> bool:
    target_backend = dataset_spec.extra.get('target_backend') or task_spec.extra.get('target_backend')
    if target_backend == 'nnunetv2_cli':
        return True
    root = root or Path(dataset_spec.root)
    if _is_nnunet_raw_dataset(root):
        return True
    if dataset_spec.extra.get('nnunet_dataset_id') is not None:
        return True
    return detected_dimension == '3d'


def _existing_label_ts(root: Path, dataset_spec: DatasetSpec) -> Optional[str]:
    labels_ts = dataset_spec.extra.get('labelsTs')
    if labels_ts:
        return str(labels_ts)
    if (root / 'labelsTs').is_dir():
        return 'labelsTs'
    return None


def _analyze_images(image_files: List[Path]) -> Dict[str, Any]:
    sample_shapes = []
    sample_modes = []
    panoramic_votes = 0
    for path in image_files[:4]:
        if _suffix_of(path) not in IMAGE_SUFFIXES_2D:
            continue
        image = Image.open(path)
        arr = np.array(image)
        sample_shapes.append(list(arr.shape))
        sample_modes.append(image.mode)
        height, width = arr.shape[:2]
        if width / max(height, 1) >= 1.6:
            panoramic_votes += 1
    return {
        'sample_shapes': sample_shapes,
        'sample_modes': sample_modes,
        'looks_like_panoramic_2d': panoramic_votes > 0,
    }


def _analyze_labels(label_files: List[Path]) -> Dict[str, Any]:
    unique_values = set()
    sample_shapes = []
    for path in label_files[:8]:
        if _suffix_of(path) not in IMAGE_SUFFIXES_2D:
            continue
        arr = np.array(Image.open(path))
        sample_shapes.append(list(arr.shape))
        unique_values.update(np.unique(arr).tolist())
    values = sorted(int(v) for v in unique_values)
    return {
        'sample_shapes': sample_shapes,
        'unique_values_preview': values[:64],
        'num_unique_values_preview': len(values),
        'max_label_value': max(values) if values else None,
    }


def _preprocess_defaults(dataset_spec: DatasetSpec, task_spec: TaskSpec) -> Dict[str, Any]:
    config = {
        'normalize_percentiles': [1.0, 99.0],
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_grid_size': 8,
        'median_blur_ksize': 3,
        'label_clip_max': task_spec.num_classes,
    }
    config.update(dataset_spec.extra.get('preprocess', {}))
    config.update(task_spec.extra.get('preprocess', {}))
    return config


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
    return array[..., 0]


def _normalize_percentiles(image: np.ndarray, low: float, high: float) -> np.ndarray:
    lo = np.percentile(image, low)
    hi = np.percentile(image, high)
    if hi <= lo:
        return image.astype(np.uint8)
    image = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _preprocess_image_2d(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    image = _to_grayscale(image)
    percentiles = config.get('normalize_percentiles', [1.0, 99.0])
    if percentiles:
        image = _normalize_percentiles(image, float(percentiles[0]), float(percentiles[1]))
    else:
        image = image.astype(np.uint8)

    if config.get('use_clahe', True):
        tile = int(config.get('clahe_tile_grid_size', 8))
        clahe = cv2.createCLAHE(
            clipLimit=float(config.get('clahe_clip_limit', 2.0)),
            tileGridSize=(tile, tile),
        )
        image = clahe.apply(image)

    ksize = int(config.get('median_blur_ksize', 3))
    if ksize > 1:
        if ksize % 2 == 0:
            ksize += 1
        image = cv2.medianBlur(image, ksize)
    return image.astype(np.uint8)


def _preprocess_label_2d(label: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    label = label.astype(np.int32)
    label[label < 0] = 0
    label = np.clip(label, 0, int(config.get('label_clip_max', label.max() if label.size else 0)))
    return label.astype(np.uint8)


def _resize_with_padding(array: np.ndarray, target_size: int, is_mask: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    if array.ndim == 2:
        height, width = array.shape
    else:
        height, width = array.shape[:2]
    scale = min(target_size / max(height, 1), target_size / max(width, 1))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = np.array(Image.fromarray(array).resize((new_width, new_height), resample=resample))
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    if array.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=resized.dtype)
        canvas[top:top + new_height, left:left + new_width] = resized
    else:
        channels = resized.shape[2]
        canvas = np.zeros((target_size, target_size, channels), dtype=resized.dtype)
        canvas[top:top + new_height, left:left + new_width, :] = resized
    return canvas, {
        'orig_height': height,
        'orig_width': width,
        'new_height': new_height,
        'new_width': new_width,
        'top': top,
        'left': left,
    }


def _restore_from_padding(mask: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    cropped = mask[
        meta['top']:meta['top'] + meta['new_height'],
        meta['left']:meta['left'] + meta['new_width'],
    ]
    restored = Image.fromarray(cropped.astype(np.uint8)).resize(
        (meta['orig_width'], meta['orig_height']),
        resample=Image.NEAREST,
    )
    return np.array(restored)


class ToothPanoramicDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Optional[Path], image_size: int, num_classes: int) -> None:
        self.image_files = _collect_files(image_dir)
        self.label_lookup = {path.name: path for path in _collect_files(label_dir)} if label_dir else {}
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_files[index]
        image = np.array(Image.open(image_path).convert('L'))
        image_resized, meta = _resize_with_padding(image, self.image_size, is_mask=False)
        sample = {
            'name': image_path.name,
            'image': torch.from_numpy(image_resized.astype(np.float32) / 255.0).unsqueeze(0),
            'resize_meta': meta,
        }
        label_path = self.label_lookup.get(image_path.name)
        if label_path and label_path.exists():
            label = np.array(Image.open(label_path))
            label = np.clip(label, 0, self.num_classes)
            label_resized, _ = _resize_with_padding(label, self.image_size, is_mask=True)
            sample['label'] = torch.from_numpy(label_resized.astype(np.int64))
        return sample


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TinyUNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 24, depth: int = 4) -> None:
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_in = in_channels
        for channels_out in channels:
            self.down_blocks.append(ConvBlock(current_in, channels_out))
            self.pools.append(nn.MaxPool2d(2))
            current_in = channels_out

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        decoder_in = channels[-1] * 2
        for channels_out in reversed(channels):
            self.up_transpose.append(nn.ConvTranspose2d(decoder_in, channels_out, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(channels_out * 2, channels_out))
            decoder_in = channels_out

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips = []
        x = inputs
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upsample, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, num_channels: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_channels).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    denominator = torch.sum(probs, dims) + torch.sum(one_hot, dims)
    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    foreground = dice[1:] if num_channels > 1 else dice
    return 1 - foreground.mean()


def _binary_hd95(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if not pred_mask.any() and not target_mask.any():
        return 0.0
    diagonal = float(np.hypot(*pred_mask.shape))
    if not pred_mask.any() or not target_mask.any():
        return diagonal

    pred_surface = np.logical_xor(pred_mask, binary_erosion(pred_mask, border_value=0))
    target_surface = np.logical_xor(target_mask, binary_erosion(target_mask, border_value=0))
    if not pred_surface.any():
        pred_surface = pred_mask
    if not target_surface.any():
        target_surface = target_mask

    dist_to_target = distance_transform_edt(~target_surface)
    dist_to_pred = distance_transform_edt(~pred_surface)
    surface_distances = np.concatenate([
        dist_to_target[pred_surface],
        dist_to_pred[target_surface],
    ])
    if surface_distances.size == 0:
        return 0.0
    return float(np.percentile(surface_distances, 95.0))


def _compute_metrics(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, Any]:
    eps = 1e-5
    per_class_dice: Dict[str, float] = {}
    per_class_hd95: Dict[str, float] = {}
    per_class_iou: Dict[str, float] = {}
    dice_values: List[float] = []
    hd95_values: List[float] = []
    iou_values: List[float] = []
    for class_index in range(1, num_classes + 1):
        pred_mask = prediction == class_index
        target_mask = target == class_index
        denom = pred_mask.sum() + target_mask.sum()
        if denom == 0:
            continue
        intersection = float((pred_mask & target_mask).sum())
        union = float((pred_mask | target_mask).sum())
        dice = (2.0 * intersection + eps) / (float(denom) + eps)
        iou = (intersection + eps) / (union + eps)
        hd95 = _binary_hd95(pred_mask, target_mask)
        per_class_dice[str(class_index)] = dice
        per_class_hd95[str(class_index)] = hd95
        per_class_iou[str(class_index)] = iou
        dice_values.append(dice)
        hd95_values.append(hd95)
        iou_values.append(iou)
    return {
        'mean_dice': _safe_mean(dice_values, default=0.0),
        'mean_hd95': _safe_mean(hd95_values, default=0.0),
        'mean_iou': _safe_mean(iou_values, default=0.0),
        'pixel_accuracy': float((prediction == target).mean()),
        'per_class_dice': per_class_dice,
        'per_class_hd95': per_class_hd95,
        'per_class_iou': per_class_iou,
    }


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8)).save(path)


def _save_overlay(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    rgb = np.stack([image, image, image], axis=-1).astype(np.uint8)
    overlay = rgb.copy()
    for class_index in np.unique(mask):
        if class_index == 0:
            continue
        color = np.array([
            (37 * int(class_index)) % 255,
            (97 * int(class_index)) % 255,
            (167 * int(class_index)) % 255,
        ], dtype=np.uint8)
        class_mask = mask == class_index
        overlay[class_mask] = (0.55 * overlay[class_mask] + 0.45 * color).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def _write_inference_summary_bundle(output_dir: Path, summary: Dict[str, Any]) -> None:
    _write_json(output_dir / 'inference_summary.json', summary)
    lines = [
        '# Inference Summary',
        '',
        '- Output directory: {}'.format(summary.get('output_dir')),
        '- Prediction directory: {}'.format(summary.get('prediction_dir', summary.get('output_dir'))),
        '- Overlay directory: {}'.format(summary.get('overlay_dir') or 'n/a'),
        '- Ground-truth directory: {}'.format(summary.get('gt_dir') or 'n/a'),
        '- Evaluated cases: {}'.format(summary.get('num_evaluated_cases', 0)),
        '- Mean Dice: {}'.format(_format_metric(summary.get('mean_dice'))),
        '- Mean HD95: {}'.format(_format_metric(summary.get('mean_hd95'))),
        '- Mean IoU: {}'.format(_format_metric(summary.get('mean_iou'))),
        '- Pixel accuracy: {}'.format(_format_metric(summary.get('pixel_accuracy'))),
        '',
        '## Artifacts',
        '',
        '- Summary JSON: {}'.format(output_dir / 'inference_summary.json'),
        '- Predict command JSON: {}'.format(output_dir / 'nnunet_predict_command.json'),
    ]
    if summary.get('warnings'):
        lines.extend(['', '## Warnings', ''])
        for item in summary['warnings']:
            lines.append('- {}'.format(item))
    (output_dir / 'inference_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def _evaluate_nnunet_prediction_dir(
    prediction_dir: Path,
    input_dir: Path,
    gt_dir: Optional[Path],
    num_classes: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'prediction_dir': str(prediction_dir),
        'input_dir': str(input_dir),
        'gt_dir': str(gt_dir) if gt_dir else None,
        'metrics_per_case': {},
        'num_predictions': 0,
        'num_evaluated_cases': 0,
        'overlay_dir': None,
        'warnings': [],
    }
    prediction_files = [path for path in _collect_files(prediction_dir) if _suffix_of(path) in IMAGE_SUFFIXES_2D]
    summary['num_predictions'] = len(prediction_files)
    if not gt_dir or not gt_dir.is_dir():
        summary['warnings'].append('No labelsTs directory was available, so no test metrics were computed.')
        return summary

    gt_lookup = {_case_key_from_path(path): path for path in _collect_files(gt_dir) if _suffix_of(path) in IMAGE_SUFFIXES_2D}
    input_lookup = {_case_key_from_path(path): path for path in _collect_files(input_dir) if _suffix_of(path) in IMAGE_SUFFIXES_2D}
    overlay_dir = prediction_dir / 'overlays'
    metrics_per_case: Dict[str, Any] = {}
    unmatched_predictions: List[str] = []

    for prediction_path in prediction_files:
        case_key = _case_key_from_path(prediction_path)
        gt_path = gt_lookup.get(case_key)
        if gt_path is None:
            unmatched_predictions.append(prediction_path.name)
            continue
        prediction = np.array(Image.open(prediction_path))
        target = np.array(Image.open(gt_path))
        if prediction.shape != target.shape:
            summary['warnings'].append(
                'Shape mismatch for {}: prediction {} vs target {}'.format(
                    prediction_path.name,
                    list(prediction.shape),
                    list(target.shape),
                )
            )
            continue
        case_metrics = _compute_metrics(prediction.astype(np.int32), target.astype(np.int32), num_classes)
        metrics_per_case[case_key] = case_metrics
        input_path = input_lookup.get(case_key)
        if input_path is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            image = _to_grayscale(np.array(Image.open(input_path))).astype(np.uint8)
            _save_overlay(overlay_dir / '{}.png'.format(case_key), image, prediction.astype(np.uint8))

    summary['metrics_per_case'] = metrics_per_case
    summary['num_evaluated_cases'] = len(metrics_per_case)
    if overlay_dir.exists():
        summary['overlay_dir'] = str(overlay_dir)
    if unmatched_predictions:
        summary['warnings'].append(
            'Predictions without matching ground truth: {}'.format(', '.join(unmatched_predictions[:20]))
        )
    if metrics_per_case:
        summary['mean_dice'] = _safe_mean([item['mean_dice'] for item in metrics_per_case.values()])
        summary['mean_hd95'] = _safe_mean([item['mean_hd95'] for item in metrics_per_case.values()])
        summary['mean_iou'] = _safe_mean([item['mean_iou'] for item in metrics_per_case.values()])
        summary['pixel_accuracy'] = _safe_mean([item['pixel_accuracy'] for item in metrics_per_case.values()])
    else:
        summary['warnings'].append('No prediction-ground truth pairs were evaluated.')
    return summary


class ToothSegmentationSkill(BaseSkill):
    def __init__(self) -> None:
        self.last_dataset_info: Optional[Dict[str, Any]] = None
        self.preprocess_info: Optional[Dict[str, Any]] = None
        self.memory_history: List[Dict[str, Any]] = []
        self.last_task_spec: Optional[TaskSpec] = None

    def set_experiment_memory(self, history: List[Dict[str, Any]]) -> None:
        self.memory_history = list(history or [])

    def set_preprocess_info(self, info: Optional[Dict[str, Any]]) -> None:
        self.preprocess_info = info

    def preprocess_dataset(self, dataset_spec: DatasetSpec, task_spec: TaskSpec, workspace: str) -> Tuple[DatasetSpec, Dict[str, Any]]:
        root = Path(dataset_spec.root)
        all_images = _collect_files(root / dataset_spec.imagesTr) + _collect_files(root / dataset_spec.imagesVal) + _collect_files(root / dataset_spec.imagesTs)
        detected_dimension = _detect_dimension(all_images)
        if _wants_nnunet_cli(dataset_spec, task_spec, detected_dimension, root=root):
            nnunet_roots = _resolve_nnunet_roots(root, dataset_spec)
            info = {
                'source_root': str(root),
                'output_root': str(root),
                'detected_dimension': detected_dimension,
                'mode': 'skipped_for_nnunet_cli',
                'reason': 'Dataset already follows nnUNet layout or is configured for nnUNetv2 CLI.',
                'nnunet_env': {key: str(value) for key, value in nnunet_roots.items()},
            }
            self.preprocess_info = info
            merged_extra = dict(dataset_spec.extra)
            merged_extra.update({
                'target_backend': 'nnunetv2_cli',
                'nnunet_dataset_id': _infer_nnunet_dataset_id(root, dataset_spec),
                'nnunet_raw': str(nnunet_roots['nnUNet_raw']),
                'nnunet_preprocessed': str(nnunet_roots['nnUNet_preprocessed']),
                'nnunet_results': str(nnunet_roots['nnUNet_results']),
            })
            if _existing_label_ts(root, dataset_spec):
                merged_extra['labelsTs'] = _existing_label_ts(root, dataset_spec)
            return DatasetSpec(
                root=str(root),
                imagesTr=dataset_spec.imagesTr,
                labelsTr=dataset_spec.labelsTr,
                imagesVal=dataset_spec.imagesVal,
                labelsVal=dataset_spec.labelsVal,
                imagesTs=dataset_spec.imagesTs,
                extra=merged_extra,
            ), info
        config = _preprocess_defaults(dataset_spec, task_spec)
        output_root = Path(workspace) / 'preprocessed_dataset'
        output_root.mkdir(parents=True, exist_ok=True)

        def process_split(split_name: str, label_split: bool = False) -> Dict[str, Any]:
            src_name = getattr(dataset_spec, split_name)
            src_dir = root / src_name
            dst_dir = output_root / src_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            processed = 0
            copied = 0
            for src_path in _collect_files(src_dir):
                dst_path = dst_dir / src_path.name
                suffix = _suffix_of(src_path)
                if detected_dimension == '2d' and suffix in IMAGE_SUFFIXES_2D:
                    array = np.array(Image.open(src_path))
                    if label_split:
                        array = _preprocess_label_2d(array, config)
                    else:
                        array = _preprocess_image_2d(array, config)
                    Image.fromarray(array).save(dst_path)
                    processed += 1
                else:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
            return {
                'source_dir': str(src_dir),
                'output_dir': str(dst_dir),
                'processed_files': processed,
                'copied_files': copied,
            }

        splits = {
            'imagesTr': process_split('imagesTr', label_split=False),
            'labelsTr': process_split('labelsTr', label_split=True),
            'imagesVal': process_split('imagesVal', label_split=False),
            'labelsVal': process_split('labelsVal', label_split=True),
            'imagesTs': process_split('imagesTs', label_split=False),
        }
        info = {
            'source_root': str(root),
            'output_root': str(output_root),
            'detected_dimension': detected_dimension,
            'config': config,
            'splits': splits,
        }
        _write_json(output_root / 'preprocess_report.json', info)
        processed_spec = DatasetSpec(
            root=str(output_root),
            imagesTr=dataset_spec.imagesTr,
            labelsTr=dataset_spec.labelsTr,
            imagesVal=dataset_spec.imagesVal,
            labelsVal=dataset_spec.labelsVal,
            imagesTs=dataset_spec.imagesTs,
            extra=dict(dataset_spec.extra, source_root=str(root), preprocess=config),
        )
        self.preprocess_info = info
        return processed_spec, info

    def analyze_dataset(self, dataset_spec: DatasetSpec, task_spec: TaskSpec) -> Dict[str, Any]:
        self.last_task_spec = task_spec
        root = Path(dataset_spec.root)
        images_tr = _collect_files(root / dataset_spec.imagesTr)
        images_val = _collect_files(root / dataset_spec.imagesVal)
        images_ts = _collect_files(root / dataset_spec.imagesTs)
        labels_tr = _collect_files(root / dataset_spec.labelsTr)
        labels_val = _collect_files(root / dataset_spec.labelsVal)
        detected_dimension = _detect_dimension(images_tr + images_val + images_ts)
        label_info = _analyze_labels(labels_tr or labels_val)
        warnings = []
        counts = {
            'imagesTr': len(images_tr),
            'labelsTr': len(labels_tr),
            'imagesVal': len(images_val),
            'labelsVal': len(labels_val),
            'imagesTs': len(images_ts),
        }
        if _is_nnunet_raw_dataset(root) and counts['imagesVal'] == 0 and counts['labelsVal'] == 0:
            warnings.append('检测到 nnUNet 原始数据集布局，训练将直接走 nnUNetv2 CLI，不再要求 imagesVal/labelsVal。')
        max_label_value = label_info.get('max_label_value')
        if detected_dimension == '2d' and task_spec.num_classes == 32 and max_label_value is not None and max_label_value < 32:
            warnings.append('标签预览最大值小于 32，当前数据更像二值或部分分类标签，流程可运行但不代表真实 32 类效果。')
        recommended_backend = 'nnUNetv2_cli' if _wants_nnunet_cli(dataset_spec, task_spec, detected_dimension, root=root) else (
            'builtin_nnunet_style_2d' if detected_dimension == '2d' else 'nnUNetv2_cli'
        )
        info = {
            'root': str(root),
            'detected_dimension': detected_dimension,
            'counts': counts,
            'image_analysis': _analyze_images(images_tr or images_val or images_ts),
            'label_analysis': label_info,
            'nnunet_cli_available': bool(_resolve_nnunet_cli('train')),
            'recommended_backend': recommended_backend,
            'nnunet_dataset_id': _infer_nnunet_dataset_id(root, dataset_spec),
            'nnunet_layout_detected': _is_nnunet_raw_dataset(root),
            'nnunet_env': {key: str(value) for key, value in _resolve_nnunet_roots(root, dataset_spec).items()},
            'training_requirements': task_spec.extra.get('training_requirements', {}),
            'warnings': warnings,
        }
        self.last_dataset_info = info
        return info

    def bootstrap_existing_experiments(
        self,
        dataset_spec: DatasetSpec,
        task_spec: TaskSpec,
        workspace: str,
    ) -> List[Dict[str, Any]]:
        if not task_spec.extra.get('reuse_existing_baseline_if_available', False):
            return []
        dataset_info = self.last_dataset_info or self.analyze_dataset(dataset_spec, task_spec)
        if dataset_info.get('recommended_backend') != 'nnUNetv2_cli':
            return []

        root = Path(dataset_spec.root)
        nnunet_roots = _resolve_nnunet_roots(root, dataset_spec)
        results_root = Path(nnunet_roots['nnUNet_results'])
        configuration = str(task_spec.extra.get('nnunet_configuration', _default_nnunet_configuration(dataset_info['detected_dimension'])))
        fold = str(task_spec.extra.get('nnunet_search_fold', 'all'))
        trainer = 'nnUNetTrainer'
        fold_dir = results_root / root.name / '{}__nnUNetPlans__{}'.format(trainer, configuration) / 'fold_{}'.format(fold)
        validation_summary_path = fold_dir / 'validation' / 'summary.json'
        checkpoint_best_path = fold_dir / 'checkpoint_best.pth'
        checkpoint_final_path = fold_dir / 'checkpoint_final.pth'
        training_curve_path = fold_dir / 'progress.png'
        if not validation_summary_path.exists():
            return []
        best_model_path = checkpoint_best_path if checkpoint_best_path.exists() else checkpoint_final_path
        if not best_model_path.exists():
            return []

        metrics = _parse_nnunet_summary(validation_summary_path)
        notes = 'Reused pre-existing baseline fold_all result from {}'.format(fold_dir)
        if training_curve_path.exists():
            notes += '; progress_png={}'.format(training_curve_path)
        record = ExperimentRecord(
            exp_id='bootstrap_existing_baseline',
            task_id=task_spec.task_id,
            model_name='nnUNet-{}'.format(configuration),
            config={
                'backend': 'nnUNetv2_cli',
                'trial_name': 'reused_baseline_fold_all',
                'nnunet_trainer': trainer,
                'configuration': configuration,
                'fold': fold,
                'selection_reason': 'Reuse the existing completed default nnUNetTrainer fold=all baseline as trial 1 before launching new hyperparameter trials.',
            },
            best_model_path=str(best_model_path),
            metrics=metrics,
            work_dir=str(fold_dir),
            notes=notes,
        )
        payload = _serializable(record)
        payload['backend'] = 'nnUNetv2_cli'
        payload['artifacts'] = {
            'result_dir': str(fold_dir),
            'trainer_dir': str(fold_dir.parent),
            'nnunet_results_root': str(results_root),
            'training_curve_path': str(training_curve_path) if training_curve_path.exists() else None,
            'validation_summary_path': str(validation_summary_path),
            'train_command_path': None,
            'preprocess_command_path': None,
            'selected_gpu': None,
        }
        payload['timing'] = {
            'started_at': None,
            'completed_at': None,
            'duration_seconds': None,
        }
        payload['gpu_snapshot'] = None
        return [payload]

    def _nnunet_search_is_adaptive(self) -> bool:
        if self.last_task_spec is None:
            return False
        search_mode = str(self.last_task_spec.extra.get('nnunet_search_mode') or '').strip().lower()
        strategy = str(self.last_task_spec.extra.get('nnunet_search_strategy', 'adaptive')).strip().lower()
        return search_mode == 'trainer_hparam_search' and strategy != 'batch'

    def _nnunet_cli_followup_candidates(
        self,
        best_record: Dict[str, Any],
        budget_spec: BudgetSpec,
    ) -> List[Dict[str, Any]]:
        best_config = dict(best_record.get('config') or {})
        best_metrics = best_record.get('metrics') or {}
        dataset_info = self.last_dataset_info or {}
        task_spec = self.last_task_spec or default_teeth32_task_spec()
        configuration = str(
            best_config.get('configuration')
            or task_spec.extra.get('nnunet_configuration', _default_nnunet_configuration(dataset_info.get('detected_dimension', '2d')))
        )
        parent_trainer = str(
            best_config.get('generated_trainer_name')
            or best_config.get('nnunet_trainer')
            or task_spec.extra.get('nnunet_followup_parent_trainer')
            or 'nnUNetTrainer'
        )
        fold_value = str(best_config.get('fold') or task_spec.extra.get('nnunet_search_fold', 'all'))
        base_epochs = int(best_config.get('epochs') or task_spec.extra.get('nnunet_search_epochs') or budget_spec.max_epochs_per_trial or 100)
        base_lr = float(best_config.get('initial_lr') or 1e-2)
        base_wd = float(best_config.get('weight_decay') or 3e-5)
        base_oversample = float(best_config.get('oversample_foreground_percent') or 0.33)
        base_scheduler = str(best_config.get('lr_scheduler') or 'poly').strip().lower()
        best_trial_name = str(best_config.get('trial_name') or best_record.get('exp_id') or 'best_trial')
        metric_context = _metric_context(best_metrics)

        variants = [
            {
                'backend': 'nnUNetv2_cli',
                'configuration': configuration,
                'fold': fold_value,
                'trial_name': '{}_lower_lr'.format(best_trial_name),
                'initial_lr': max(1e-4, round(base_lr * 0.75, 6)),
                'weight_decay': base_wd,
                'oversample_foreground_percent': base_oversample,
                'lr_scheduler': base_scheduler,
                'epochs': base_epochs,
                'materialize_trainer_subclass': True,
                'inherits_from_trainer': parent_trainer,
                'trainer_identity': 'adaptive_search',
                'selection_reason': (
                    'Follow-up after {} ({}). Materialize a new trainer subclass that inherits from {} and lowers the initial learning rate by 25% to test whether smoother optimization improves validation Dice.'
                ).format(best_trial_name, metric_context, parent_trainer),
            },
            {
                'backend': 'nnUNetv2_cli',
                'configuration': configuration,
                'fold': fold_value,
                'trial_name': '{}_higher_wd'.format(best_trial_name),
                'initial_lr': base_lr,
                'weight_decay': min(1e-3, round(max(base_wd * 3.0, 5e-5), 6)),
                'oversample_foreground_percent': base_oversample,
                'lr_scheduler': base_scheduler,
                'epochs': base_epochs,
                'materialize_trainer_subclass': True,
                'inherits_from_trainer': parent_trainer,
                'trainer_identity': 'adaptive_search',
                'selection_reason': (
                    'Follow-up after {} ({}). Materialize a new trainer subclass that inherits from {} and increases weight decay to probe whether stronger regularization improves holdout generalization.'
                ).format(best_trial_name, metric_context, parent_trainer),
            },
            {
                'backend': 'nnUNetv2_cli',
                'configuration': configuration,
                'fold': fold_value,
                'trial_name': '{}_swap_scheduler'.format(best_trial_name),
                'initial_lr': base_lr,
                'weight_decay': base_wd,
                'oversample_foreground_percent': base_oversample,
                'lr_scheduler': 'cosine' if base_scheduler == 'poly' else 'poly',
                'epochs': base_epochs,
                'materialize_trainer_subclass': True,
                'inherits_from_trainer': parent_trainer,
                'trainer_identity': 'adaptive_search',
                'selection_reason': (
                    'Follow-up after {} ({}). Materialize a new trainer subclass that inherits from {} and changes only the LR schedule to test whether the current best recipe is scheduler-limited.'
                ).format(best_trial_name, metric_context, parent_trainer),
            },
            {
                'backend': 'nnUNetv2_cli',
                'configuration': configuration,
                'fold': fold_value,
                'trial_name': '{}_fg_focus'.format(best_trial_name),
                'initial_lr': max(1e-4, round(base_lr * 0.85, 6)),
                'weight_decay': min(1e-3, round(max(base_wd * 1.5, 5e-5), 6)),
                'oversample_foreground_percent': min(0.6, round(max(base_oversample, 0.5), 2)),
                'lr_scheduler': 'cosine' if base_scheduler == 'poly' else base_scheduler,
                'epochs': base_epochs,
                'materialize_trainer_subclass': True,
                'inherits_from_trainer': parent_trainer,
                'trainer_identity': 'adaptive_search',
                'selection_reason': (
                    'Follow-up after {} ({}). Materialize a new trainer subclass that inherits from {} and increases foreground oversampling to test whether tooth-pixel recall is the remaining bottleneck.'
                ).format(best_trial_name, metric_context, parent_trainer),
            },
        ]

        dice = float(best_metrics.get('mean_dice', 0.0) or 0.0)
        if dice < 0.90:
            order = [3, 0, 2, 1]
        elif dice < 0.94:
            order = [0, 2, 1, 3]
        else:
            order = [2, 1, 0, 3]
        return [variants[index] for index in order]

    def _score_record(self, record: Dict[str, Any]) -> Tuple[float, float, float, float]:
        metrics = record.get('metrics', {})
        return (
            float(metrics.get('mean_dice', 0.0) or 0.0),
            float(metrics.get('mean_iou', 0.0) or 0.0),
            -float(metrics.get('mean_hd95', 1e9) if metrics.get('mean_hd95') is not None else 1e9),
            float(metrics.get('pixel_accuracy', 0.0) or 0.0),
        )

    def _candidate_pool(self, task_spec: TaskSpec, budget_spec: BudgetSpec) -> List[Dict[str, Any]]:
        requirements = task_spec.extra.get('training_requirements', {})
        if requirements.get('initial_configs'):
            return [dict(cfg) for cfg in requirements['initial_configs']]

        learning_rates = requirements.get('learning_rates', [1e-3, 5e-4, 2e-4])
        img_sizes = requirements.get('img_sizes', [512, 640, 768])
        base_channels = requirements.get('base_channels', [16, 24, 32])
        batch_sizes = requirements.get('batch_sizes', [2, 1, 1])
        depths = requirements.get('depths', [4])
        weight_decays = requirements.get('weight_decays', [1e-4, 5e-5])
        seeds = requirements.get('seeds', [2026, 2027, 2028, 2029])

        pool = []
        index = 0
        for lr, img_size, channels, depth, weight_decay in itertools.product(
            learning_rates,
            img_sizes,
            base_channels,
            depths,
            weight_decays,
        ):
            batch_size = batch_sizes[min(index % len(batch_sizes), len(batch_sizes) - 1)]
            pool.append({
                'backend': 'builtin_nnunet_style_2d',
                'model_name': 'nnunet2d_{}c_{}px'.format(channels, img_size),
                'base_channels': channels,
                'depth': depth,
                'img_size': img_size,
                'batch_size': batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
                'epochs': budget_spec.max_epochs_per_trial,
                'seed': seeds[index % len(seeds)],
            })
            index += 1
            if len(pool) >= 18:
                break
        return pool

    def _nnunet_cli_candidate_pool(
        self,
        dataset_info: Dict[str, Any],
        task_spec: TaskSpec,
        budget_spec: BudgetSpec,
    ) -> List[Dict[str, Any]]:
        default_configuration = str(
            task_spec.extra.get('nnunet_configuration', _default_nnunet_configuration(dataset_info['detected_dimension']))
        )
        search_mode = str(task_spec.extra.get('nnunet_search_mode') or '').strip().lower()
        if search_mode == 'trainer_hparam_search':
            fold_value = str(task_spec.extra.get('nnunet_search_fold', 'all'))
            requested_epochs = task_spec.extra.get('nnunet_search_epochs')
            baseline_trainer = str(task_spec.extra.get('nnunet_baseline_trainer', 'nnUNetTrainer'))
            custom_candidates = task_spec.extra.get('nnunet_search_candidates') or [
                {
                    'trial_name': 'baseline_default_trainer',
                    'nnunet_trainer': baseline_trainer,
                    'selection_reason': 'Start with the default nnUNetTrainer on the QC-filtered dataset to establish a clean baseline before introducing trainer-level changes.',
                },
                {
                    'trial_name': 'lower_lr_poly',
                    'initial_lr': 7.5e-3,
                    'weight_decay': 3e-5,
                    'oversample_foreground_percent': 0.33,
                    'lr_scheduler': 'poly',
                    'materialize_trainer_subclass': True,
                    'selection_reason': 'Create an inherited trainer subclass after the baseline and reduce the initial learning rate to stabilize optimization on panoramic binary masks.',
                },
                {
                    'trial_name': 'higher_wd_poly',
                    'initial_lr': 1e-2,
                    'weight_decay': 1e-4,
                    'oversample_foreground_percent': 0.33,
                    'lr_scheduler': 'poly',
                    'materialize_trainer_subclass': True,
                    'selection_reason': 'Create an inherited trainer subclass after the baseline and increase weight decay to test whether stronger regularization improves generalization on the TDD holdout split.',
                },
                {
                    'trial_name': 'baseline_cosine',
                    'initial_lr': 1e-2,
                    'weight_decay': 3e-5,
                    'oversample_foreground_percent': 0.33,
                    'lr_scheduler': 'cosine',
                    'materialize_trainer_subclass': True,
                    'selection_reason': 'Create an inherited trainer subclass after the baseline and swap the scheduler to cosine annealing while holding the other trainer hyperparameters near baseline.',
                },
                {
                    'trial_name': 'fg_focus_cosine',
                    'initial_lr': 7.5e-3,
                    'weight_decay': 5e-5,
                    'oversample_foreground_percent': 0.5,
                    'lr_scheduler': 'cosine',
                    'materialize_trainer_subclass': True,
                    'selection_reason': 'Create an inherited trainer subclass after the baseline and bias sampling more toward foreground patches to test recall on tooth regions.',
                },
            ]
            candidates: List[Dict[str, Any]] = []
            for item in custom_candidates:
                candidate = {
                    'backend': 'nnUNetv2_cli',
                    'nnunet_trainer': item.get('nnunet_trainer'),
                    'configuration': str(item.get('configuration', default_configuration)),
                    'fold': str(item.get('fold', fold_value)),
                    'trial_name': item.get('trial_name'),
                    'initial_lr': item.get('initial_lr'),
                    'weight_decay': item.get('weight_decay'),
                    'oversample_foreground_percent': item.get('oversample_foreground_percent'),
                    'lr_scheduler': item.get('lr_scheduler'),
                    'selection_reason': item.get('selection_reason', 'DentalClaw custom trainer search trial.'),
                    'materialize_trainer_subclass': bool(item.get('materialize_trainer_subclass', False)),
                    'inherits_from_trainer': item.get('inherits_from_trainer') or baseline_trainer,
                    'trainer_identity': item.get('trainer_identity') or (
                        'adaptive_search' if item.get('materialize_trainer_subclass') else str(item.get('nnunet_trainer') or baseline_trainer)
                    ),
                }
                if item.get('epochs') is not None:
                    candidate['epochs'] = int(item['epochs'])
                elif requested_epochs is not None:
                    candidate['epochs'] = int(requested_epochs)
                candidates.append(candidate)
            return candidates[:max(1, min(len(candidates), budget_spec.max_trials))]
        configurations = [
            str(item)
            for item in (task_spec.extra.get('nnunet_search_configurations') or [default_configuration])
        ]
        folds = [int(item) for item in (task_spec.extra.get('nnunet_search_folds') or [0, 1, 2, 3, 4])]
        candidates: List[Dict[str, Any]] = []
        for configuration in configurations:
            for fold in folds:
                if fold == 0 and configuration == default_configuration:
                    reason = (
                        'Baseline default nnUNetTrainer run on fold 0 using the canonical {} configuration.'
                    ).format(configuration)
                elif configuration == default_configuration:
                    reason = (
                        'Keep the default nnUNetTrainer and configuration fixed, then test fold {} to measure '
                        'fold sensitivity before changing the pipeline.'
                    ).format(fold)
                else:
                    reason = (
                        'Keep the default nnUNetTrainer but try configuration {} on fold {} as a secondary search axis.'
                    ).format(configuration, fold)
                candidates.append({
                    'backend': 'nnUNetv2_cli',
                    'nnunet_trainer': 'nnUNetTrainer',
                    'configuration': configuration,
                    'fold': fold,
                    'selection_reason': reason,
                })
        return candidates[:max(1, min(len(candidates), budget_spec.max_trials))]

    def generate_initial_experiments(self, dataset_spec: DatasetSpec, task_spec: TaskSpec, budget_spec: BudgetSpec) -> List[Dict[str, Any]]:
        dataset_info = self.last_dataset_info or self.analyze_dataset(dataset_spec, task_spec)
        if dataset_info['recommended_backend'] == 'nnUNetv2_cli':
            candidates = self._nnunet_cli_candidate_pool(dataset_info, task_spec, budget_spec)
            if self._nnunet_search_is_adaptive():
                return candidates[:1]
            return candidates

        pool = self._candidate_pool(task_spec, budget_spec)
        if self.memory_history:
            usable = [record for record in self.memory_history if record.get('status') == 'completed']
            if usable:
                best_memory = max(usable, key=self._score_record)
                focus = dict(best_memory['config'])
                focus['epochs'] = budget_spec.max_epochs_per_trial
                focus['seed'] = int(focus.get('seed', 2026)) + 100
                focus['lr'] = max(1e-5, float(focus.get('lr', 1e-3)) * 0.8)
                return [focus] + [cfg for cfg in pool if cfg != focus][:max(0, budget_spec.max_trials - 1)]
        return pool[:max(1, min(len(pool), budget_spec.max_trials))]

    def run_training(self, dataset_spec: DatasetSpec, task_spec: TaskSpec, exp_config: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        dataset_info = self.last_dataset_info or self.analyze_dataset(dataset_spec, task_spec)
        if dataset_info['recommended_backend'] == 'nnUNetv2_cli':
            return self._run_nnunet_cli_training(dataset_spec, task_spec, exp_config, work_dir)
        return self._run_2d_training(dataset_spec, task_spec, exp_config, work_dir)

    def run_inference(self, infer_input: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        dataset_spec = infer_input['dataset_spec']
        task_spec = infer_input.get('task_spec') or default_teeth32_task_spec()
        dataset_info = self.last_dataset_info or self.analyze_dataset(dataset_spec, task_spec)
        if dataset_info['recommended_backend'] == 'nnUNetv2_cli':
            return self._run_nnunet_cli_inference(infer_input, output_dir)
        return self._run_2d_inference(infer_input, output_dir)

    def suggest_next_experiments(self, history: List[Dict[str, Any]], budget_spec: BudgetSpec) -> List[Dict[str, Any]]:
        if (self.last_dataset_info or {}).get('recommended_backend') == 'nnUNetv2_cli':
            if not self._nnunet_search_is_adaptive():
                return []
            task_spec = self.last_task_spec or default_teeth32_task_spec()
            dataset_info = self.last_dataset_info or {}
            seen = {
                _config_identity(record.get('config', {}))
                for record in list(self.memory_history) + list(history)
            }
            completed = [record for record in history if record.get('status') == 'completed']
            if completed:
                best = max(completed, key=self._score_record)
                for candidate in self._nnunet_cli_followup_candidates(best, budget_spec):
                    if _config_identity(candidate) not in seen:
                        return [candidate]
            candidate_pool = self._nnunet_cli_candidate_pool(dataset_info, task_spec, budget_spec)
            for candidate in candidate_pool:
                if _config_identity(candidate) in seen:
                    continue
                if history:
                    last_record = history[-1]
                    last_status = last_record.get('status', 'unknown')
                    last_name = last_record.get('config', {}).get('trial_name') or last_record.get('exp_id', 'previous_trial')
                    candidate = dict(candidate)
                    candidate['selection_reason'] = (
                        'Continue the recorded search after {} (status={}). {}'
                    ).format(last_name, last_status, candidate.get('selection_reason', 'DentalClaw search follow-up.'))
                return [candidate]
            return []
        combined = list(self.memory_history) + list(history)
        completed = [record for record in combined if record.get('status') == 'completed']
        if not completed:
            return []
        best = max(completed, key=self._score_record)
        best_config = dict(best['config'])
        requirements = (self.last_dataset_info or {}).get('training_requirements', {})
        img_sizes = requirements.get('img_sizes', [512, 640, 768])
        channel_choices = requirements.get('base_channels', [16, 24, 32])
        weight_decays = requirements.get('weight_decays', [1e-4, 5e-5])
        candidates = []
        for lr_factor, img_size, channels, weight_decay, seed_offset in [
            (0.5, best_config.get('img_size', 512), best_config.get('base_channels', 24), best_config.get('weight_decay', 1e-4), 301),
            (1.25, min(max(img_sizes), best_config.get('img_size', 512) + 128), best_config.get('base_channels', 24), best_config.get('weight_decay', 1e-4), 302),
            (1.0, best_config.get('img_size', 512), channel_choices[min(len(channel_choices) - 1, 1)], min(weight_decays), 303),
            (0.8, min(max(img_sizes), best_config.get('img_size', 512)), channel_choices[-1], max(weight_decays), 304),
        ]:
            candidate = dict(best_config)
            candidate['lr'] = max(1e-5, float(best_config.get('lr', 1e-3)) * lr_factor)
            candidate['img_size'] = int(img_size)
            candidate['base_channels'] = int(channels)
            candidate['weight_decay'] = float(weight_decay)
            candidate['epochs'] = budget_spec.max_epochs_per_trial
            candidate['seed'] = int(best_config.get('seed', 2026)) + seed_offset
            candidates.append(candidate)

        seen = {_config_identity(record.get('config', {})) for record in combined}
        return [cfg for cfg in candidates if _config_identity(cfg) not in seen]

    def select_best_model(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        completed = [record for record in history if record.get('status') == 'completed']
        if not completed:
            raise RuntimeError('No completed experiments were found.')
        return max(completed, key=self._score_record)

    def generate_report(self, history: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        best = self.select_best_model(history)
        with (output_path / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['exp_id', 'model_name', 'mean_dice', 'mean_hd95', 'mean_iou', 'pixel_accuracy', 'duration_seconds', 'selection_reason', 'best_model_path', 'status'])
            writer.writeheader()
            for record in history:
                metrics = record.get('metrics', {})
                writer.writerow({
                    'exp_id': record.get('exp_id'),
                    'model_name': record.get('model_name'),
                    'mean_dice': metrics.get('mean_dice'),
                    'mean_hd95': metrics.get('mean_hd95'),
                    'mean_iou': metrics.get('mean_iou'),
                    'pixel_accuracy': metrics.get('pixel_accuracy'),
                    'duration_seconds': (record.get('timing') or {}).get('duration_seconds'),
                    'selection_reason': record.get('config', {}).get('selection_reason'),
                    'best_model_path': record.get('best_model_path'),
                    'status': record.get('status'),
                })
        summary = {
            'num_trials': len(history),
            'best_experiment': best,
            'dataset_info': self.last_dataset_info,
            'preprocess_info': self.preprocess_info,
            'memory_reference_count': len(self.memory_history),
        }
        _write_json(output_path / 'summary.json', summary)
        lines = [
            '# Tooth Segmentation Experiment Report',
            '',
            '- Trials: {}'.format(len(history)),
            '- Best experiment: {}'.format(best['exp_id']),
            '- Best mean Dice: {}'.format(_format_metric(best.get('metrics', {}).get('mean_dice'))),
            '- Best mean HD95: {}'.format(_format_metric(best.get('metrics', {}).get('mean_hd95'))),
            '- Best pixel accuracy: {}'.format(_format_metric(best.get('metrics', {}).get('pixel_accuracy'))),
            '- Best mean IoU: {}'.format(_format_metric(best.get('metrics', {}).get('mean_iou'))),
            '- Best model path: {}'.format(best['best_model_path']),
            '- Best trainer: {}'.format(best.get('config', {}).get('nnunet_trainer', 'n/a')),
            '- Best requested epochs: {}'.format(best.get('config', {}).get('epochs', 'n/a')),
            '- Best training curve: {}'.format(best.get('artifacts', {}).get('training_curve_path', 'n/a')),
            '- Best validation summary: {}'.format(best.get('artifacts', {}).get('validation_summary_path', 'n/a')),
            '- Memory reference count: {}'.format(len(self.memory_history)),
            '',
            '## Dataset Analysis',
            '',
            '- Detected dimension: {}'.format(self.last_dataset_info.get('detected_dimension') if self.last_dataset_info else 'unknown'),
            '- imagesTr: {}'.format(self.last_dataset_info.get('counts', {}).get('imagesTr', 0) if self.last_dataset_info else 0),
            '- imagesVal: {}'.format(self.last_dataset_info.get('counts', {}).get('imagesVal', 0) if self.last_dataset_info else 0),
            '- imagesTs: {}'.format(self.last_dataset_info.get('counts', {}).get('imagesTs', 0) if self.last_dataset_info else 0),
            '',
            '## Preprocessing',
            '',
            '- Preprocessed root: {}'.format(self.preprocess_info.get('output_root') if self.preprocess_info else 'not available'),
            '- Config: {}'.format(json.dumps(_serializable(self.preprocess_info.get('config', {})) if self.preprocess_info else {}, ensure_ascii=False)),
            '',
            '## Trials',
            '',
        ]
        for record in history:
            metrics = record.get('metrics', {})
            lines.append('- {}: {} | dice={} | hd95={} | iou={} | pixel_accuracy={} | duration_s={} | epochs={} | trainer={} | reason={}'.format(
                record['exp_id'],
                record['model_name'],
                _format_metric(metrics.get('mean_dice')),
                _format_metric(metrics.get('mean_hd95')),
                _format_metric(metrics.get('mean_iou')),
                _format_metric(metrics.get('pixel_accuracy')),
                (record.get('timing') or {}).get('duration_seconds', 'n/a'),
                record.get('config', {}).get('epochs', 'n/a'),
                record.get('config', {}).get('nnunet_trainer', 'n/a'),
                record.get('config', {}).get('selection_reason', 'n/a'),
            ))
        (output_path / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
        return summary

    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, Any]:
        model.eval()
        metrics_list = []
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(device)
                labels = batch['label'].cpu().numpy()[0]
                prediction = torch.argmax(model(images), dim=1).cpu().numpy()[0]
                metrics_list.append(_compute_metrics(prediction, labels, num_classes))
        if not metrics_list:
            return {'mean_dice': 0.0, 'mean_hd95': 0.0, 'mean_iou': 0.0, 'pixel_accuracy': 0.0, 'per_class_dice': {}, 'per_class_hd95': {}, 'per_class_iou': {}}
        per_class_dice: Dict[str, List[float]] = {}
        per_class_hd95: Dict[str, List[float]] = {}
        per_class_iou: Dict[str, List[float]] = {}
        for item in metrics_list:
            for key, value in item['per_class_dice'].items():
                per_class_dice.setdefault(key, []).append(float(value))
            for key, value in item['per_class_hd95'].items():
                per_class_hd95.setdefault(key, []).append(float(value))
            for key, value in item.get('per_class_iou', {}).items():
                per_class_iou.setdefault(key, []).append(float(value))
        return {
            'mean_dice': _safe_mean([item['mean_dice'] for item in metrics_list]),
            'mean_hd95': _safe_mean([item['mean_hd95'] for item in metrics_list]),
            'mean_iou': _safe_mean([item.get('mean_iou', 0.0) for item in metrics_list]),
            'pixel_accuracy': _safe_mean([item['pixel_accuracy'] for item in metrics_list]),
            'per_class_dice': {key: _safe_mean(values) for key, values in sorted(per_class_dice.items(), key=lambda item: int(item[0]))},
            'per_class_hd95': {key: _safe_mean(values) for key, values in sorted(per_class_hd95.items(), key=lambda item: int(item[0]))},
            'per_class_iou': {key: _safe_mean(values) for key, values in sorted(per_class_iou.items(), key=lambda item: int(item[0]))},
        }

    def _run_2d_training(self, dataset_spec: DatasetSpec, task_spec: TaskSpec, exp_config: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        _seed_everything(int(exp_config.get('seed', 2026)))
        root = Path(dataset_spec.root)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_output_channels = task_spec.num_classes + 1
        train_dataset = ToothPanoramicDataset(root / dataset_spec.imagesTr, root / dataset_spec.labelsTr, int(exp_config.get('img_size', 512)), task_spec.num_classes)
        val_dataset = ToothPanoramicDataset(root / dataset_spec.imagesVal, root / dataset_spec.labelsVal, int(exp_config.get('img_size', 512)), task_spec.num_classes)
        if len(train_dataset) == 0:
            raise RuntimeError('No training images found under {}'.format(root / dataset_spec.imagesTr))
        if len(val_dataset) == 0:
            raise RuntimeError('No validation images found under {}'.format(root / dataset_spec.imagesVal))

        model = TinyUNet2D(1, num_output_channels, int(exp_config.get('base_channels', 24)), int(exp_config.get('depth', 4))).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(exp_config.get('lr', 1e-3)),
            weight_decay=float(exp_config.get('weight_decay', 1e-4)),
        )
        epochs = int(exp_config.get('epochs', 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        train_loader = DataLoader(train_dataset, batch_size=int(exp_config.get('batch_size', 1)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        best_metrics = {'mean_dice': -1.0, 'mean_hd95': 1e9, 'mean_iou': 0.0, 'pixel_accuracy': 0.0, 'per_class_dice': {}, 'per_class_hd95': {}, 'per_class_iou': {}}
        best_model_path = work_path / 'model.pth'
        train_history = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(images)
                ce_loss = F.cross_entropy(logits, labels)
                loss = 0.5 * ce_loss + 0.5 * _dice_loss(logits, labels, num_output_channels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            scheduler.step()
            val_metrics = self._evaluate_model(model, val_loader, device, task_spec.num_classes)
            train_history.append({
                'epoch': epoch + 1,
                'loss': running_loss / max(len(train_loader), 1),
                'val_mean_dice': val_metrics['mean_dice'],
                'val_mean_hd95': val_metrics['mean_hd95'],
                'val_mean_iou': val_metrics['mean_iou'],
                'val_pixel_accuracy': val_metrics['pixel_accuracy'],
            })
            if (val_metrics['mean_dice'] > best_metrics['mean_dice']) or (
                abs(val_metrics['mean_dice'] - best_metrics['mean_dice']) < 1e-8 and val_metrics['mean_hd95'] < best_metrics['mean_hd95']
            ):
                best_metrics = val_metrics
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': _serializable(exp_config),
                    'num_classes': task_spec.num_classes,
                    'class_names': task_spec.class_names,
                    'image_size': int(exp_config.get('img_size', 512)),
                    'model_name': exp_config.get('model_name', 'nnunet2d'),
                    'preprocess_info': self.preprocess_info,
                }, best_model_path)

        _write_json(work_path / 'train_meta.json', {
            'backend': 'builtin_nnunet_style_2d',
            'device': str(device),
            'config': exp_config,
            'epochs': epochs,
            'final_loss': train_history[-1]['loss'] if train_history else None,
            'num_model_classes': num_output_channels,
            'train_history': train_history,
            'preprocess_info': self.preprocess_info,
        })
        _write_json(work_path / 'metrics.json', best_metrics)
        record = ExperimentRecord(
            exp_id=work_path.name,
            task_id=task_spec.task_id,
            model_name=str(exp_config.get('model_name', 'nnunet2d')),
            config=dict(exp_config),
            best_model_path=str(best_model_path),
            metrics=best_metrics,
            work_dir=str(work_path),
            notes='; '.join((self.last_dataset_info or {}).get('warnings', [])) or None,
        )
        return _serializable(record)

    def _run_2d_inference(self, infer_input: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        output_path = Path(output_dir)
        mask_dir = output_path / 'masks'
        overlay_dir = output_path / 'overlays'
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(infer_input['model_path'])
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        image_size = int(infer_input.get('img_size') or checkpoint.get('image_size', 512))
        num_classes = int(checkpoint.get('num_classes', infer_input.get('num_classes', 32)))
        model = TinyUNet2D(1, num_classes + 1, int(config.get('base_channels', 24)), int(config.get('depth', 4)))
        model.load_state_dict(checkpoint['state_dict'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        dataset_spec = infer_input['dataset_spec']
        root = Path(dataset_spec.root)
        input_dir = Path(infer_input.get('input_dir') or (root / dataset_spec.imagesTs))
        gt_dir = Path(infer_input['gt_dir']) if infer_input.get('gt_dir') else None
        metrics_per_case = {}
        with torch.no_grad():
            for image_path in _collect_files(input_dir):
                image = np.array(Image.open(image_path).convert('L'))
                resized, meta = _resize_with_padding(image, image_size, is_mask=False)
                image_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                prediction = torch.argmax(model(image_tensor), dim=1).cpu().numpy()[0].astype(np.uint8)
                restored = _restore_from_padding(prediction, meta)
                _save_mask(mask_dir / image_path.name, restored)
                _save_overlay(overlay_dir / image_path.name, image, restored)
                if gt_dir and (gt_dir / image_path.name).exists():
                    target = np.array(Image.open(gt_dir / image_path.name))
                    metrics_per_case[image_path.name] = _compute_metrics(restored, target, num_classes)
        summary = {
            'model_path': str(model_path),
            'mask_dir': str(mask_dir),
            'overlay_dir': str(overlay_dir),
            'num_cases': len(_collect_files(input_dir)),
            'metrics_per_case': metrics_per_case,
        }
        if metrics_per_case:
            summary['mean_dice'] = _safe_mean([item['mean_dice'] for item in metrics_per_case.values()])
            summary['mean_hd95'] = _safe_mean([item['mean_hd95'] for item in metrics_per_case.values()])
            summary['mean_iou'] = _safe_mean([item['mean_iou'] for item in metrics_per_case.values()])
            summary['pixel_accuracy'] = _safe_mean([item['pixel_accuracy'] for item in metrics_per_case.values()])
        _write_json(output_path / 'inference_summary.json', summary)
        return summary

    def _run_nnunet_cli_training(self, dataset_spec: DatasetSpec, task_spec: TaskSpec, exp_config: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        nnunet_train = _resolve_nnunet_cli('train')
        nnunet_preprocess = _resolve_nnunet_cli('plan_and_preprocess')
        if not nnunet_train or not nnunet_preprocess:
            raise RuntimeError('当前环境没有可用的 nnUNetv2 CLI。需要 nnUNetv2_train 和 nnUNetv2_plan_and_preprocess。')
        root = Path(dataset_spec.root)
        dataset_id = _infer_nnunet_dataset_id(root, dataset_spec)
        if dataset_id is None:
            raise RuntimeError('nnUNet CLI 训练需要 dataset_spec.extra.nnunet_dataset_id，或让数据集根目录命名为 DatasetXXX_NAME。')
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        generated_trainer = _materialize_nnunet_trainer_subclass(exp_config, task_spec, work_dir)
        if generated_trainer:
            exp_config = {
                **exp_config,
                'nnunet_trainer': generated_trainer['trainer_name'],
                'generated_trainer_name': generated_trainer['trainer_name'],
                'trainer_definition_path': generated_trainer['trainer_definition_path'],
                'inherits_from_trainer': generated_trainer['inherits_from_trainer'],
                'trainer_identity': exp_config.get('trainer_identity') or 'adaptive_search',
            }
        dataset_info = self.last_dataset_info or self.analyze_dataset(dataset_spec, task_spec)
        configuration = str(exp_config.get('configuration', task_spec.extra.get('nnunet_configuration', _default_nnunet_configuration(dataset_info['detected_dimension']))))
        fold = str(exp_config.get('fold', task_spec.extra.get('fold', 0)))
        trainer, trainer_env, requested_epochs = _resolve_nnunet_trainer(exp_config, task_spec)
        trainer_env.update(_resolve_nnunet_trainer_env(exp_config))
        nnunet_env = {
            **dict(os.environ),
            **{key: str(value) for key, value in _resolve_nnunet_roots(root, dataset_spec).items()},
            **trainer_env,
        }
        selected_gpu = None
        if exp_config.get('auto_select_gpu', task_spec.extra.get('auto_select_gpu', True)):
            selected_gpu = _select_available_gpu()
            if selected_gpu is not None:
                nnunet_env['CUDA_VISIBLE_DEVICES'] = str(selected_gpu['index'])
                nnunet_env[DENTALCLAW_SELECTED_GPU_ENV] = str(selected_gpu['index'])
        for env_path in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
            Path(nnunet_env[env_path]).mkdir(parents=True, exist_ok=True)

        preprocess_command = [nnunet_preprocess, '-d', str(dataset_id), '--verify_dataset_integrity']
        preprocess_result = subprocess.run(preprocess_command, cwd=str(work_path), capture_output=True, text=True, env=nnunet_env)
        env_keys = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
        if NNUNET_DYNAMIC_EPOCH_ENV in nnunet_env:
            env_keys.append(NNUNET_DYNAMIC_EPOCH_ENV)
        for env_key in (
            DENTALCLAW_NNUNET_INITIAL_LR_ENV,
            DENTALCLAW_NNUNET_WEIGHT_DECAY_ENV,
            DENTALCLAW_NNUNET_OVERSAMPLE_ENV,
            DENTALCLAW_NNUNET_LR_SCHEDULER_ENV,
            DENTALCLAW_SELECTED_GPU_ENV,
            'CUDA_VISIBLE_DEVICES',
        ):
            if env_key in nnunet_env:
                env_keys.append(env_key)
        _write_json(
            work_path / 'nnunet_preprocess_command.json',
            {
                'command': preprocess_command,
                'returncode': preprocess_result.returncode,
                'env': {k: nnunet_env[k] for k in env_keys},
                'selected_gpu': selected_gpu,
            },
        )
        (work_path / 'preprocess_stdout.log').write_text(preprocess_result.stdout or '', encoding='utf-8')
        (work_path / 'preprocess_stderr.log').write_text(preprocess_result.stderr or '', encoding='utf-8')
        if preprocess_result.returncode != 0:
            raise RuntimeError('nnUNet plan_and_preprocess failed with code {}'.format(preprocess_result.returncode))

        train_command = [nnunet_train, str(dataset_id), configuration, fold]
        if trainer:
            train_command.extend(['-tr', str(trainer)])
        train_result = subprocess.run(train_command, cwd=str(work_path), capture_output=True, text=True, env=nnunet_env)
        _write_json(
            work_path / 'nnunet_command.json',
            {
                'command': train_command,
                'returncode': train_result.returncode,
                'env': {k: nnunet_env[k] for k in env_keys},
                'selected_gpu': selected_gpu,
            },
        )
        (work_path / 'stdout.log').write_text(train_result.stdout or '', encoding='utf-8')
        (work_path / 'stderr.log').write_text(train_result.stderr or '', encoding='utf-8')
        if train_result.returncode != 0:
            raise RuntimeError('nnUNet training failed with code {}'.format(train_result.returncode))

        dataset_dir_pattern = 'Dataset{:03d}_*'.format(dataset_id)
        raw_root = Path(nnunet_env['nnUNet_raw'])
        results_root = Path(nnunet_env['nnUNet_results'])
        dataset_dir = next(iter(sorted(raw_root.glob(dataset_dir_pattern))), None)
        dataset_name = dataset_dir.name if dataset_dir else dataset_dir_pattern
        trainer_dir = results_root / dataset_name / '{}__nnUNetPlans__{}'.format(trainer, configuration)
        fold_dir = trainer_dir / 'fold_{}'.format(fold)
        if not fold_dir.exists():
            fold_candidates = sorted(results_root.glob('{}/**/fold_{}'.format(dataset_dir_pattern, fold)))
            fold_dir = fold_candidates[-1] if fold_candidates else trainer_dir

        snapshot_root = work_path / 'nnUNet_results_snapshot'
        snapshot_trainer_dir = snapshot_root / dataset_name / '{}__nnUNetPlans__{}'.format(trainer, configuration)
        snapshot_fold_dir = snapshot_trainer_dir / 'fold_{}'.format(fold)
        if trainer_dir.exists():
            _replace_tree(trainer_dir, snapshot_trainer_dir)
        elif fold_dir.exists():
            _replace_tree(fold_dir, snapshot_fold_dir)

        best_model_path = fold_dir / 'checkpoint_best.pth'
        if not best_model_path.exists():
            checkpoint_candidates = sorted(results_root.glob('{}/**/checkpoint_best.pth'.format(dataset_dir_pattern)))
            best_model_path = checkpoint_candidates[-1] if checkpoint_candidates else fold_dir

        validation_summary_path = fold_dir / 'validation' / 'summary.json'
        if not validation_summary_path.exists():
            metrics_path_candidates = sorted(results_root.glob('{}/**/validation/summary.json'.format(dataset_dir_pattern)))
            validation_summary_path = metrics_path_candidates[-1] if metrics_path_candidates else Path()
        progress_png_path = fold_dir / 'progress.png'
        if not progress_png_path.exists():
            progress_candidates = sorted(results_root.glob('{}/**/progress.png'.format(dataset_dir_pattern)))
            progress_png_path = progress_candidates[-1] if progress_candidates else Path()

        snapshot_best_model_path = snapshot_fold_dir / 'checkpoint_best.pth'
        if snapshot_best_model_path.exists():
            best_model_path = snapshot_best_model_path
        snapshot_validation_summary_path = snapshot_fold_dir / 'validation' / 'summary.json'
        if snapshot_validation_summary_path.exists():
            validation_summary_path = snapshot_validation_summary_path
        snapshot_progress_png_path = snapshot_fold_dir / 'progress.png'
        if snapshot_progress_png_path.exists():
            progress_png_path = snapshot_progress_png_path

        metrics = {'mean_dice': 0.0, 'mean_hd95': None, 'pixel_accuracy': None, 'mean_iou': None, 'per_class_dice': {}, 'per_class_hd95': {}, 'per_class_iou': {}}
        notes = 'nnUNet CLI path stores checkpoints under {}'.format(results_root)
        if requested_epochs is not None:
            notes += '; requested_epochs={}'.format(requested_epochs)
        notes += '; trainer={}'.format(trainer)
        if selected_gpu is not None:
            notes += '; selected_gpu={} ({})'.format(selected_gpu['index'], selected_gpu['name'])
        if validation_summary_path.exists():
            metrics = _parse_nnunet_summary(validation_summary_path)
            notes += '; validation_summary={}'.format(validation_summary_path)
        if progress_png_path.exists():
            notes += '; progress_png={}'.format(progress_png_path)
        if snapshot_root.exists():
            notes += '; snapshot_root={}'.format(snapshot_root)
        record = ExperimentRecord(
            exp_id=work_path.name,
            task_id=task_spec.task_id,
            model_name='nnUNet-{}'.format(configuration),
            config=dict(
                exp_config,
                configuration=configuration,
                fold=fold,
                nnunet_trainer=trainer,
                epochs=requested_epochs or exp_config.get('epochs'),
                selected_gpu_index=selected_gpu.get('index') if selected_gpu else None,
                selected_gpu_name=selected_gpu.get('name') if selected_gpu else None,
            ),
            best_model_path=str(best_model_path),
            metrics=metrics,
            work_dir=str(work_path),
            notes=notes,
        )
        payload = _serializable(record)
        payload['backend'] = 'nnUNetv2_cli'
        payload['artifacts'] = {
            'result_dir': str(fold_dir),
            'trainer_dir': str(trainer_dir),
            'trainer_definition_path': exp_config.get('trainer_definition_path'),
            'snapshot_result_dir': str(snapshot_fold_dir) if snapshot_fold_dir.exists() else None,
            'snapshot_trainer_dir': str(snapshot_trainer_dir) if snapshot_trainer_dir.exists() else None,
            'nnunet_results_root': str(snapshot_root) if snapshot_root.exists() else str(results_root),
            'training_curve_path': str(progress_png_path) if progress_png_path.exists() else None,
            'validation_summary_path': str(validation_summary_path) if validation_summary_path.exists() else None,
            'train_command_path': str(work_path / 'nnunet_command.json'),
            'preprocess_command_path': str(work_path / 'nnunet_preprocess_command.json'),
            'selected_gpu': selected_gpu,
        }
        return payload

    def _run_nnunet_cli_inference(self, infer_input: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        nnunet_predict = _resolve_nnunet_cli('predict')
        if not nnunet_predict:
            raise RuntimeError('当前环境没有可用的 nnUNetv2_predict。')
        dataset_spec = infer_input['dataset_spec']
        root = Path(dataset_spec.root)
        dataset_id = _infer_nnunet_dataset_id(root, dataset_spec)
        if dataset_id is None:
            raise RuntimeError('nnUNet CLI 推理需要 dataset_spec.extra.nnunet_dataset_id，或让数据集根目录命名为 DatasetXXX_NAME。')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        input_dir = infer_input.get('input_dir') or str(root / dataset_spec.imagesTs)
        input_dir_path = Path(input_dir)
        gt_dir = Path(infer_input['gt_dir']) if infer_input.get('gt_dir') else None
        configuration = str(infer_input.get('configuration') or infer_input.get('task_spec').extra.get('nnunet_configuration', '2d'))
        fold = str(infer_input.get('fold') or infer_input.get('task_spec').extra.get('fold', 0))
        trainer = str(
            infer_input.get('nnunet_trainer')
            or ((infer_input.get('config') or {}).get('nnunet_trainer'))
            or infer_input.get('task_spec').extra.get('nnunet_trainer')
            or 'nnUNetTrainer'
        )
        resolved_roots = {key: str(value) for key, value in _resolve_nnunet_roots(root, dataset_spec).items()}
        results_override = infer_input.get('nnunet_results_root')
        if results_override:
            resolved_roots['nnUNet_results'] = str(results_override)
        nnunet_env = {**dict(os.environ), **resolved_roots}
        command = [nnunet_predict, '-i', str(input_dir), '-o', str(output_path), '-d', str(dataset_id), '-c', configuration, '-f', fold]
        if trainer:
            command.extend(['-tr', trainer])
        result = subprocess.run(command, capture_output=True, text=True, env=nnunet_env)
        _write_json(output_path / 'nnunet_predict_command.json', {'command': command, 'returncode': result.returncode, 'env': {k: nnunet_env[k] for k in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')}})
        (output_path / 'stdout.log').write_text(result.stdout or '', encoding='utf-8')
        (output_path / 'stderr.log').write_text(result.stderr or '', encoding='utf-8')
        if result.returncode != 0:
            raise RuntimeError('nnUNet inference failed with code {}'.format(result.returncode))
        summary = _evaluate_nnunet_prediction_dir(
            prediction_dir=output_path,
            input_dir=input_dir_path,
            gt_dir=gt_dir,
            num_classes=int(infer_input.get('num_classes') or infer_input.get('task_spec').num_classes),
        )
        summary.update({
            'output_dir': str(output_path),
            'command': command,
            'summary_json_path': str(output_path / 'inference_summary.json'),
            'summary_md_path': str(output_path / 'inference_summary.md'),
        })
        _write_inference_summary_bundle(output_path, summary)
        return summary
