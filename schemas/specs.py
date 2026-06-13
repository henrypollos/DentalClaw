# schemas/specs.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DatasetSpec:
    root: str
    imagesTr: str = "imagesTr"
    labelsTr: str = "labelsTr"
    imagesVal: str = "imagesVal"
    labelsVal: str = "labelsVal"
    imagesTs: str = "imagesTs"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    task_id: str
    modality: str
    task_type: str
    num_classes: int
    class_names: List[str]
    primary_metric: str = "mean_dice"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetSpec:
    max_trials: int = 5
    max_epochs_per_trial: int = 100
    max_parallel: int = 1
