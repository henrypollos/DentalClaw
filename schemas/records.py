# schemas/records.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ExperimentRecord:
    exp_id: str
    task_id: str
    model_name: str
    config: Dict[str, Any]
    best_model_path: str
    metrics: Dict[str, Any]
    work_dir: str
    status: str = "completed"
    notes: Optional[str] = None