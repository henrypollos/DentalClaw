from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _build_predictor(
    trainer_folder: str,
    use_folds: Sequence[int] = (0,),
    checkpoint_name: str = "checkpoint_final.pth",
    use_tta: bool = True,
):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        trainer_folder,
        use_folds=tuple(use_folds),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def _majority_vote(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("masks cannot be empty")
    stacked = np.stack(masks, axis=0)  # [N, H, W]
    return np.apply_along_axis(lambda x: np.bincount(x.astype(np.int64)).argmax(), 0, stacked).astype(np.uint8)


from pathlib import Path
import cv2
import numpy as np

def predict_single_case(
    image_path: str,
    trainer_folder: str,
    output_dir: str,
    use_folds=(0,),
    checkpoint_name="checkpoint_final.pth",
    use_tta=True,
):
    predictor = _build_predictor(
        trainer_folder=trainer_folder,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name,
        use_tta=use_tta,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor.predict_from_files(
        [[str(image_path)]],
        str(out_dir),   
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

    case_name = Path(image_path).name
    out_mask = out_dir / case_name

    if not out_mask.exists():
        raise FileNotFoundError(f"nnUNet output not found: {out_mask}")

    mask = cv2.imread(str(out_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {out_mask}")

    return mask, str(out_mask)


def predict_ensemble(
    image_path: str,
    trainer_folders: Sequence[str],
    output_dir: str,
    use_folds: Sequence[int] = (0,),
    checkpoint_name: str = "checkpoint_final.pth",
    use_tta: bool = True,
) -> Tuple[np.ndarray, List[dict], List[str]]:
    """
    Ensemble by running multiple nnUNet model folders and majority-voting masks.
    Returns:
        merged_mask, per_model_outputs, mask_paths
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    per_model_outputs = []
    mask_paths = []

    for idx, trainer_folder in enumerate(trainer_folders, start=1):
        model_dir = out_dir / f"model_{idx}"
        mask, mask_path = predict_single_case(
            image_path=image_path,
            trainer_folder=trainer_folder,
            output_dir=str(model_dir),
            use_folds=use_folds,
            checkpoint_name=checkpoint_name,
            use_tta=use_tta,
        )
        masks.append(mask)
        mask_paths.append(mask_path)
        per_model_outputs.append(
            {
                "model_name": Path(trainer_folder).name,
                "foreground_pixels": int((mask > 0).sum()),
                "labels_detected": [int(v) for v in np.unique(mask) if int(v) != 0],
                "notes": "nnUNet single-model prediction",
                "mask_path": mask_path,
            }
        )

    merged = _majority_vote(masks)
    merged_path = out_dir / f"{Path(image_path).stem}_ensemble.png"
    cv2.imwrite(str(merged_path), merged)

    return merged, per_model_outputs, mask_paths