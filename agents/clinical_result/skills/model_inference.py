# agents/clinical_result/skills/model_inference.py

from pathlib import Path
from types import SimpleNamespace
import shutil

from skills.tooth_segmentation_skill import ToothSegmentationSkill


def model_inference(case, model_path, work_dir):
    """
    单模型推理：
    - case["image_path"] 是新病例原图
    - model_path 是某个 fold 的 model.pth
    - work_dir 是本次模型推理的工作目录
    """
    image_path = Path(case["image_path"])
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) 准备临时输入目录
    input_dir = work_dir / "input_case"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, input_dir / image_path.name)

    # 2) 构造最小 dataset_spec
    dataset_spec = SimpleNamespace(
        root=str(work_dir),
        imagesTr="empty_imagesTr",
        labelsTr="empty_labelsTr",
        imagesVal="empty_imagesVal",
        labelsVal="empty_labelsVal",
        imagesTs="input_case",
        extra={},
    )

    skill = ToothSegmentationSkill()
    infer_input = {
        "dataset_spec": dataset_spec,
        "input_dir": str(input_dir),
        "model_path": str(model_path),
        "img_size": case.get("img_size", 512),
    }

    # 3) 调用你训练工程里的推理
    result = skill.run_inference(
        infer_input=infer_input,
        output_dir=str(work_dir / "outputs"),
    )

    return result