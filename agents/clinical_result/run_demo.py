from pathlib import Path
from agents.clinical_result.agent import ClinicalResultAgent

def main():
    print(">>> START DEMO")

    case_id = "teeth_0001"

    case = {
        "id": case_id,
        "image_path": "/data/data2/yiyang/JoD/nnUNet/nnUNet_raw/Dataset106_Teeth32_Labelbox/imagesTr/teeth_0001_0000.png",
        "label_path": "/data/data2/yiyang/JoD/nnUNet/nnUNet_raw/Dataset106_Teeth32_Labelbox/labelsTr/teeth_0001.png",
    }

    agent = ClinicalResultAgent(config={
        "use_ensemble": False,
        "use_tta": True,
        "anonymize": True,

        # ✅ nnUNet trainer folder（关键）
        "model_paths": [
            "/data/data2/yiyang/JoD/nnUNet/nnUNet_results/"
            "Dataset106_Teeth32_Labelbox/"
            "nnUNetTrainer__nnUNetPlans__2d"
        ],

        "nnunet_folds": (0,1,2,3,4),
        "checkpoint_name": "checkpoint_best.pth",
    })

    out_dir = Path("./artifacts/demo_output")

    result = agent.run(case, out_dir=str(out_dir))

    print(">>> DONE")
    print(result["summary"])
    print(result["review_list"])
    print("HTML report:", result.get("html_path"))
    print("Overlay:", result.get("overlay_path"))


if __name__ == "__main__":
    main()