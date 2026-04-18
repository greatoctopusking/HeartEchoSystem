import os
import shutil
import sys
import numpy as np
import nibabel as nib
import torch

from config import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
NNUNET_MODEL_ROOT = os.path.join(PROJECT_ROOT, "models", "nnUNet_results")

# 添加 models 目录到 Python 路径，以便导入 nnunetv2
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

_predictors = {"2ch": None, "4ch": None}


def detect_device():
    if torch.cuda.is_available():
        print(">>> Using GPU (CUDA)")
        return torch.device("cuda")
    else:
        print(">>> Using CPU")
        return torch.device("cpu")


def check_model_available(view: str) -> bool:
    """检查 nnUNet 模型是否可用"""
    dataset_name = NNUNET_DATASET_2CH if view == "2ch" else NNUNET_DATASET_4CH
    model_folder = os.path.join(
        NNUNET_MODEL_ROOT,
        dataset_name,
        f"nnUNetTrainer__nnUNetPlans__{NNUNET_CONFIGURATION}"
    )
    return os.path.exists(model_folder)


def _get_predictor(view: str):
    """单例模式加载 nnUNet Predictor"""
    if _predictors[view] is None:
        # 先检查模型是否存在
        if not check_model_available(view):
            dataset_name = NNUNET_DATASET_2CH if view == "2ch" else NNUNET_DATASET_4CH
            model_folder = os.path.join(
                NNUNET_MODEL_ROOT,
                dataset_name,
                f"nnUNetTrainer__nnUNetPlans__{NNUNET_CONFIGURATION}"
            )
            raise FileNotFoundError(
                f"nnUNet model not found: {model_folder}\n"
                f"Please train the model or use ONNX inference instead."
            )
        
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = detect_device()
        
        dataset_name = NNUNET_DATASET_2CH if view == "2ch" else NNUNET_DATASET_4CH
        model_folder = os.path.join(
            NNUNET_MODEL_ROOT,
            dataset_name,
            f"nnUNetTrainer__nnUNetPlans__{NNUNET_CONFIGURATION}"
        )
        
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Model folder not found: {model_folder}")
        
        print(f"[nnUNet API] Loading model from: {model_folder}")
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            allow_tqdm=False
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(int(NNUNET_FOLD),),
            checkpoint_name="checkpoint_final.pth"
        )
        _predictors[view] = predictor
        print(f"[nnUNet API] ✅ {view.upper()} model loaded")
    
    return _predictors[view]


def run_inference(image_path, case_name, dataset_name):
    print(f"Processing file: {image_path}")
    print(f"Using dataset: {dataset_name}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在: {image_path}")

    orig_img = nib.load(image_path)
    data = orig_img.get_fdata().astype(np.float32)
    header = orig_img.header

    if data.ndim != 3:
        raise RuntimeError(f"Input must be 3D time-sequence (H,W,T), but got shape={data.shape}")

    if data.shape[2] <= 0:
        raise RuntimeError("输入序列时间帧数为 0")

    view = "2ch" if "2ch" in case_name.lower() or "2ch" in dataset_name.lower() else "4ch"

    output_dir = os.path.join(RESULT_FOLDER, case_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    T = data.shape[2]

    zooms = header.get_zooms()
    dx = float(zooms[0]) if len(zooms) > 0 else 1.0
    dy = float(zooms[1]) if len(zooms) > 1 else 1.0

    print(f"[INFO] Original spacing: dx={dx}, dy={dy}, zooms={zooms}")

    affine_2d = np.diag([dx, dy, 1.0, 1.0])

    case_input_dir = os.path.join(UPLOAD_FOLDER, f"nnunet_input_{case_name}")
    if os.path.exists(case_input_dir):
        shutil.rmtree(case_input_dir)
    os.makedirs(case_input_dir, exist_ok=True)

    for t in range(T):
        frame = data[:, :, t].astype(np.float32)

        p1, p99 = np.percentile(frame, [1, 99])
        if p99 > p1:
            frame = np.clip((frame - p1) / (p99 - p1), 0.0, 1.0) * 255.0
        else:
            mn, mx = float(frame.min()), float(frame.max())
            if mx > mn:
                frame = (frame - mn) / (mx - mn) * 255.0
            else:
                frame = np.zeros_like(frame, dtype=np.float32)

        frame_nii = nib.Nifti1Image(frame.astype(np.float32), affine_2d)
        frame_nii.header.set_zooms((dx, dy))

        frame_name = f"{case_name}_{t:03d}"
        input_path = os.path.join(case_input_dir, f"{frame_name}_0000.nii.gz")
        nib.save(frame_nii, input_path)

    predictor = _get_predictor(view)

    input_list = [[os.path.join(case_input_dir, f"{case_name}_{t:03d}_0000.nii.gz")] for t in range(T)]
    output_list = [os.path.join(output_dir, f"{case_name}_{t:03d}") for t in range(T)]

    print(f"[nnUNet API] Running inference on {T} frames...")
    
    predictor.predict_from_files(
        input_list,
        output_list,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        num_parts=1,
        part_id=0
    )

    pred_paths = []
    for t in range(T):
        frame_name = f"{case_name}_{t:03d}"
        pred_file = os.path.join(output_dir, f"{frame_name}.nii.gz")
        if not os.path.exists(pred_file):
            raise RuntimeError(f"预测结果缺失: {pred_file}")
        pred_paths.append(pred_file)

    return output_dir, pred_paths
