import shutil
import subprocess
import numpy as np
import nibabel as nib
import torch
from config import *


def detect_device():
    if torch.cuda.is_available():
        print(">>> Using GPU (CUDA)")
        return "cuda"
    else:
        print(">>> Using CPU")
        return "cpu"


def run_inference(image_path, case_name, dataset_name):
    """
    执行 nnUNet 推理
    返回:
        output_dir
        所有预测 mask 路径列表
    """
    print(f"Processing file: {image_path}")
    print(f"Using dataset: {dataset_name}")

    orig_img = nib.load(image_path)
    data = orig_img.get_fdata()
    header = orig_img.header

    if data.ndim != 3:
        raise RuntimeError("Input must be 3D time-sequence (H,W,T)")

    if not os.environ.get("nnUNet_raw"):
        raise RuntimeError("nnUNet environment variables not set correctly")

    case_input_dir = os.path.join(UPLOAD_FOLDER, f"nnunet_input_{case_name}")
    if os.path.exists(case_input_dir):
        shutil.rmtree(case_input_dir)
    os.makedirs(case_input_dir, exist_ok=True)

    output_dir = os.path.join(RESULT_FOLDER, case_name)
    os.makedirs(output_dir, exist_ok=True)

    T = data.shape[2]

    # ✅ 修复：获取 spacing，构建正确的 2D affine 和 header
    zooms = header.get_zooms()
    dx = float(zooms[0])
    dy = float(zooms[1])

    print(f"[INFO] Original spacing: dx={dx}, dy={dy}, zooms={zooms}")

    # 构建标准 2D affine（对角矩阵，单位 mm）
    affine_2d = np.diag([dx, dy, 1.0, 1.0])

    for t in range(T):
        frame = data[:, :, t].astype(np.float32)


        frame_nii = nib.Nifti1Image(frame, affine_2d)
        frame_nii.header.set_zooms((dx, dy))

        frame_name = f"{case_name}_{t:03d}"
        input_path = os.path.join(case_input_dir, f"{frame_name}_0000.nii.gz")
        nib.save(frame_nii, input_path)

    device = detect_device()
    # nnUNet 推理
    command = [
        "nnUNetv2_predict",
        "-i", case_input_dir,
        "-o", output_dir,
        "-d", dataset_name,
        "-c", NNUNET_CONFIGURATION,
        "-f", str(NNUNET_FOLD),
        "--disable_tta",
        "-device", device,
        "-npp", "1",
        "-nps", "1",
    ]

    print("Running command:", " ".join(command))
    proc = subprocess.run(command, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"nnUNet failed:\n{proc.stderr}")

    pred_paths = []
    for t in range(T):
        frame_name = f"{case_name}_{t:03d}"
        pred_file = os.path.join(output_dir, f"{frame_name}.nii.gz")
        pred_paths.append(pred_file)

    return output_dir, pred_paths
