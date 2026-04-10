import os
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


def run_inference(image_path,case_name,dataset_name):
    print(f"Processing file:{image_path}")
    print(f"Using dataset:{dataset_name}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入文件不存在:{image_path}")

    orig_img=nib.load(image_path)
    data=orig_img.get_fdata().astype(np.float32)
    header=orig_img.header

    if data.ndim!=3:
        raise RuntimeError(f"Input must be 3D time-sequence (H,W,T),but got shape={data.shape}")

    if data.shape[2]<=0:
        raise RuntimeError("输入序列时间帧数为 0")

    if not os.environ.get("nnUNet_raw"):
        raise RuntimeError("nnUNet environment variables not set correctly")

    case_input_dir=os.path.join(UPLOAD_FOLDER,f"nnunet_input_{case_name}")
    if os.path.exists(case_input_dir):
        shutil.rmtree(case_input_dir)
    os.makedirs(case_input_dir,exist_ok=True)

    output_dir=os.path.join(RESULT_FOLDER,case_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    T=data.shape[2]

    zooms=header.get_zooms()
    dx=float(zooms[0]) if len(zooms)>0 else 1.0
    dy=float(zooms[1]) if len(zooms)>1 else 1.0

    print(f"[INFO] Original spacing:dx={dx},dy={dy},zooms={zooms}")

    affine_2d=np.diag([dx,dy,1.0,1.0])

    for t in range(T):
        frame=data[:,:,t].astype(np.float32)

        p1,p99=np.percentile(frame,[1,99])
        if p99>p1:
            frame=np.clip((frame - p1)/(p99 - p1),0.0,1.0)*255.0
        else:
            mn,mx=float(frame.min()),float(frame.max())
            if mx>mn:
                frame=(frame - mn)/(mx - mn)*255.0
            else:
                frame=np.zeros_like(frame,dtype=np.float32)

        frame_nii=nib.Nifti1Image(frame.astype(np.float32),affine_2d)
        frame_nii.header.set_zooms((dx,dy))

        frame_name=f"{case_name}_{t:03d}"
        input_path=os.path.join(case_input_dir,f"{frame_name}_0000.nii.gz")
        nib.save(frame_nii,input_path)

    command=[
        "nnUNetv2_predict",
        "-i",case_input_dir,
        "-o",output_dir,
        "-d",dataset_name,
        "-c",NNUNET_CONFIGURATION,
        "-f",str(NNUNET_FOLD),
        "--disable_tta",
        "-device","cpu",
        "-npp","1",
        "-nps","1",
    ]

    print("Running command:"," ".join(command))
    proc=subprocess.run(command,capture_output=True,text=True)

    if proc.returncode!=0:
        raise RuntimeError(f"nnUNet failed:\n{proc.stderr}")

    pred_paths=[]
    for t in range(T):
        frame_name=f"{case_name}_{t:03d}"
        pred_file=os.path.join(output_dir,f"{frame_name}.nii.gz")
        if not os.path.exists(pred_file):
            raise RuntimeError(f"预测结果缺失:{pred_file}")
        pred_paths.append(pred_file)

    return output_dir,pred_paths
