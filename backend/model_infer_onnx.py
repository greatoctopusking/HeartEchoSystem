import os
import sys
import numpy as np
import nibabel as nib
import torch
import onnxruntime as ort
from scipy.ndimage import label
from config import RESULT_FOLDER  
import math

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 添加 models 目录到 Python 路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

ONNX_PATHS={
    "2ch":os.path.join(PROJECT_ROOT,"models","nnUNet_results","echo_seg_2ch_fast.onnx"),
    "4ch":os.path.join(PROJECT_ROOT,"models","nnUNet_results","echo_seg_4ch_fast.onnx"),
}

_sessions={"2ch":None,"4ch":None}

def detect_device():
    if torch.cuda.is_available():
        print(">>> Using GPU (CUDA)")
        return "cuda"
    else:
        print(">>> Using CPU")
        return "cpu"

def _get_session(view:str):
    """单例模式加载 ONNX 模型，保证启动速度"""
    if _sessions[view] is None:
        if not os.path.exists(ONNX_PATHS[view]):
            raise FileNotFoundError(f"[报错] 找不到 {view.upper()} 的 ONNX 模型文件:{ONNX_PATHS[view]}")
            
        providers=(
            ["CUDAExecutionProvider","CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        _sessions[view]=ort.InferenceSession(ONNX_PATHS[view],providers=providers)
        print(f"[ONNX] 🚀 {view.upper()} 极速模型加载完毕 (Providers:{providers[0]})")
    return _sessions[view]

def _preprocess_frame(frame:np.ndarray):
    """图像预处理：Z-score归一化 -> Padding到128的倍数 -> 调整形状"""
    orig_h,orig_w=frame.shape
    
    #1. 先做归一化
    frame_norm=(frame - frame.mean())/(frame.std() + 1e-8)
    
    #2. 计算需要 padding 到的目标尺寸 (改为 128 的倍数！)
    MULTIPLE=128
    target_h=math.ceil(orig_h/MULTIPLE)*MULTIPLE
    target_w=math.ceil(orig_w/MULTIPLE)*MULTIPLE
    
    pad_h=target_h - orig_h
    pad_w=target_w - orig_w

    print(f"\n[DEBUG PADDING] 原始尺寸:{orig_h}x{orig_w} -> 目标尺寸:{target_h}x{target_w}")
    
    #3. 在右侧和下方填充 0
    if pad_h > 0 or pad_w > 0:
        frame_padded=np.pad(frame_norm,((0,pad_h),(0,pad_w)),mode='constant',constant_values=0)
    else:
        frame_padded=frame_norm
        
    #调整形状并返回：同时返回原始尺寸，留着后面裁剪用
    tensor=frame_padded[np.newaxis,np.newaxis,:,:].astype(np.float32)
    return tensor,(orig_h,orig_w)

def _postprocess_logits(logits:np.ndarray) -> np.ndarray:
    """后处理：求最大概率类别 + 提取最大连通域(去除零星噪点)"""
    mask=np.argmax(logits,axis=1)[0]  #shape:(H,W)
    cleaned_mask=np.zeros_like(mask)
    classes=np.unique(mask)
    
    for cls in classes:
        if cls == 0:continue #跳过背景
        binary_mask=(mask == cls).astype(np.uint8)
        labeled_array,num_features=label(binary_mask)
        if num_features > 0:
            bincount=np.bincount(labeled_array.ravel())
            bincount[0]=0 #忽略背景
            largest_cc_label=bincount.argmax()
            cleaned_mask[labeled_array == largest_cc_label]=cls
            
    return cleaned_mask


#2. 核心对接函数
def run_inference(image_path,case_name,dataset_name):
    """
    执行 ONNX 极速推理 (直接替换原来的 nnUNet 命令行调用)
    返回:
        output_dir:结果存放目录
        pred_paths:所有预测 mask 路径列表
    """
    print(f"\n[ONNX INFER] Processing file:{image_path}")
    
    #1. 自动判断当前是 2腔心 还是 4腔心
    view="2ch" if "2ch" in case_name.lower() or "2ch" in dataset_name.lower() else "4ch"
    
    #2. 加载原图，读取 Header 等信息
    orig_img=nib.load(image_path)
    data=orig_img.get_fdata()
    header=orig_img.header

    if data.ndim != 3:
        raise RuntimeError("Input must be 3D time-sequence (H,W,T)")

    #3. 创建输出目录
    output_dir=os.path.join(RESULT_FOLDER,case_name)
    os.makedirs(output_dir,exist_ok=True)

    #获取 spacing，构建正确的 2D affine
    zooms=header.get_zooms()
    dx,dy=float(zooms[0]),float(zooms[1])
    affine_2d=np.diag([dx,dy,1.0,1.0])
    
    T=data.shape[2]
    pred_paths=[]
    
    #获取 ONNX Session
    session=_get_session(view)
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name

    print(f"[ONNX INFER] 开始逐帧推理 ({T} 帧)...")
    
    #4. 核心推理循环 (跳过一切磁盘临时文件，全在内存里飞)
#4. 核心推理循环 (跳过一切磁盘临时文件，全在内存里飞)
    for t in range(T):
        frame=data[:,:,t].astype(np.float32)

        #推理流水线
        #注意这里接收了 orig_shape
        input_tensor,orig_shape=_preprocess_frame(frame) 
        
        logits=session.run([output_name],{input_name:input_tensor})[0]
        mask_padded=_postprocess_logits(logits)
        
        #【新增】：将 padding 过的 mask 裁剪回原始尺寸
        orig_h,orig_w=orig_shape
        mask=mask_padded[:orig_h,:orig_w] 
        
        #将内存中的 mask 保存为 NIfTI，给 app.py 用
        mask_nii=nib.Nifti1Image(mask.astype(np.int16),affine_2d)
        mask_nii.header.set_zooms((dx,dy))

        frame_name=f"{case_name}_{t:03d}"
        pred_file=os.path.join(output_dir,f"{frame_name}.nii.gz")
        nib.save(mask_nii,pred_file)
        
        pred_paths.append(pred_file)

    print(f"[ONNX INFER] ✅ {case_name} 推理完毕！输出目录:{output_dir}")
    
    #返回格式和原版完全一致，完美对接 app.py
    return output_dir,pred_paths