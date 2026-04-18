"""
心脏超声视图分类器 V2 - 改进版
支持多帧投票和翻转检测
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, List
import nibabel as nib
from PIL import Image

# 添加模型路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

from .classifier import load_model
from .transforms import get_val_transforms, load_image
from .constants import ALL_CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

# 分类器配置
CLASSIFIER_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "classifier_mode", "best_model.pth")
CONFIDENCE_THRESHOLD = 0.6

# 视图类型映射（7分类 -> 简化类型）
VIEW_TYPE_MAPPING = {
    'A2C': '2ch',      # 两腔心
    'A4C': '4ch',      # 四腔心
    'PL': 'unknown',   # 胸骨旁长轴
    'PSAV': 'unknown', # 胸骨旁短轴-心底
    'PSMV': 'unknown', # 胸骨旁短轴-二尖瓣
    'SC': 'unknown',   # 锁骨下
    'Random': 'unknown',  # 随机/非标准视图
}

# 类别中文名
CLASS_NAMES_CN = {
    'A2C': '两腔心(2CH)',
    'A4C': '四腔心(4CH)',
    'PL': '胸骨旁长轴',
    'PSAV': '短轴-心底',
    'PSMV': '短轴-二尖瓣',
    'SC': '锁骨下',
    'Random': '非标准视图',
}

# 全局单例模型
_classifier_model = None
_classifier_transform = None
_classifier_device = None


def get_classifier_device():
    """获取推理设备"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_classifier(device: Optional[str] = None) -> bool:
    """初始化分类器模型（单例模式）"""
    global _classifier_model, _classifier_transform, _classifier_device
    
    if _classifier_model is not None:
        return True
    
    if not os.path.exists(CLASSIFIER_CHECKPOINT):
        print(f"[WARN] Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}")
        return False
    
    try:
        _classifier_device = device or get_classifier_device()
        print(f"[INFO] Loading 7-class classifier model from {CLASSIFIER_CHECKPOINT}")
        
        _classifier_model = load_model(
            checkpoint_path=CLASSIFIER_CHECKPOINT,
            num_classes=7,
            freeze_backbone=True,
            device=_classifier_device
        )
        
        _classifier_transform = get_val_transforms(image_size=224)
        
        print(f"[INFO] Classifier loaded successfully on {_classifier_device}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """预处理单帧图像为 tensor"""
    # 归一化到 0-255
    if frame.max() > frame.min():
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
    
    # 转为 PIL Image
    pil_img = Image.fromarray(frame.astype(np.uint8), mode='L').convert('RGB')
    
    # 应用 transform
    tensor = _classifier_transform(pil_img)
    return tensor


def classify_single_frame(frame: np.ndarray, use_flip: bool = False) -> Dict:
    """
    分类单帧图像
    
    Args:
        frame: 单帧图像数组 (H, W)
        use_flip: 是否垂直翻转图像
    
    Returns:
        分类结果字典
    """
    if use_flip:
        frame = np.flipud(frame)  # 上下翻转
    
    tensor = preprocess_frame(frame)
    tensor = tensor.unsqueeze(0).to(_classifier_device)
    
    with torch.no_grad():
        output = _classifier_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    # 转为 numpy
    probs_np = probs.cpu().numpy()
    
    return {
        'probs': probs_np,
        'pred_idx': int(probs.argmax()),
        'confidence': float(probs.max())
    }


def extract_all_frames(nifti_path: str) -> Optional[List[np.ndarray]]:
    """
    从 NIfTI 文件中提取所有帧
    
    Returns:
        帧列表，每个帧是 (H, W) 数组
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        print(f"[INFO] NIfTI shape: {data.shape}, dtype: {data.dtype}")
        
        # 处理 4D (H, W, Z, T) 或 3D (H, W, T) 数据
        if data.ndim == 4:
            # 4D: 取中间切片的所有时间帧
            z_idx = data.shape[2] // 2
            data = data[:, :, z_idx, :]  # (H, W, T)
            print(f"[INFO] Extracted middle slice (z={z_idx}), new shape: {data.shape}")
        
        if data.ndim == 3:
            # 3D: (H, W, T)
            frames = [data[:, :, t] for t in range(data.shape[2])]
        elif data.ndim == 2:
            # 2D: 单帧
            frames = [data]
        else:
            print(f"[WARN] Unsupported data shape: {data.shape}")
            return None
        
        print(f"[INFO] Extracted {len(frames)} frames from NIfTI")
        return frames
        
    except Exception as e:
        print(f"[WARN] Failed to extract frames from {nifti_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_nifti_voting(nifti_path: str, max_frames: int = 10) -> Dict:
    """
    使用多帧投票分类 NIfTI 文件
    
    Args:
        nifti_path: NIfTI 文件路径
        max_frames: 最大处理的帧数（均匀采样）
    
    Returns:
        投票结果
    """
    frames = extract_all_frames(nifti_path)
    if not frames:
        return {
            'view_type': 'unknown',
            'view_class': 'Random',
            'view_class_cn': '非标准视图',
            'confidence': 0.0,
            'is_reliable': False,
            'error': 'Failed to extract frames'
        }
    
    # 如果帧数太多，均匀采样
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in indices]
        print(f"[INFO] Sampled {len(frames)} frames from {len(frames)} total")
    
    # 对所有帧进行分类（原始 + 翻转）
    all_probs = []
    flip_probs = []
    
    for i, frame in enumerate(frames):
        # 原始方向
        result = classify_single_frame(frame, use_flip=False)
        all_probs.append(result['probs'])
        
        # 翻转方向
        result_flip = classify_single_frame(frame, use_flip=True)
        flip_probs.append(result_flip['probs'])
    
    # 平均概率
    avg_probs = np.mean(all_probs, axis=0)
    avg_flip_probs = np.mean(flip_probs, axis=0)
    
    # 选择置信度更高的方向
    orig_conf = float(np.max(avg_probs))
    flip_conf = float(np.max(avg_flip_probs))
    
    if flip_conf > orig_conf:
        print(f"[INFO] Using FLIPPED orientation (conf: {flip_conf:.2%} vs {orig_conf:.2%})")
        final_probs = avg_flip_probs
        is_flipped = True
    else:
        print(f"[INFO] Using original orientation (conf: {orig_conf:.2%})")
        final_probs = avg_probs
        is_flipped = False
    
    # 最终预测
    pred_idx = int(np.argmax(final_probs))
    confidence = float(final_probs[pred_idx])
    
    view_class = ALL_CLASS_NAMES[pred_idx]
    view_type = VIEW_TYPE_MAPPING.get(view_class, 'unknown')
    
    # 构建概率字典
    all_probs_dict = {
        ALL_CLASS_NAMES[i]: round(float(final_probs[i]), 4)
        for i in range(len(ALL_CLASS_NAMES))
    }
    
    # 判断可靠性
    is_reliable = confidence >= CONFIDENCE_THRESHOLD
    
    # 输出结果
    filename = os.path.basename(nifti_path)
    class_cn = CLASS_NAMES_CN.get(view_class, view_class)
    reliable_str = "✓" if is_reliable else "✗"
    
    print(f"[CLASSIFY-V2] {filename:30s} -> {view_class:6s} ({class_cn:10s}) | "
          f"置信度: {confidence:.2%} [{reliable_str}] | 帧数: {len(frames)} | 翻转: {is_flipped}")
    
    return {
        'view_type': view_type,
        'view_class': view_class,
        'view_class_cn': class_cn,
        'confidence': round(confidence, 4),
        'all_probabilities': all_probs_dict,
        'is_reliable': is_reliable,
        'threshold': CONFIDENCE_THRESHOLD,
        'n_frames': len(frames),
        'is_flipped': is_flipped,
        'flip_confidence': round(flip_conf, 4) if is_flipped else None
    }


def classify_view_v2(
    image_path: str,
    device: Optional[str] = None,
    threshold: float = CONFIDENCE_THRESHOLD
) -> Dict:
    """
    改进版视图分类（支持多帧投票）
    
    Args:
        image_path: 图像路径
        device: 推理设备
        threshold: 置信度阈值
    
    Returns:
        分类结果
    """
    global _classifier_model, _classifier_device
    
    # 初始化分类器
    if not init_classifier(device):
        return {
            'view_type': 'unknown',
            'view_class': 'Random',
            'view_class_cn': '非标准视图',
            'confidence': 0.0,
            'is_reliable': False,
            'error': 'Classifier not available'
        }
    
    try:
        # 判断文件类型
        filename_lower = image_path.lower()
        
        if filename_lower.endswith('.nii') or filename_lower.endswith('.nii.gz'):
            # NIfTI 文件：使用多帧投票
            return classify_nifti_voting(image_path)
        
        else:
            # 普通图像：单帧分类（也尝试翻转）
            pil_img = load_image(image_path)
            img_array = np.array(pil_img.convert('L'))
            
            # 原始方向
            result_orig = classify_single_frame(img_array, use_flip=False)
            # 翻转方向
            result_flip = classify_single_frame(img_array, use_flip=True)
            
            # 选择更好的方向
            if result_flip['confidence'] > result_orig['confidence']:
                final_result = result_flip
                is_flipped = True
            else:
                final_result = result_orig
                is_flipped = False
            
            pred_idx = final_result['pred_idx']
            confidence = final_result['confidence']
            probs = final_result['probs']
            
            view_class = ALL_CLASS_NAMES[pred_idx]
            view_type = VIEW_TYPE_MAPPING.get(view_class, 'unknown')
            
            all_probs_dict = {
                ALL_CLASS_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(ALL_CLASS_NAMES))
            }
            
            is_reliable = confidence >= threshold
            
            filename = os.path.basename(image_path)
            class_cn = CLASS_NAMES_CN.get(view_class, view_class)
            reliable_str = "✓" if is_reliable else "✗"
            
            print(f"[CLASSIFY-V2] {filename:30s} -> {view_class:6s} ({class_cn:10s}) | "
                  f"置信度: {confidence:.2%} [{reliable_str}] | 翻转: {is_flipped}")
            
            return {
                'view_type': view_type,
                'view_class': view_class,
                'view_class_cn': class_cn,
                'confidence': round(confidence, 4),
                'all_probabilities': all_probs_dict,
                'is_reliable': is_reliable,
                'threshold': threshold,
                'is_flipped': is_flipped
            }
    
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'view_type': 'unknown',
            'view_class': 'Random',
            'view_class_cn': '非标准视图',
            'confidence': 0.0,
            'is_reliable': False,
            'error': str(e)
        }


# 保持向后兼容的别名
classify_view = classify_view_v2
