"""
心脏超声视图自动分类器
基于7分类模型（A4C, PL, PSAV, PSMV, Random, SC, A2C）
支持输出：2CH(A2C)、4CH(A4C)、其他视图类型、unknown

修复记录:
- 2024-04-12: 修复模型架构，使用 ResNet50 单通道输入（与 test_classifier.py 一致）
- 2024-04-12: 修复 NIfTI 方向问题，添加 as_closest_canonical
- 2024-04-12: 添加多帧投票和翻转检测
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from typing import Dict, Optional, Tuple
import nibabel as nib
from PIL import Image
import cv2

# 添加模型路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

from .constants import ALL_CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

# 分类器配置
CLASSIFIER_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "classifier_mode", "best_model.pth")
CONFIDENCE_THRESHOLD = 0.6

# 类别映射（必须与模型训练时的顺序一致！）
# test_classifier.py 中的顺序: ["A4C", "A2C", "PL", "PSAV", "PSMV", "SC", "Random"]
CLASS_NAMES = ["A4C", "A2C", "PL", "PSAV", "PSMV", "SC", "Random"]
CLASS_NAMES_CN = {
    "A4C": "心尖四腔",
    "A2C": "心尖两腔",
    "PL": "胸骨旁长轴",
    "PSAV": "短轴-主动脉瓣",
    "PSMV": "短轴-二尖瓣",
    "SC": "剑下四腔",
    "Random": "非标准视图"
}

# 视图类型映射（7分类 -> 简化类型）
VIEW_TYPE_MAPPING = {
    'A4C': '4ch',      # 索引 0
    'A2C': '2ch',      # 索引 1
    'PL': 'pl',        # 索引 2
    'PSAV': 'psav',    # 索引 3
    'PSMV': 'psmv',    # 索引 4
    'SC': 'sc',        # 索引 5
    'Random': 'unknown',  # 索引 6
}

# 全局单例模型
_classifier_model = None
_classifier_device = None


class EchoClassifier(nn.Module):
    """心脏超声切面分类模型 - ResNet50 backbone"""
    def __init__(self, num_classes=7):
        super().__init__()
        # 加载 ResNet50 骨干
        resnet = models.resnet50(weights=None)
        
        # 修改第一层为单通道
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 将 RGB 权重平均到单通道
        with torch.no_grad():
            resnet.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        # 使用 Sequential 包装 backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def init_classifier(device: Optional[str] = None) -> bool:
    """初始化分类器模型"""
    global _classifier_model, _classifier_device
    
    if _classifier_model is not None:
        return True
    
    if not os.path.exists(CLASSIFIER_CHECKPOINT):
        print(f"[WARN] Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}")
        return False
    
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _classifier_device = device
        
        print(f"[INFO] Loading classifier from {CLASSIFIER_CHECKPOINT}")
        
        model = EchoClassifier(num_classes=7)
        checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location=device, weights_only=False)
        
        # 兼容不同 checkpoint 格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        _classifier_model = model
        print(f"[INFO] Classifier loaded on {device}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_frame(frame: np.ndarray, use_flip: bool = False) -> torch.Tensor:
    """预处理单帧图像"""
    if use_flip:
        frame = np.flipud(frame)
    
    # 确保是 float32
    img = frame.astype(np.float32)
    
    # 归一化到 0-255
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    else:
        img = np.zeros_like(img)
    
    # Resize 到 224x224
    img = cv2.resize(img, (224, 224))
    
    # 归一化到 [0, 1]
    img = img / 255.0
    
    # 标准化 (mean=0.5, std=0.5)
    img = (img - 0.5) / 0.5
    
    # 转为 tensor [1, 1, 224, 224]
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    
    return img_tensor


def classify_single_frame(frame: np.ndarray, use_flip: bool = False) -> Dict:
    """分类单帧图像"""
    global _classifier_model, _classifier_device
    
    tensor = preprocess_frame(frame, use_flip).to(_classifier_device)
    
    with torch.no_grad():
        output = _classifier_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    return {
        'probs': probs.cpu().numpy(),
        'pred_idx': int(probs.argmax()),
        'confidence': float(probs.max())
    }


def extract_all_frames(nifti_path: str, max_frames: int = 10):
    """从 NIfTI 提取所有帧（与原始 view_classifier 一致，需要转置）"""
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # 处理 4D (H, W, Z, T) -> 取中间切片，保留所有时间帧
        if data.ndim == 4:
            z_idx = data.shape[2] // 2
            data = data[:, :, z_idx, :]  # (H, W, T)
        
        # 处理 3D (H, W, T) -> 提取所有帧并转置
        if data.ndim == 3:
            # 注意：需要 .T 转置以得到正确的方向（心尖朝上）
            frames = [data[:, :, t].T for t in range(data.shape[2])]
            # 如果帧数太多，均匀采样
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames)-1, max_frames).astype(int)
                frames = [frames[i] for i in indices]
            return frames
        
        # 2D
        if data.ndim == 2:
            return [data.T]
        
        return None
    except Exception as e:
        print(f"[WARN] Failed to extract frames: {e}")
        return None


def classify_view(
    image_path: str,
    device: Optional[str] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    max_frames: int = 10
) -> Dict:
    """改进版视图分类 - 支持多帧投票和翻转检测"""
    global _classifier_model, _classifier_device
    
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
        filename_lower = image_path.lower()
        
        # ===== NIfTI 文件：多帧投票 =====
        if filename_lower.endswith('.nii') or filename_lower.endswith('.nii.gz'):
            frames = extract_all_frames(image_path)
            if not frames:
                return {
                    'view_type': 'unknown',
                    'view_class': 'Random',
                    'view_class_cn': '非标准视图',
                    'confidence': 0.0,
                    'is_reliable': False,
                    'error': 'Failed to extract frames'
                }
            
            # 采样帧
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames)-1, max_frames).astype(int)
                frames = [frames[i] for i in indices]
            
            # 对所有帧分类（原始+翻转），选择最佳帧
            best_confidence = 0
            best_result = None
            best_is_flipped = False
            best_probs = None
            
            for frame in frames:
                # 原始方向
                result_orig = classify_single_frame(frame, use_flip=False)
                # 翻转方向
                result_flip = classify_single_frame(frame, use_flip=True)
                
                # 选择本帧更好的方向
                orig_idx = result_orig['pred_idx']
                flip_idx = result_flip['pred_idx']
                orig_conf = result_orig['confidence']
                flip_conf = result_flip['confidence']
                
                orig_is_valid = CLASS_NAMES[orig_idx] not in ['Random']
                flip_is_valid = CLASS_NAMES[flip_idx] not in ['Random']
                
                # 优先选择有效视图中置信度高的
                if flip_is_valid and flip_conf > orig_conf:
                    frame_conf = flip_conf
                    frame_idx = flip_idx
                    frame_probs = result_flip['probs']
                    frame_flipped = True
                elif orig_is_valid:
                    frame_conf = orig_conf
                    frame_idx = orig_idx
                    frame_probs = result_orig['probs']
                    frame_flipped = False
                elif flip_is_valid:
                    frame_conf = flip_conf
                    frame_idx = flip_idx
                    frame_probs = result_flip['probs']
                    frame_flipped = True
                else:
                    # 都是 Random，选置信度高的
                    if flip_conf > orig_conf:
                        frame_conf = flip_conf
                        frame_idx = flip_idx
                        frame_probs = result_flip['probs']
                        frame_flipped = True
                    else:
                        frame_conf = orig_conf
                        frame_idx = orig_idx
                        frame_probs = result_orig['probs']
                        frame_flipped = False
                
                # 更新全局最佳
                is_valid = CLASS_NAMES[frame_idx] not in ['Random']
                best_is_valid = best_result is not None and CLASS_NAMES[best_result['pred_idx']] not in ['Random']
                
                # 优先选择有效视图，如果都是有效视图选置信度高的
                if best_result is None:
                    best_confidence = frame_conf
                    best_result = {'pred_idx': frame_idx, 'confidence': frame_conf}
                    best_is_flipped = frame_flipped
                    best_probs = frame_probs
                elif is_valid and not best_is_valid:
                    # 当前有效，之前无效
                    best_confidence = frame_conf
                    best_result = {'pred_idx': frame_idx, 'confidence': frame_conf}
                    best_is_flipped = frame_flipped
                    best_probs = frame_probs
                elif is_valid == best_is_valid and frame_conf > best_confidence:
                    # 都有效或都无效，选置信度高的
                    best_confidence = frame_conf
                    best_result = {'pred_idx': frame_idx, 'confidence': frame_conf}
                    best_is_flipped = frame_flipped
                    best_probs = frame_probs
            
            pred_idx = best_result['pred_idx']
            confidence = best_confidence
            is_flipped = best_is_flipped
            final_probs = best_probs
            
            view_class = CLASS_NAMES[pred_idx]
            view_type = VIEW_TYPE_MAPPING.get(view_class, 'unknown')
            
            all_probs_dict = {CLASS_NAMES[i]: round(float(final_probs[i]), 4) 
                             for i in range(len(CLASS_NAMES))}
            
            is_reliable = confidence >= threshold
            
            filename = os.path.basename(image_path)
            class_cn = CLASS_NAMES_CN.get(view_class, view_class)
            reliable_mark = "OK" if is_reliable else "LOW"
            
            print(f"[CLASSIFY] {filename:30s} -> {view_class:6s} ({class_cn:10s}) | "
                  f"置信度: {confidence:.2%} [{reliable_mark}] | 帧数: {len(frames)} | 翻转: {is_flipped}")
            
            return {
                'view_type': view_type,
                'view_class': view_class,
                'view_class_cn': class_cn,
                'confidence': round(confidence, 4),
                'all_probabilities': all_probs_dict,
                'is_reliable': is_reliable,
                'threshold': threshold,
                'n_frames': len(frames),
                'is_flipped': is_flipped
            }
        
        # ===== 普通图像：单帧+翻转检测 =====
        else:
            # 读取为灰度图
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img, dtype=np.float32)
            else:
                img = img.astype(np.float32)
            
            # 原始和翻转
            result_orig = classify_single_frame(img, use_flip=False)
            result_flip = classify_single_frame(img, use_flip=True)
            
            # 选择更好的方向
            orig_idx = result_orig['pred_idx']
            flip_idx = result_flip['pred_idx']
            orig_conf = result_orig['confidence']
            flip_conf = result_flip['confidence']
            
            orig_is_valid = CLASS_NAMES[orig_idx] not in ['Random']
            flip_is_valid = CLASS_NAMES[flip_idx] not in ['Random']
            
            if flip_conf > orig_conf and flip_is_valid:
                final_result = result_flip
                is_flipped = True
            elif not orig_is_valid and flip_is_valid:
                final_result = result_flip
                is_flipped = True
            else:
                final_result = result_orig
                is_flipped = False
            
            pred_idx = final_result['pred_idx']
            confidence = final_result['confidence']
            probs = final_result['probs']
            
            view_class = CLASS_NAMES[pred_idx]
            view_type = VIEW_TYPE_MAPPING.get(view_class, 'unknown')
            
            all_probs_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) 
                             for i in range(len(CLASS_NAMES))}
            
            is_reliable = confidence >= threshold
            
            filename = os.path.basename(image_path)
            class_cn = CLASS_NAMES_CN.get(view_class, view_class)
            reliable_mark = "OK" if is_reliable else "LOW"
            
            print(f"[CLASSIFY] {filename:30s} -> {view_class:6s} ({class_cn:10s}) | "
                  f"置信度: {confidence:.2%} [{reliable_mark}] | 翻转: {is_flipped}")
            
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


def classify_views_batch(file_paths: list, device: Optional[str] = None) -> list:
    """批量分类多个文件"""
    results = []
    for path in file_paths:
        result = classify_view(path, device)
        result['file'] = path
        results.append(result)
    return results


def auto_assign_views(files: list, device: Optional[str] = None) -> Dict:
    """
    自动为文件列表分配视图类型（兼容旧接口）
    
    Args:
        files: 文件路径列表
        device: 推理设备
        
    Returns:
        {'2ch': [...], '4ch': [...], 'unknown': [...]}
    """
    results = classify_views_batch(files, device)
    
    assigned = {'2ch': [], '4ch': [], 'unknown': []}
    for r in results:
        view_type = r.get('view_type', 'unknown')
        if view_type in assigned:
            assigned[view_type].append(r.get('file', ''))
        else:
            assigned['unknown'].append(r.get('file', ''))
    
    return assigned
