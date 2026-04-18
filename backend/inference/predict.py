"""推理模块核心功能（独立模块）"""
import torch
import os
import time
import json
from typing import List, Dict, Optional, Union
from pathlib import Path

from .classifier import CardiacClassifier, load_model
from .transforms import get_val_transforms, preprocess_image, is_supported_image
from .constants import ALL_CLASS_NAMES, NUM_CLASSES, DEFAULT_IMAGE_SIZE, get_class_names


def predict_single(
    model: CardiacClassifier,
    image_path: str,
    transform=None,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    单图推理
    
    Args:
        model: 加载好的模型
        image_path: 图像路径
        transform: 图像预处理 transform
        device: 推理设备
        class_names: 类别名称列表（根据任务类型自动确定）
        
    Returns:
        预测结果字典
    """
    if transform is None:
        transform = get_val_transforms()
    
    if class_names is None:
        class_names = ALL_CLASS_NAMES
    
    tensor = preprocess_image(image_path, transform)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    prob_dict = {
        class_names[i]: round(probs[0, i].item(), 4)
        for i in range(len(class_names))
    }
    
    return {
        'image_path': image_path,
        'predicted_class': class_names[pred_idx],
        'class_index': pred_idx,
        'confidence': round(confidence, 4),
        'all_probabilities': prob_dict
    }


def predict_batch(
    model: CardiacClassifier,
    image_paths: List[str],
    transform=None,
    device: str = 'cuda',
    batch_size: int = 32,
    show_progress: bool = True,
    class_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    批量推理
    
    Args:
        model: 加载好的模型
        image_paths: 图像路径列表
        transform: 图像预处理 transform
        device: 推理设备
        batch_size: 批大小
        show_progress: 是否显示进度条
        class_names: 类别名称列表
        
    Returns:
        预测结果列表
    """
    if transform is None:
        transform = get_val_transforms()
    
    if class_names is None:
        class_names = ALL_CLASS_NAMES
    
    results = []
    total = len(image_paths)
    
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        batch_tensors = []
        for path in batch_paths:
            tensor = preprocess_image(path, transform)
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        batch_tensor = batch_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_indices = probs.argmax(dim=1)
            confidences = probs.max(dim=1).values
        
        for j, path in enumerate(batch_paths):
            pred_idx = pred_indices[j].item()
            confidence = confidences[j].item()
            
            prob_dict = {
                class_names[k]: round(probs[j, k].item(), 4)
                for k in range(len(class_names))
            }
            
            results.append({
                'image_path': path,
                'predicted_class': class_names[pred_idx],
                'class_index': pred_idx,
                'confidence': round(confidence, 4),
                'all_probabilities': prob_dict
            })
        
        if show_progress:
            progress = min(i + batch_size, total)
            print(f"\rProcessing: {progress}/{total}", end='', flush=True)
    
    if show_progress:
        print()
    
    return results


def predict_directory(
    model: CardiacClassifier,
    image_dir: str,
    output_path: Optional[str] = None,
    transform=None,
    device: str = 'cuda',
    batch_size: int = 32,
    recursive: bool = True,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    目录批量推理
    
    Args:
        model: 加载好的模型
        image_dir: 图像目录路径
        output_path: 输出JSON文件路径（可选）
        transform: 图像预处理 transform
        device: 推理设备
        batch_size: 批大小
        recursive: 是否递归搜索子目录
        class_names: 类别名称列表
        
    Returns:
        完整结果字典
    """
    if class_names is None:
        class_names = ALL_CLASS_NAMES
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.nii', '.gz', '.nii.gz'}
    
    image_paths = []
    dir_path = Path(image_dir)
    
    if recursive:
        for ext in image_extensions:
            image_paths.extend(dir_path.rglob(f'*{ext}'))
            image_paths.extend(dir_path.rglob(f'*{ext.upper()}'))
    else:
        for ext in image_extensions:
            image_paths.extend(dir_path.glob(f'*{ext}'))
            image_paths.extend(dir_path.glob(f'*{ext.upper()}'))
    
    image_paths = [str(p) for p in image_paths]
    image_paths.sort()
    
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    start_time = time.time()
    results = predict_batch(
        model=model,
        image_paths=image_paths,
        transform=transform,
        device=device,
        batch_size=batch_size,
        show_progress=True,
        class_names=class_names
    )
    inference_time = time.time() - start_time
    
    class_counts = {}
    for r in results:
        cls = r['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    output = {
        'model_checkpoint': None,
        'total_images': len(results),
        'inference_time_seconds': round(inference_time, 2),
        'class_distribution': class_counts,
        'results': results
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return output


def predict_from_path(
    checkpoint_path: str,
    image_path: str,
    num_classes: int = NUM_CLASSES,
    device: str = 'cuda',
    task_type: str = 'multi_class'
) -> Dict:
    """
    一键推理 - 从图像路径直接得到结果
    
    Args:
        checkpoint_path: 模型权重路径
        image_path: 图像路径
        num_classes: 分类类别数
        device: 推理设备
        task_type: 任务类型 'multi_class' 或 'binary'
        
    Returns:
        预测结果字典
    """
    class_names = get_class_names(task_type)
    
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, num_classes=num_classes, device=device)
    
    print(f"Predicting {image_path}...")
    result = predict_single(model, image_path, device=device, class_names=class_names)
    
    return result