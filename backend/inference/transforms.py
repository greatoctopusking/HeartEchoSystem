"""图像预处理（独立模块）"""
import torch
from torchvision import transforms
from PIL import Image
from typing import Union
import numpy as np

try:
    import nibabel as nib
    NIFTI_AVAILABLE = True
except ImportError:
    NIFTI_AVAILABLE = False

from .constants import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    获取验证/推理时的预处理
    
    Args:
        image_size: 目标图像尺寸
        
    Returns:
        验证数据的 transforms 组合
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image(image_path: str) -> Image.Image:
    """
    加载图像并转换为RGB
    
    Args:
        image_path: 图像路径
        
    Returns:
        PIL Image (RGB模式)
    """
    ext = image_path.lower().split('.')[-1]
    
    if ext in ['nii', 'gz', 'nii.gz']:
        return load_nifti_image(image_path)
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def load_nifti_image(file_path: str) -> Image.Image:
    """
    加载NIfTI图像并转换为PIL Image
    
    Args:
        file_path: NIfTI文件路径
        
    Returns:
        PIL Image (RGB模式)
    """
    if not NIFTI_AVAILABLE:
        raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")
    
    img = nib.load(file_path)
    data = img.get_fdata()
    
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    
    if len(data.shape) == 3:
        if data.shape[2] > 1:
            slice_idx = data.shape[2] // 2
            data = data[:, :, slice_idx]
        else:
            data = data[:, :, 0]
    
    data = np.squeeze(data)
    
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
    data = data.astype(np.uint8)
    
    pil_img = Image.fromarray(data, mode='L')
    return pil_img.convert('RGB')


def preprocess_image(
    image_path: str,
    transform: transforms.Compose = None,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """
    预处理单张图像
    
    Args:
        image_path: 图像路径
        transform: 自定义transform（可选）
        image_size: 图像尺寸
        
    Returns:
        预处理后的 tensor (1, C, H, W)
    """
    if transform is None:
        transform = get_val_transforms(image_size)
    
    img = load_image(image_path)
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    return tensor


def get_supported_extensions() -> list:
    """获取支持的图像扩展名"""
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if NIFTI_AVAILABLE:
        exts.extend(['.nii', '.nii.gz', '.gz'])
    return exts


def is_supported_image(file_path: str) -> bool:
    """检查文件是否为支持的图像格式"""
    ext = '.' + file_path.lower().split('.')[-1]
    return ext in get_supported_extensions()