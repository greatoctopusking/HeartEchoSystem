
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def save_overlay(image_path: str, mask_path: str, out_png: str, frame_idx: int = None):
    """
    image_path: 原始输入 (H, W, T) 或 (H, W) 的 nii/nii.gz
    mask_path : 预测 mask（通常是 2D (H, W)，也可能是 3D）
    out_png   : 输出 png 路径
    frame_idx : 如果原图是 (H,W,T)，指定用哪一帧做底图；不传就用中间帧
    """

    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # --- 准备底图 base ---
    if img.ndim == 3:
        T = img.shape[2]
        if frame_idx is None:
            frame_idx = T // 2
        base = img[:, :, frame_idx]
    elif img.ndim == 2:
        base = img
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}, shape={img.shape}")

    # --- 准备 mask2d ---
    if mask.ndim == 2:
        mask2d = mask
    elif mask.ndim == 3:
        # 取中间 slice
        z = mask.shape[2] // 2
        mask2d = mask[:, :, z]
    else:
        raise ValueError(f"Unsupported mask ndim: {mask.ndim}, shape={mask.shape}")

    # 二值化（你也可以根据类别做不同颜色）
    mask_bin = (mask2d > 0)

    # --- 绘制 overlay ---
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(base, cmap="gray")
    # alpha 叠加
    plt.imshow(mask_bin, alpha=0.35)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

