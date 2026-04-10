
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# def plot_foolproof_zscore_for_ppt(image_path):
#     # 1. 读取原图
#     img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img_raw is None:
#         print("未找到图片，请检查路径！")
#         return

#     # 2. 提取有效组织（屏蔽黑背景）
#     tissue_pixels = img_raw[img_raw > 15].astype(np.float32)

#     # 3. 将组织特征归一化到基础区间 (0~1)
#     base_norm = (tissue_pixels - tissue_pixels.min()) / (tissue_pixels.max() - tissue_pixels.min() + 1e-8)

#     # 4. 完美的线性模拟（绝对不会产生 255 截断）
#     # 设备A：偏暗，低对比度 (被限制在 40~100)
#     data_dark = base_norm * 60 + 40
#     # 设备B：偏亮，高对比度 (被限制在 120~220)
#     data_bright = base_norm * 100 + 120

#     # 5. 分别计算 Z-score
#     z_dark = (data_dark - np.mean(data_dark)) / (np.std(data_dark) + 1e-8)
#     z_bright = (data_bright - np.mean(data_bright)) / (np.std(data_bright) + 1e-8)

#     # 6. 绘图
#     fig = plt.figure(figsize=(14, 6))
#     plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
#     plt.rcParams['axes.unicode_minus'] = False
#     fig.patch.set_facecolor('#f8f9fa')

#     # --- 左侧：处理前 ---
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.hist(data_dark, bins=80, color='royalblue', alpha=0.7, label='设备A (偏暗/低对比)')
#     ax1.hist(data_bright, bins=80, color='crimson', alpha=0.6, label='设备B (偏亮/高对比)')
#     ax1.set_title("处理前：超声组织灰度分布跨设备差异大", fontsize=15, fontweight='bold')
#     ax1.set_xlabel("像素灰度值 (0-255)")
#     ax1.set_ylabel("组织像素数量")
#     ax1.legend(fontsize=13)
#     ax1.set_xlim(0, 255) # 固定 X 轴范围，视觉对比更强烈
#     ax1.grid(True, linestyle='--', alpha=0.5)

#     # --- 右侧：处理后 ---
#     ax2 = fig.add_subplot(1, 2, 2)
#     # 因为数学上100%重合，画图时调一下透明度和边缘让两层都清晰可见
#     ax2.hist(z_dark, bins=80, color='royalblue', alpha=1.0, label='设备A (Z-score后)')
#     ax2.hist(z_bright, bins=80, color='crimson', alpha=0.5, label='设备B (Z-score后)')
#     ax2.set_title("Z-score 处理后：消除物理差异，分布重合度高", fontsize=15, fontweight='bold')
#     ax2.set_xlabel("标准正态分布值")
#     ax2.legend(fontsize=13)
#     ax2.grid(True, linestyle='--', alpha=0.5)

#     plt.tight_layout()
#     plt.show()

# 替换为你自己的图像路径跑一下

import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_true_perfect_zscore_for_ppt(image_path):
    # 1. 读取原图
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print("未找到图片，请检查路径！")
        return

    # 2. 提取有效组织
    tissue_pixels = img_raw[img_raw > 15].astype(np.float32)

    # 3. 将组织特征归一化到基础区间 (0~1)
    base_norm = (tissue_pixels - tissue_pixels.min()) / (tissue_pixels.max() - tissue_pixels.min() + 1e-8)

    # 4. 完美的线性模拟（绝对不会产生 255 截断）
    data_dark = base_norm * 60 + 40     # 设备A
    data_bright = base_norm * 100 + 120 # 设备B

    # 5. 分别计算 Z-score
    z_dark = (data_dark - np.mean(data_dark)) / (np.std(data_dark) + 1e-8)
    z_bright = (data_bright - np.mean(data_bright)) / (np.std(data_bright) + 1e-8)

    # 6. 绘图
    fig = plt.figure(figsize=(14, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    fig.patch.set_facecolor('#f8f9fa')

    # --- 左侧：处理前 ---
    ax1 = fig.add_subplot(1, 2, 1)
    # 【强制统一原始图像网格】
    raw_bins = np.linspace(0, 255, 100)
    ax1.hist(data_dark, bins=raw_bins, color='royalblue', alpha=0.8, label='设备A (偏暗/低对比)')
    ax1.hist(data_bright, bins=raw_bins, color='crimson', alpha=0.7, label='设备B (偏亮/高对比)')
    ax1.set_title("处理前：灰度分布跨设备差异大（模拟亮度/对比度不同）", fontsize=15, fontweight='bold')
    ax1.set_xlabel("像素灰度值 (0-255)")
    ax1.set_ylabel("组织像素数量")
    ax1.legend(fontsize=13)
    ax1.set_xlim(0, 255)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 右侧：处理后 ---
    ax2 = fig.add_subplot(1, 2, 2)
    # 🌟【终极修复核心】：强制使用绝对一致的 Bin 网格边界，杜绝任何自动计算误差！
    shared_bins = np.linspace(-4, 4, 100)
    
    # 调整透明度，让底部的蓝色和顶部的红色混合成纯正的紫色
    ax2.hist(z_dark, bins=shared_bins, color='royalblue', alpha=0.9, label='设备A (Z-score后)')
    ax2.hist(z_bright, bins=shared_bins, color='crimson', alpha=0.6, label='设备B (Z-score后)')
    
    ax2.set_title("Z-score 处理后：消除物理差异，分布重合", fontsize=15, fontweight='bold')
    ax2.set_xlabel("标准正态分布值")
    ax2.legend(fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# 运行代码
# plot_true_perfect_zscore_for_ppt("你的超声图路径.jpg")
plot_true_perfect_zscore_for_ppt("C:\\Users\\29809\\Desktop\\echo\\frontend\\ui_res\\login_back.png")