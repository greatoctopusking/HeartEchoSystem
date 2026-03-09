"""
批量调试脚本：可视化多个病人的心尖和瓣环中点坐标
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

# 定义要处理的病人和文件
base_path = r"D:\SRTP_Project__DeepLearning\project\Resources\database_nifti"
output_dir = r"D:\SRTP_Project__DeepLearning\HeartEchoSystem10\HeartEchoSystem\backend\debug_output"

# 文件列表：(病人ID, 视图, 帧类型)
files_to_process = [
    # patient0002
    ("patient0002", "2CH", "ED"),
    ("patient0002", "2CH", "ES"),
    ("patient0002", "4CH", "ED"),
    ("patient0002", "4CH", "ES"),
    # patient0003
    ("patient0003", "2CH", "ED"),
    ("patient0003", "2CH", "ES"),
    ("patient0003", "4CH", "ED"),
    ("patient0003", "4CH", "ES"),
    # patient0004
    ("patient0004", "2CH", "ED"),
    ("patient0004", "2CH", "ES"),
    ("patient0004", "4CH", "ED"),
    ("patient0004", "4CH", "ES"),
]

# ============ 新算法：基于边缘和极坐标 ============
def annulus_points_from_polar(mask, spacing):
    """
    新算法：
    1. 找到质心位置
    2. 提取LV的边缘
    3. 以质心为中心作坐标系，每个边缘点有(theta, r)
    4. r是theta的函数：r(theta)
    5. 找到3个峰值
    
    返回: (theta_array, r_array, edge_pts, centroid_mm, top3_peaks_refined, r_smoothed)
    """
    try:
        from scipy.ndimage import binary_erosion, binary_dilation
    except Exception:
        binary_erosion = None
        binary_dilation = None
    
    try:
        import cv2
    except Exception:
        cv2 = None
    
    if cv2 is None:
        print("    [WARNING] cv2 not available, using morphological edge")
    
    arr = np.rint(np.asarray(mask)).astype(np.int16)
    lv = (arr == 1)  # Cavity
    wall = (arr == 2)  # Wall
    lv_all = lv | wall  # 腔体和壁
    
    if np.count_nonzero(lv) < 30:
        return None, None, None, None, None, None
    
    dx, dy = float(spacing[0]), float(spacing[1])
    
    # 1. 找质心
    cav_coords = np.column_stack(np.where(lv))
    cav_pts_mm = cav_coords[:, [1, 0]] * [dx, dy]
    centroid_mm = cav_pts_mm.mean(axis=0)
    
    print(f"    [Step 1] Centroid: ({centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}) mm")
    
    # 2. 提取LV边缘
    if cv2 is not None:
        lv_uint8 = lv.astype(np.uint8)
        contours, _ = cv2.findContours(lv_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            edge_pts = largest_contour[:, 0, :].astype(float)
            edge_pts_mm = edge_pts * [dx, dy]
            print(f"    [Step 2] Edge points: {len(edge_pts)} pixels")
        else:
            print("    [ERROR] No contour found")
            return None, None, None, None, None, None
    else:
        if binary_erosion is not None:
            eroded = binary_erosion(lv, iterations=2)
            edge = lv ^ eroded
        else:
            edge = lv
        
        edge_coords = np.column_stack(np.where(edge))
        edge_pts_mm = edge_coords[:, [1, 0]] * [dx, dy]
        print(f"    [Step 2] Edge points (morphological): {len(edge_pts_mm)} pixels")
    
    # 3. 极坐标转换
    centered_pts = edge_pts_mm - centroid_mm
    r = np.sqrt(centered_pts[:, 0]**2 + centered_pts[:, 1]**2)
    theta = np.arctan2(centered_pts[:, 1], centered_pts[:, 0])
    theta = np.mod(theta, 2 * np.pi)
    
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    r_sorted = r[sort_idx]
    
    # 4. 移动平均平滑
    window_size = 15
    r_smoothed = np.convolve(r_sorted, np.ones(window_size)/window_size, mode='same')
    print(f"    [Step 3.5] Smoothed with moving average (window={window_size})")
    
    # 5. 找峰值
    def find_peaks_numpy(data, min_distance=20, threshold_ratio=0.3):
        peaks = []
        threshold = threshold_ratio * (data.max() - data.min()) + data.min()
        for i in range(min_distance, len(data) - min_distance):
            left = max(0, i - min_distance)
            right = min(len(data), i + min_distance)
            if data[i] >= threshold and data[i] >= max(data[left:right]):
                peaks.append(i)
        return peaks
    
    all_peaks = find_peaks_numpy(r_sorted, min_distance=10, threshold_ratio=0.08)
    
    if len(all_peaks) >= 3:
        peak_heights = r_sorted[all_peaks]
        top3_idx = np.argsort(peak_heights)[-3:][::-1]
        top3_peaks = [all_peaks[i] for i in top3_idx]
    else:
        top3_peaks = all_peaks
    
    # 抛物线插值精确定位
    refined_peaks = []
    for peak_idx in top3_peaks:
        if peak_idx > 0 and peak_idx < len(r_sorted) - 1:
            y0, y1, y2 = r_sorted[peak_idx-1], r_sorted[peak_idx], r_sorted[peak_idx+1]
            a = (y0 + y2 - 2*y1) / 2
            if abs(a) > 1e-10:
                delta = 0.5 * (y0 - y2) / a
                refined_idx = peak_idx + delta
                refined_peaks.append(refined_idx)
            else:
                refined_peaks.append(peak_idx)
        else:
            refined_peaks.append(peak_idx)
    
    top3_peaks_refined = refined_peaks
    
    print(f"    [Step 4] Found {len(top3_peaks_refined)} peaks")
    for i, p_idx in enumerate(top3_peaks_refined):
        idx = int(p_idx)
        print(f"      Peak {i+1}: theta={np.degrees(theta_sorted[idx]):.1f}deg, r={r_sorted[idx]:.1f}mm")
    
    # 6. 计算峰值之间的距离，找出瓣环的两个点
    # 瓣环的两个点应该是距离最近的两个峰值
    annulus_left_mm = None
    annulus_right_mm = None
    
    if len(top3_peaks_refined) >= 2:
        peak_points = []
        peak_indices = []
        for i, peak_idx in enumerate(top3_peaks_refined):
            idx = int(peak_idx)
            t = theta_sorted[idx]
            r_val = r_sorted[idx]
            x = centroid_mm[0] + r_val * np.cos(t)
            y = centroid_mm[1] + r_val * np.sin(t)
            peak_points.append(np.array([x, y]))
            peak_indices.append(idx)
        
        # 找距离最近的两个峰值作为瓣环点
        min_dist = float('inf')
        min_pair = (0, 1)
        for i in range(len(peak_points)):
            for j in range(i+1, len(peak_points)):
                dist = np.linalg.norm(peak_points[i] - peak_points[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
        
        # 这两个就是瓣环点
        annulus_left_mm = peak_points[min_pair[0]]
        annulus_right_mm = peak_points[min_pair[1]]
        
        print(f"    [Step 6] Annulus points: Peak {min_pair[0]+1} and Peak {min_pair[1]+1}, dist={min_dist:.1f}mm")
        
        print(f"    [Step 5] Distance between peaks:")
        for i in range(len(peak_points)):
            for j in range(i+1, len(peak_points)):
                dist = np.linalg.norm(peak_points[i] - peak_points[j])
                t1 = np.degrees(theta_sorted[int(top3_peaks_refined[i])])
                t2 = np.degrees(theta_sorted[int(top3_peaks_refined[j])])
                print(f"      Peak {i+1} (theta={t1:.1f}deg) to Peak {j+1} (theta={t2:.1f}deg): {dist:.1f} mm")
    
    return theta_sorted, r_sorted, edge_pts_mm, centroid_mm, top3_peaks_refined, r_smoothed, annulus_left_mm, annulus_right_mm


def axis_and_points_from_mask_annulus_apex(mask, spacing):
    """使用新算法找到长轴、心尖和瓣环中点"""
    result = annulus_points_from_polar(mask, spacing)
    
    if result[0] is None:
        return None, None, None, None, None
    
    theta_sorted, r_sorted, edge_pts_mm, centroid_mm, top3_peaks, r_smoothed, annulus_left_mm, annulus_right_mm = result
    
    # 瓣环中点 = 左右两点的中点
    if annulus_left_mm is not None and annulus_right_mm is not None:
        annulus_mid_mm = 0.5 * (annulus_left_mm + annulus_right_mm)
    else:
        # 兜底：用质心
        annulus_mid_mm = centroid_mm
    
    # 心尖：排除瓣环两个点后，剩下那个就是心尖
    # 先获取瓣环两个峰值在top3_peaks中的索引
    if len(top3_peaks) >= 3 and annulus_left_mm is not None and annulus_right_mm is not None:
        # 计算每个峰值到瓣环左右点的距离，找出不是瓣环的那个
        apex_peak_idx = None
        for i, peak_idx in enumerate(top3_peaks):
            idx = int(peak_idx)
            t = theta_sorted[idx]
            r_val = r_sorted[idx]
            peak_pt = np.array([
                centroid_mm[0] + r_val * np.cos(t),
                centroid_mm[1] + r_val * np.sin(t)
            ])
            # 检查这个点是否接近瓣环点
            dist_to_left = np.linalg.norm(peak_pt - annulus_left_mm)
            dist_to_right = np.linalg.norm(peak_pt - annulus_right_mm)
            if dist_to_left > 10 and dist_to_right > 10:  # 不接近瓣环点
                apex_peak_idx = i
                break
        
        if apex_peak_idx is not None:
            apex_idx = int(top3_peaks[apex_peak_idx])
        else:
            # 兜底：选r最大的
            apex_idx = int(top3_peaks[0])
    else:
        # 兜底
        apex_idx = int(top3_peaks[0]) if len(top3_peaks) > 0 else None
    
    if apex_idx is None:
        return None, None, None, None, None
    
    t = theta_sorted[apex_idx]
    r_val = r_sorted[apex_idx]
    apex_mm = np.array([
        centroid_mm[0] + r_val * np.cos(t),
        centroid_mm[1] + r_val * np.sin(t)
    ])
    
    # 长轴方向
    axis_u = apex_mm - annulus_mid_mm
    nrm = float(np.linalg.norm(axis_u))
    if nrm <= 1e-12:
        return None, None, None, None, None
    axis_u = axis_u / nrm
    
    return axis_u.astype(float), apex_mm, annulus_mid_mm, annulus_left_mm, annulus_right_mm

# ============ 批量处理 ============
results = []

for patient_id, view, frame_type in files_to_process:
    file_name = f"{patient_id}_{view}_{frame_type}_gt.nii.gz"
    file_path = os.path.join(base_path, patient_id, file_name)
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        continue
    
    # 加载图像
    img = nib.load(file_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:2]
    
    # 取第一帧
    if len(data.shape) == 3:
        mask = data[:, :, 0]
    elif len(data.shape) == 4:
        mask = data[:, :, :, 0]
    else:
        mask = data
    
    print(f"  Shape: {mask.shape}, Spacing: {spacing}")
    
    # 检测关键点
    try:
        result = axis_and_points_from_mask_annulus_apex(mask, spacing)
        
        if result[0] is None:
            print(f"  [ERROR] Failed to detect key points")
            continue
        
        axis_u, apex_mm, annulus_mid_mm, annulus_left_mm, annulus_right_mm = result
        
        # 新算法：point_type 已经是 Annulus Mid（使用质心作为参考）
        point_type = "Annulus Mid (Polar)"
        
        # 像素坐标
        apex_px = (apex_mm[0] / spacing[0], apex_mm[1] / spacing[1])
        annulus_px = (annulus_mid_mm[0] / spacing[0], annulus_mid_mm[1] / spacing[1])
        annulus_left_px = (annulus_left_mm[0] / spacing[0], annulus_left_mm[1] / spacing[1]) if annulus_left_mm is not None else None
        annulus_right_px = (annulus_right_mm[0] / spacing[0], annulus_right_mm[1] / spacing[1]) if annulus_right_mm is not None else None
        
        print(f"  Point Type: {point_type}")
        print(f"  Apex (mm): ({apex_mm[0]:.1f}, {apex_mm[1]:.1f})")
        print(f"  Annulus Mid (mm): ({annulus_mid_mm[0]:.1f}, {annulus_mid_mm[1]:.1f})")
        if annulus_left_mm is not None:
            print(f"  Annulus Left (mm): ({annulus_left_mm[0]:.1f}, {annulus_left_mm[1]:.1f})")
        if annulus_right_mm is not None:
            print(f"  Annulus Right (mm): ({annulus_right_mm[0]:.1f}, {annulus_right_mm[1]:.1f})")
        
        # 保存结果
        results.append({
            'patient': patient_id,
            'view': view,
            'frame': frame_type,
            'point_type': point_type,
            'apex_mm': apex_mm,
            'annulus_mm': annulus_mid_mm,
            'annulus_left_mm': annulus_left_mm,
            'annulus_right_mm': annulus_right_mm,
            'axis_u': axis_u,
            'mask': mask,
            'spacing': spacing,
            'apex_px': apex_px,
            'annulus_px': annulus_px,
            'annulus_left_px': annulus_left_px,
            'annulus_right_px': annulus_right_px
        })
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

# ============ 生成汇总表 ============
print(f"\n\n{'='*60}")
print("SUMMARY TABLE")
print(f"{'='*60}")
print(f"{'Patient':<12} {'View':<6} {'Frame':<6} {'Point Type':<12} {'Apex (mm)':<20} {'Annulus (mm)':<20}")
print("-" * 80)
for r in results:
    print(f"{r['patient']:<12} {r['view']:<6} {r['frame']:<6} {r['point_type']:<12} ({r['apex_mm'][0]:.1f}, {r['apex_mm'][1]:.1f})    ({r['annulus_mm'][0]:.1f}, {r['annulus_mm'][1]:.1f})")

# ============ 生成每个病人的图像 ============
print(f"\n\nGenerating images...")

for i, r in enumerate(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mask = r['mask']
    apex_px = r['apex_px']
    annulus_px = r['annulus_px']
    annulus_left_px = r.get('annulus_left_px')
    annulus_right_px = r.get('annulus_right_px')
    apex_mm = r['apex_mm']
    annulus_mm = r['annulus_mm']
    annulus_left_mm = r.get('annulus_left_mm')
    annulus_right_mm = r.get('annulus_right_mm')
    point_type = r['point_type']
    
    # 1. 分割掩码
    ax1 = axes[0]
    im1 = ax1.imshow(mask, cmap='jet', interpolation='nearest')
    ax1.set_title(f'{r["patient"]}_{r["view"]}_{r["frame"]}\nSegmentation')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 叠加显示
    ax2 = axes[1]
    cavity_mask = (mask == 1)
    ax2.imshow(cavity_mask, cmap='gray', alpha=0.5)
    ax2.scatter(apex_px[0], apex_px[1], c='red', s=200, marker='*', label='Apex', edgecolors='black')
    ax2.scatter(annulus_px[0], annulus_px[1], c='blue', s=150, marker='o', label='Annulus Mid', edgecolors='black')
    # 标注瓣环左右点
    if annulus_left_px is not None:
        ax2.scatter(annulus_left_px[0], annulus_left_px[1], c='lime', s=100, marker='s', label='Annulus Left', edgecolors='black')
    if annulus_right_px is not None:
        ax2.scatter(annulus_right_px[0], annulus_right_px[1], c='yellow', s=100, marker='s', label='Annulus Right', edgecolors='black')
    ax2.plot([annulus_px[0], apex_px[0]], [annulus_px[1], apex_px[1]], 'g-', linewidth=2)
    ax2.set_title('Key Points')
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. 特写
    ax3 = axes[2]
    ax3.imshow(mask, cmap='gray')
    ax3.scatter(apex_px[0], apex_px[1], c='red', s=300, marker='*', label='Apex', edgecolors='white', linewidths=2)
    ax3.scatter(annulus_px[0], annulus_px[1], c='cyan', s=200, marker='o', label='Annulus Mid', edgecolors='white', linewidths=2)
    # 标注瓣环左右点
    if annulus_left_px is not None:
        ax3.scatter(annulus_left_px[0], annulus_left_px[1], c='lime', s=200, marker='s', label='Annulus Left', edgecolors='white', linewidths=2)
    if annulus_right_px is not None:
        ax3.scatter(annulus_right_px[0], annulus_right_px[1], c='yellow', s=200, marker='s', label='Annulus Right', edgecolors='white', linewidths=2)
    ax3.plot([annulus_px[0], apex_px[0]], [annulus_px[1], apex_px[1]], 'yellow', linewidth=3)
    # 连接瓣环左右点
    if annulus_left_px is not None and annulus_right_px is not None:
        ax3.plot([annulus_left_px[0], annulus_right_px[0]], [annulus_left_px[1], annulus_right_px[1]], 'white', linewidth=2, linestyle='--')
    ax3.set_title(f'Point Type: {point_type}')
    ax3.legend(loc='upper right', fontsize=8)
    
    # 标注
    ax3.annotate(f'Apex\n({apex_mm[0]:.1f}, {apex_mm[1]:.1f})', 
                xy=apex_px, xytext=(apex_px[0]+20, apex_px[1]-20),
                color='red', fontsize=9, fontweight='bold')
    ax3.annotate(f'Ann Mid\n({annulus_mm[0]:.1f}, {annulus_mm[1]:.1f})', 
                xy=annulus_px, xytext=(annulus_px[0]+20, annulus_px[1]+20),
                color='cyan', fontsize=9, fontweight='bold')
    if annulus_left_mm is not None and annulus_left_px is not None:
        ax3.annotate(f'L\n({annulus_left_mm[0]:.1f}, {annulus_left_mm[1]:.1f})', 
                    xy=annulus_left_px, xytext=(annulus_left_px[0]-40, annulus_left_px[1]-20),
                    color='lime', fontsize=9, fontweight='bold')
    if annulus_right_mm is not None and annulus_right_px is not None:
        ax3.annotate(f'R\n({annulus_right_mm[0]:.1f}, {annulus_right_mm[1]:.1f})', 
                    xy=annulus_right_px, xytext=(annulus_right_px[0]+10, annulus_right_px[1]-20),
                    color='yellow', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'apex_annulus_{r["patient"]}_{r["view"]}_{r["frame"]}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

print(f"\nDone! Total: {len(results)} images generated.")
