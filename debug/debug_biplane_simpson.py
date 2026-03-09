"""
Biplane Simpson 调试脚本
批量处理 patient0002-0004 的 2CH/4CH series 分割图像
输出三张图片：面积曲线、ED/ES标注、Simpson分层线
"""
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding='utf-8')

from config import *
from model_infer import run_inference
from biplane_simpson_clinical import BiplaneSimpsonClinical


BASE_PATH = r"D:\SRTP_Project__DeepLearning\project\Resources\database_nifti"
OUTPUT_DIR = r"D:\SRTP_Project__DeepLearning\HeartEchoSystem10\HeartEchoSystem\backend\debug_simpson_output"

PATIENTS = ["patient0002", "patient0003", "patient0004"]

LABEL_MAP = {0: "background", 1: "LV", 2: "LVWall", 3: "LA"}


def _ensure_int_labels(mask_arr):
    arr = np.asarray(mask_arr)
    if arr.dtype.kind in ("i", "u"):
        return arr
    return np.rint(arr).astype(np.int16)


def draw_simpson_lines(ax, mask, spacing, n_discs=20, band_frac=1.6, min_band_points=3, 
                       axis_u_override=None, apex_mm=None, annulus_mid_mm=None):
    dx, dy = float(spacing[0]), float(spacing[1])
    
    def to_disp(x_pix, y_pix):
        return (y_pix, x_pix)

    m = (_ensure_int_labels(mask) == 1)
    coords = np.column_stack(np.where(m))
    if coords.shape[0] < 30:
        return

    ys = coords[:, 0].astype(float)
    xs = coords[:, 1].astype(float)
    pts = np.column_stack([xs * dx, ys * dy])

    mean_pt = pts.mean(axis=0)
    X = pts - mean_pt
    C = (X.T @ X) / max(1, pts.shape[0] - 1)
    
    if axis_u_override is not None:
        axis_u = np.asarray(axis_u_override, dtype=float)
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    elif apex_mm is not None and annulus_mid_mm is not None:
        axis_u = np.asarray(apex_mm, dtype=float) - np.asarray(annulus_mid_mm, dtype=float)
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    else:
        vals, vecs = np.linalg.eigh(C)
        axis_u = vecs[:, int(np.argmax(vals))]
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)

    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=float)

    if apex_mm is not None and annulus_mid_mm is not None:
        origin_pt = np.asarray(annulus_mid_mm, dtype=float)
        apex_pt = np.asarray(apex_mm, dtype=float)

        if float(np.dot(apex_pt - origin_pt, axis_u)) < 0:
            axis_u = -axis_u
            perp_u = np.array([-axis_u[1], axis_u[0]], dtype=float)

        centered = pts - origin_pt
        t = centered @ axis_u
        s = centered @ perp_u

        L = float(np.dot(apex_pt - origin_pt, axis_u))
        if L <= 1e-6:
            return

        tmin, tmax = 0.0, L
    else:
        origin_pt = mean_pt
        centered = pts - origin_pt
        t = centered @ axis_u
        s = centered @ perp_u

        tmin, tmax = float(np.min(t)), float(np.max(t))
        L = tmax - tmin
        if L <= 1e-6:
            return

    h = L / float(n_discs)
    centers = tmin + (np.arange(n_discs) + 0.5) * h
    band_half = 0.5 * band_frac * h

    if apex_mm is not None and annulus_mid_mm is not None:
        p1 = origin_pt
        p2 = apex_pt
    else:
        p1 = origin_pt + axis_u * tmin
        p2 = origin_pt + axis_u * tmax

    x1_pix, y1_pix = p1[0] / dx, p1[1] / dy
    x2_pix, y2_pix = p2[0] / dx, p2[1] / dy
    X1, Y1 = to_disp(x1_pix, y1_pix)
    X2, Y2 = to_disp(x2_pix, y2_pix)
    ax.plot([X1, X2], [Y1, Y2], color='cyan', linewidth=2)

    for c in centers:
        band = (t >= c - band_half) & (t <= c + band_half)
        if np.count_nonzero(band) < int(min_band_points):
            continue

        smin = float(np.min(s[band]))
        smax = float(np.max(s[band]))
        if smax <= smin:
            continue

        center_pt = origin_pt + axis_u * c

        a = center_pt + perp_u * smin
        b = center_pt + perp_u * smax

        ax_x1_pix, ax_y1_pix = a[0] / dx, a[1] / dy
        ax_x2_pix, ax_y2_pix = b[0] / dx, b[1] / dy

        XA1, YA1 = to_disp(ax_x1_pix, ax_y1_pix)
        XA2, YA2 = to_disp(ax_x2_pix, ax_y2_pix)

        ax.plot([XA1, XA2], [YA1, YA2], color='yellow', linewidth=1.0, alpha=0.75)


def process_patient(patient_id):
    """处理单个病人的数据"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")
    
    file_2ch = os.path.join(BASE_PATH, patient_id, f"{patient_id}_2CH_half_sequence_gt.nii.gz")
    file_4ch = os.path.join(BASE_PATH, patient_id, f"{patient_id}_4CH_half_sequence_gt.nii.gz")
    
    if not os.path.exists(file_2ch):
        print(f"  [SKIP] File not found: {file_2ch}")
        return None
    if not os.path.exists(file_4ch):
        print(f"  [SKIP] File not found: {file_4ch}")
        return None
    
    img2 = nib.load(file_2ch)
    img4 = nib.load(file_4ch)
    
    masks2 = img2.get_fdata()
    masks4 = img4.get_fdata()
    spacing2 = img2.header.get_zooms()[:2]
    spacing4 = img4.header.get_zooms()[:2]
    
    print(f"  Raw data shape: 2CH={masks2.shape}, 4CH={masks4.shape}")
    
    # Data is already (H, W, T), keep as is
    # Convert to list of 2D frames
    masks2_list = [masks2[:, :, t] for t in range(masks2.shape[2])]
    masks4_list = [masks4[:, :, t] for t in range(masks4.shape[2])]
    
    T2 = len(masks2_list)
    T4 = len(masks4_list)
    
    print(f"  Processed: 2CH has {T2} frames, 4CH has {T4} frames")
    
    print(f"  2CH: {masks2.shape}, spacing={spacing2}")
    print(f"  4CH: {masks4.shape}, spacing={spacing4}")
    
    calculator = BiplaneSimpsonClinical(n_discs=20)
    
    result = calculator.compute_ed_es_from_series(
        masks2_list, masks4_list, spacing2, spacing4
    )
    
    ED_index_2ch = int(result['ED_index'])
    ES_index_2ch = int(result['ES_index'])
    ED_index_4ch = int(result.get('ED_index_4ch', 0))
    ES_index_4ch = int(result.get('ES_index_4ch', 0))
    
    print(f"\n  === Results ===")
    print(f"  2CH: ED frame = {ED_index_2ch}, ES frame = {ES_index_2ch}")
    print(f"  4CH: ED frame = {ED_index_4ch}, ES frame = {ES_index_4ch}")
    print(f"  EDV = {result['EDV']:.2f} mL")
    print(f"  ESV = {result['ESV']:.2f} mL")
    print(f"  LVEF = {result['EF']:.2f}%")
    
    return {
        'patient': patient_id,
        'masks2': masks2_list,
        'masks4': masks4_list,
        'spacing2': spacing2,
        'spacing4': spacing4,
        'ED_index_2ch': ED_index_2ch,
        'ES_index_2ch': ES_index_2ch,
        'ED_index_4ch': ED_index_4ch,
        'ES_index_4ch': ES_index_4ch,
        'result': result,
        'T2': T2,
        'T4': T4
    }


def plot_area_curves(results, output_path):
    """图1: 像素面积曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LV Cavity Area Curves (Pixel Count)", fontsize=14, fontweight='bold')
    
    for idx, r in enumerate(results):
        if r is None:
            continue
        
        masks2 = r['masks2']
        masks4 = r['masks4']
        ED_2ch = r['ED_index_2ch']
        ES_2ch = r['ES_index_2ch']
        
        area2 = [np.sum(_ensure_int_labels(masks2[t]) == 1) for t in range(r['T2'])]
        area4 = [np.sum(_ensure_int_labels(masks4[t]) == 1) for t in range(r['T4'])]
        
        ax = axes[idx // 2, idx % 2]
        ax.plot(area2, 'b-', linewidth=2, label='2CH')
        ax.plot(area4, 'r-', linewidth=2, label='4CH')
        
        ax.axvline(ED_2ch, color='green', linestyle='--', linewidth=2, label=f'2CH ED ({ED_2ch})')
        ax.axvline(ES_2ch, color='orange', linestyle='--', linewidth=2, label=f'2CH ES ({ES_2ch})')
        
        ax.set_title(f"{r['patient']}", fontsize=12)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Pixel Count (LV Cavity)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_ed_es_overlay(results, output_path):
    """图2: ED/ES帧分割结果 + 心尖瓣环标注"""
    n_patients = len([r for r in results if r is not None])
    fig, axes = plt.subplots(n_patients, 4, figsize=(16, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("ED/ES Frame Segmentation with Apex/Annulus Markers", fontsize=14, fontweight='bold')
    
    for idx, r in enumerate(results):
        if r is None:
            continue
        
        masks2 = r['masks2']
        masks4 = r['masks4']
        spacing2 = r['spacing2']
        spacing4 = r['spacing4']
        ED_2ch = r['ED_index_2ch']
        ES_2ch = r['ES_index_2ch']
        ED_4ch = r['ED_index_4ch']
        ES_4ch = r['ES_index_4ch']
        result = r['result']
        
        frames = [
            (masks2, ED_2ch, "2CH ED", spacing2, result.get('apex_2ch_ed'), result.get('annulus_mid_2ch_ed')),
            (masks2, ES_2ch, "2CH ES", spacing2, result.get('apex_2ch_es'), result.get('annulus_mid_2ch_es')),
            (masks4, ED_4ch, "4CH ED", spacing4, result.get('apex_4ch_ed'), result.get('annulus_mid_4ch_ed')),
            (masks4, ES_4ch, "4CH ES", spacing4, result.get('apex_4ch_es'), result.get('annulus_mid_4ch_es')),
        ]
        
        for col, (mask_list, frame_idx, title, spacing, apex_mm, annulus_mm) in enumerate(frames):
            ax = axes[idx, col]
            
            mask_frame = mask_list[frame_idx]
            mask_int = _ensure_int_labels(mask_frame)
            
            cav_mask = (mask_int == 1)
            wall_mask = (mask_int == 2)
            
            ax.imshow(np.zeros_like(mask_frame), cmap='gray')
            ax.imshow(np.ma.masked_where(~cav_mask, cav_mask), cmap='Reds', alpha=0.5, origin='lower')
            ax.imshow(np.ma.masked_where(~wall_mask, wall_mask), cmap='Blues', alpha=0.3, origin='lower')
            
            if apex_mm is not None and annulus_mm is not None:
                apex_px = (apex_mm[0] / spacing[0], apex_mm[1] / spacing[1])
                annulus_px = (annulus_mm[0] / spacing[0], annulus_mm[1] / spacing[1])
                
                ax.scatter(apex_px[0], apex_px[1], c='lime', s=200, marker='*', 
                         label='Apex', edgecolors='black', linewidths=1, zorder=10)
                ax.scatter(annulus_px[0], annulus_px[1], c='cyan', s=150, marker='o', 
                         label='Annulus Mid', edgecolors='black', linewidths=1, zorder=10)
                ax.plot([annulus_px[0], apex_px[0]], [annulus_px[1], apex_px[1]], 
                       'yellow', linewidth=2, zorder=9)
                ax.legend(loc='upper right', fontsize=8)
            
            ax.set_title(f"{r['patient']} {title} (frame {frame_idx})", fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_simpson_lines(results, output_path):
    """图3: Simpson分层直径线 + EF结果"""
    n_patients = len([r for r in results if r is not None])
    fig, axes = plt.subplots(n_patients, 4, figsize=(16, 4*n_patients))
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Simpson Biplane Disc Method with LVEF Results", fontsize=14, fontweight='bold')
    
    for idx, r in enumerate(results):
        if r is None:
            continue
        
        masks2 = r['masks2']
        masks4 = r['masks4']
        spacing2 = r['spacing2']
        spacing4 = r['spacing4']
        ED_2ch = r['ED_index_2ch']
        ES_2ch = r['ES_index_2ch']
        ED_4ch = r['ED_index_4ch']
        ES_4ch = r['ES_index_4ch']
        result = r['result']
        
        frames = [
            (masks2, ED_2ch, "2CH ED", spacing2, result.get('axis_u_2ch_ed'), 
             result.get('apex_2ch_ed'), result.get('annulus_mid_2ch_ed')),
            (masks2, ES_2ch, "2CH ES", spacing2, result.get('axis_u_2ch_es'),
             result.get('apex_2ch_es'), result.get('annulus_mid_2ch_es')),
            (masks4, ED_4ch, "4CH ED", spacing4, result.get('axis_u_4ch_ed'),
             result.get('apex_4ch_ed'), result.get('annulus_mid_4ch_ed')),
            (masks4, ES_4ch, "4CH ES", spacing4, result.get('axis_u_4ch_es'),
             result.get('apex_4ch_es'), result.get('annulus_mid_4ch_es')),
        ]
        
        for col, (mask_list, frame_idx, title, spacing, axis_u, apex_mm, annulus_mm) in enumerate(frames):
            ax = axes[idx, col]
            
            mask_frame = mask_list[frame_idx]
            bg = np.zeros_like(mask_frame)
            
            ax.imshow(bg.T, cmap='gray', origin='lower')
            ax.imshow(np.ma.masked_where(_ensure_int_labels(mask_frame).T == 0, _ensure_int_labels(mask_frame).T),
                     cmap='Reds', alpha=0.5, origin='lower')
            
            draw_simpson_lines(
                ax, mask_frame, spacing,
                n_discs=20, band_frac=1.6, min_band_points=3,
                axis_u_override=axis_u,
                apex_mm=apex_mm,
                annulus_mid_mm=annulus_mm
            )
            
            ef = result['EF']
            edv = result['EDV']
            esv = result['ESV']
            
            info_text = f"LVEF: {ef:.1f}%\nEDV: {edv:.1f} mL\nESV: {esv:.1f} mL"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"{r['patient']} {title}", fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    for patient_id in PATIENTS:
        r = process_patient(patient_id)
        results.append(r)
    
    if any(results):
        plot_area_curves(results, os.path.join(OUTPUT_DIR, "01_area_curves.png"))
        plot_ed_es_overlay(results, os.path.join(OUTPUT_DIR, "02_ed_es_overlay.png"))
        plot_simpson_lines(results, os.path.join(OUTPUT_DIR, "03_simpson_lines.png"))
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"{'Patient':<15} {'EDV(mL)':<10} {'ESV(mL)':<10} {'LVEF(%)':<10}")
        print("-" * 50)
        for r in results:
            if r is not None:
                print(f"{r['patient']:<15} {r['result']['EDV']:<10.2f} {r['result']['ESV']:<10.2f} {r['result']['EF']:<10.2f}")
        print(f"\nOutput saved to: {OUTPUT_DIR}")
    else:
        print("No valid results to display")


if __name__ == "__main__":
    main()
