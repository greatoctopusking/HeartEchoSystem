"""
调试脚本：专门调试 _annulus_points_from_wall_la 函数
输出：标注两个瓣环点(pL,pR)及瓣环中点(mid)
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import cv2

#定义要处理的病人和文件
base_path=r"D:\SRTP_Project__DeepLearning\project\Resources\database_nifti"
output_dir=r"D:\SRTP_Project__DeepLearning\HeartEchoSystem10\HeartEchoSystem\backend\debug_output"

#文件列表
files_to_process=[
    ("patient0002","2CH","ED"),
    ("patient0002","2CH","ES"),
    ("patient0002","4CH","ED"),
    ("patient0002","4CH","ES"),
    ("patient0003","2CH","ED"),
    ("patient0003","2CH","ES"),
    ("patient0003","4CH","ED"),
    ("patient0003","4CH","ES"),
    ("patient0004","2CH","ED"),
    ("patient0004","2CH","ES"),
    ("patient0004","4CH","ED"),
    ("patient0004","4CH","ES"),
]

#新算法：基于边缘和极坐标
def annulus_points_from_polar(mask,spacing):
    """
    新算法：
    1. 找到质心位置
    2. 提取LV的边缘
    3. 以质心为中心作坐标系，每个边缘点有(theta,r)
    4. r是theta的函数：r(theta)
    5. 画出r(theta)图像
    
    返回: (theta_array,r_array,edge_pts,centroid_mm)
    """
    try:
        from scipy.ndimage import binary_erosion,binary_dilation
    except Exception:
        binary_erosion=None
        binary_dilation=None
    
    try:
        import cv2
    except Exception:
        cv2=None
    
    if cv2 is None:
        print("    [WARNING] cv2 not available,using morphological edge")
    
    arr=np.rint(np.asarray(mask)).astype(np.int16)
    lv=(arr==1)  #Cavity
    wall=(arr==2)  #Wall
    lv_all=lv | wall  #腔体和壁
    
    if np.count_nonzero(lv) < 30:
        return None,None,None,None
    
    dx,dy=float(spacing[0]),float(spacing[1])
    
    #1. 找质心
    cav_coords=np.column_stack(np.where(lv))
    cav_pts_mm=cav_coords[:,[1,0]]*[dx,dy]  #(x,y) in mm
    centroid_mm=cav_pts_mm.mean(axis=0)
    
    print(f"    [Step 1] Centroid: ({centroid_mm[0]:.1f},{centroid_mm[1]:.1f}) mm")
    
    #2. 提取LV边缘
    #使用cv2提取轮廓
    if cv2 is not None:
        lv_uint8=lv.astype(np.uint8)
        contours,_=cv2.findContours(lv_uint8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
        if contours:
            #取最大轮廓
            largest_contour=max(contours,key=cv2.contourArea)
            edge_pts=largest_contour[:,0,:].astype(float)  #(N,2) in pixels
            edge_pts_mm=edge_pts*[dx,dy]  #转换为mm
            print(f"    [Step 2] Edge points: {len(edge_pts)} pixels")
        else:
            print("    [ERROR] No contour found")
            return None,None,None,None
    else:
        #兜底：使用形态学操作提取边缘
        if binary_erosion is not None:
            eroded=binary_erosion(lv,iterations=2)
            edge=lv ^ eroded
        else:
            edge=lv
        
        edge_coords=np.column_stack(np.where(edge))
        edge_pts_mm=edge_coords[:,[1,0]]*[dx,dy]
        print(f"    [Step 2] Edge points (morphological): {len(edge_pts_mm)} pixels")
    
    #3. 极坐标转换
    #以质心为原点
    centered_pts=edge_pts_mm - centroid_mm  #(N,2)
    
    #计算theta和r
    #theta: 角度 [0,2*pi)
    #r: 距离
    r=np.sqrt(centered_pts[:,0]**2+centered_pts[:,1]**2)
    theta=np.arctan2(centered_pts[:,1],centered_pts[:,0])  #[-pi,pi]
    theta=np.mod(theta,2*np.pi)  #转换到 [0,2*pi)
    
    #按角度排序
    sort_idx=np.argsort(theta)
    theta_sorted=theta[sort_idx]
    r_sorted=r[sort_idx]
    
    #使用移动平均平滑曲线
    #使用移动平均而不是polyfit，避免过度拟合
    window_size=15
    r_smoothed=np.convolve(r_sorted,np.ones(window_size)/window_size,mode='same')
    
    print(f"    [Step 3.5] Smoothed with moving average (window={window_size})")
    
    #找三个极大值点（基于原始边缘点数据）
    #使用numpy实现简单峰值检测
    def find_peaks_numpy(data,min_distance=20,threshold_ratio=0.3):
        """找局部最大值（峰值）"""
        peaks=[]
        threshold=threshold_ratio*(data.max() - data.min())+data.min()
        
        for i in range(min_distance,len(data) - min_distance):
            #检查是否是局部最大值
            left=max(0,i - min_distance)
            right=min(len(data),i+min_distance)
            if data[i] >= threshold and data[i] >= max(data[left:right]):
                peaks.append(i)
        
        return peaks
    
    #在原始数据上找峰值（进一步降低阈值）
    all_peaks=find_peaks_numpy(r_sorted,min_distance=10,threshold_ratio=0.08)
    
    if len(all_peaks) >= 3:
        #取最高的三个峰值
        peak_heights=r_sorted[all_peaks]
        top3_idx=np.argsort(peak_heights)[-3:][::-1]
        top3_peaks=[all_peaks[i] for i in top3_idx]
    else:
        top3_peaks=all_peaks
    
    #使用平滑后的数据进行插值，获得更准确的峰值位置
    #在峰值附近进行二次插值精确定位
    refined_peaks=[]
    for peak_idx in top3_peaks:
        if peak_idx > 0 and peak_idx < len(r_sorted) - 1:
            #抛物线插值
            y0,y1,y2=r_sorted[peak_idx-1],r_sorted[peak_idx],r_sorted[peak_idx+1]
            a=(y0+y2 - 2*y1) / 2
            if abs(a) > 1e-10:
                delta=0.5*(y0 - y2) / a
                refined_idx=peak_idx+delta
                refined_peaks.append(refined_idx)
            else:
                refined_peaks.append(peak_idx)
        else:
            refined_peaks.append(peak_idx)
    
    top3_peaks_refined=refined_peaks
    
    print(f"    [Step 4] Found {len(top3_peaks_refined)} peaks: {[f'{np.degrees(theta_sorted[int(p)]):.1f}deg' for p in top3_peaks_refined] if top3_peaks_refined else []}")
    for i,p_idx in enumerate(top3_peaks_refined):
        idx=int(p_idx)
        print(f"      Peak {i+1}: theta={np.degrees(theta_sorted[idx]):.1f}deg,r={r_sorted[idx]:.1f}mm")
    
    #计算三个峰值点之间的物理距离（使用原始数据）
    if len(top3_peaks_refined) >= 2:
        peak_points=[]
        for i,peak_idx in enumerate(top3_peaks_refined):
            idx=int(peak_idx)
            t=theta_sorted[idx]
            r_val=r_sorted[idx]  #使用原始r值
            #转换为笛卡尔坐标
            x=centroid_mm[0]+r_val*np.cos(t)
            y=centroid_mm[1]+r_val*np.sin(t)
            peak_points.append(np.array([x,y]))
        
        #计算峰值之间的距离
        print(f"    [Step 5] Distance between peaks:")
        for i in range(len(peak_points)):
            for j in range(i+1,len(peak_points)):
                dist=np.linalg.norm(peak_points[i] - peak_points[j])
                t1=np.degrees(theta_sorted[int(top3_peaks_refined[i])])
                t2=np.degrees(theta_sorted[int(top3_peaks_refined[j])])
                print(f"      Peak {i+1} (theta={t1:.1f}deg) to Peak {j+1} (theta={t2:.1f}deg): {dist:.1f} mm")
    
    #返回数据供后续处理
    return theta_sorted,r_sorted,edge_pts_mm,centroid_mm,top3_peaks_refined,r_smoothed


#测试新算法
def test_polar_algorithm(mask,spacing,patient_id,view,frame_type):
    """测试极坐标算法"""
    print(f"\n{'='*60}")
    print(f"Testing Polar Algorithm: {patient_id}_{view}_{frame_type}")
    print(f"{'='*60}")
    
    result=annulus_points_from_polar(mask,spacing)
    
    if result[0] is None:
        print(f"  [ERROR] Failed to compute polar coordinates")
        return None
    
    theta,r,edge_pts,centroid,top3_peaks,r_smoothed=result
    
    #画图
    fig,axes=plt.subplots(2,2,figsize=(14,12))
    
    #1. 原始分割掩码
    ax1=axes[0,0]
    ax1.imshow(mask,cmap='jet',interpolation='nearest')
    ax1.set_title(f'{patient_id}_{view}_{frame_type}\nSegmentation')
    plt.colorbar(ax1.images[0],ax=ax1)
    
    #2. 边缘和质心+三个极大值点
    ax2=axes[0,1]
    ax2.imshow(mask,cmap='gray')
    
    #画边缘点
    edge_px=edge_pts / [spacing[0],spacing[1]]
    ax2.plot(edge_px[:,0],edge_px[:,1],'b.',markersize=1,alpha=0.3,label='Edge')
    
    #画质心
    centroid_px=centroid / [spacing[0],spacing[1]]
    ax2.scatter(centroid_px[0],centroid_px[1],c='red',s=200,marker='*',
                label=f'Centroid',edgecolors='white')
    
    #标注三个极大值点
    colors=['lime','yellow','cyan']
    peak_labels=['Peak 1','Peak 2','Peak 3']
    for i,peak_idx in enumerate(top3_peaks):
        if peak_idx is not None:
            idx=int(peak_idx) if isinstance(peak_idx,float) else peak_idx
            t=theta[idx]
            r_val=r[idx]
            #转换为图像坐标
            x_mm=centroid[0]+r_val*np.cos(t)
            y_mm=centroid[1]+r_val*np.sin(t)
            x_px=x_mm / spacing[0]
            y_px=y_mm / spacing[1]
            
            ax2.scatter(x_px,y_px,c=colors[i],s=200,marker='o',
                       label=f'{peak_labels[i]} ({np.degrees(t):.0f}deg)',
                       edgecolors='black',linewidths=1.5)
            
            #画从质心到极大值点的线
            ax2.plot([centroid_px[0],x_px],[centroid_px[1],y_px],
                    colors[i],linewidth=1,alpha=0.7)
    
    ax2.set_title('Edge Points & 3 Max Peaks')
    ax2.legend(loc='upper right',fontsize=8)
    
    #3. r(theta) 函数图像+标注极大值点
    ax3=axes[1,0]
    ax3.plot(np.degrees(theta),r,'b-',linewidth=0.5,alpha=0.5,label='Original')
    
    #平滑后的曲线
    ax3.plot(np.degrees(theta),r_smoothed,'r-',linewidth=2,label='Smoothed (MA)')
    
    #标注极大值点
    for i,peak_idx in enumerate(top3_peaks):
        if peak_idx is not None:
            idx=int(peak_idx)
            t_deg=np.degrees(theta[idx])
            r_val=r[idx]
            ax3.scatter(t_deg,r_val,c=colors[i],s=100,marker='o',zorder=5)
            ax3.annotate(f'P{i+1}: {r_val:.1f}mm',
                        xy=(t_deg,r_val),
                        xytext=(t_deg+10,r_val+2),
                        color=colors[i],fontsize=9,fontweight='bold')
    
    ax3.set_xlabel('Theta (degrees)')
    ax3.set_ylabel('r (mm)')
    ax3.set_title('r(theta) Function with 3 Max Peaks')
    ax3.legend()
    ax3.grid(True,alpha=0.3)
    ax3.set_xlim(0,360)
    
    #4. 极坐标图+标注极大值点
    ax4=axes[1,1]
    ax4.plot(r_smoothed*np.cos(theta),r_smoothed*np.sin(theta),'r-',linewidth=1.5,label='Smoothed')
    ax4.plot(r*np.cos(theta),r*np.sin(theta),'b.',markersize=1,alpha=0.3)
    ax4.scatter(0,0,c='red',s=200,marker='*',label='Centroid',zorder=5)
    
    #标注极大值点
    for i,peak_idx in enumerate(top3_peaks):
        if peak_idx is not None:
            idx=int(peak_idx)
            t=theta[idx]
            r_val=r[idx]
            x=r_val*np.cos(t)
            y=r_val*np.sin(t)
            ax4.scatter(x,y,c=colors[i],s=150,marker='o',
                       label=f'Peak {i+1}',zorder=5,edgecolors='black')
    
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('Polar Plot with 3 Max Peaks')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True,alpha=0.3)
    
    plt.tight_layout()
    
    output_path=os.path.join(output_dir,f'polar_{patient_id}_{view}_{frame_type}.png')
    plt.savefig(output_path,dpi=150,bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return {
        'theta': theta,
        'r': r,
        'edge_pts': edge_pts,
        'centroid': centroid,
        'top3_peaks': top3_peaks
    }

    #4) 【二次兜底】如果某侧还是没点，在该侧搜索 LV 的几何边缘点
    if pL is None or pR is None:
        #获取 LV 轮廓点作为最后保障
        all_lv_pts=cav_pts
        proj_all=(all_lv_pts - mean_c) @ v_perp
        if pL is None:
            pL=all_lv_pts[proj_all < 0][np.argmin(all_lv_pts[proj_all < 0] @ v_long)] #选最靠近 base 的
        if pR is None:
            pR=all_lv_pts[proj_all >= 0][np.argmin(all_lv_pts[proj_all >= 0] @ v_long)]

    #5) 距离安全检查：如果两点太近，说明分区失败，强制取 LV 最宽处
    min_dist=15.0 #1.5cm 物理阈值
    if np.linalg.norm(pL - pR) < min_dist:
        #这种情况下，强制取投影极值
        #proj_all 在上面分支里可能没定义，这里补一行，保持你原逻辑不动
        proj_all=(cav_pts - mean_c) @ v_perp
        pL=cav_pts[np.argmin(proj_all)]
        pR=cav_pts[np.argmax(proj_all)]

    mid=0.5*(pL+pR)
    return pR.astype(float),pL.astype(float),mid.astype(float)


#批量处理
results=[]

for patient_id,view,frame_type in files_to_process:
    file_name=f"{patient_id}_{view}_{frame_type}_gt.nii.gz"
    file_path=os.path.join(base_path,patient_id,file_name)
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        continue
    
    #加载图像
    img=nib.load(file_path)
    data=img.get_fdata()
    spacing=img.header.get_zooms()[:2]
    
    #取第一帧
    if len(data.shape)==3:
        mask=data[:,:,0]
    elif len(data.shape)==4:
        mask=data[:,:,:,0]
    else:
        mask=data
    
    print(f"  Shape: {mask.shape},Spacing: {spacing}")
    
    #调用新的极坐标算法
    try:
        result=test_polar_algorithm(mask,spacing,patient_id,view,frame_type)
        
        if result is None:
            print(f"  [ERROR] Failed to compute polar coordinates")
            continue
        
        #保存结果
        results.append({
            'patient': patient_id,
            'view': view,
            'frame': frame_type,
            'centroid': result['centroid'],
            'theta': result['theta'],
            'r': result['r'],
            'edge_pts': result['edge_pts'],
            'mask': mask,
            'spacing': spacing,
        })
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        continue


#汇总表
print(f"\n\n{'='*60}")
print("SUMMARY - Centroid Results")
print(f"{'='*60}")
print(f"{'Patient':<12} {'View':<6} {'Frame':<6} {'Centroid (mm)':<20}")
print("-"*60)
for r in results:
    print(f"{r['patient']:<12} {r['view']:<6} {r['frame']:<6} ({r['centroid'][0]:.1f},{r['centroid'][1]:.1f})")

print(f"\nDone! Total: {len(results)} images generated.")
