"""
Flask 后端 — 心脏超声智能分析系统
支持 annulus_strategy 参数（"polar"/"wall_la"/"auto"）
支持 AVI/MP4/MOV 输入
视频自动转换为 NIfTI(H,W,T) 后进入 nnUNet 推理
3D series 统一调用 align_series_indices（全序列重采样）
"""

import os
import sys
import json
import datetime
import traceback
import uuid

import numpy as np
import nibabel as nib
import pymysql
import jwt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functools import wraps
from flask import Flask,request,jsonify,send_from_directory,Response, abort, send_file
from werkzeug.utils import secure_filename

from config import *
from model_infer import run_inference
from biplane_simpson_clinical import BiplaneSimpsonClinical,ALGORITHM_LABELS

# 视图自动分类器
from inference.view_classifier import classify_view, auto_assign_views, init_classifier

#视频处理
import cv2
from scipy.ndimage import label


UPLOAD_FOLDER="uploads"
RESULT_FOLDER="results"
BASE_URL="http://127.0.0.1:5000"

#补全视频扩展名
SUPPORTED_EXT={
    ".nii",".gz",
    ".dcm",".dicom",".ima",".img",
    ".avi",".mp4",".mov"
}

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(RESULT_FOLDER,exist_ok=True)

app=Flask(__name__)

LABEL_MAP={0:"background",1:"LV",2:"LVWall",3:"LA"}


# 数据库
def get_db(use_database=True):
    conn_kwargs=dict(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=False,
    )
    if use_database:
        conn_kwargs["database"]=MYSQL_DB
    return pymysql.connect(**conn_kwargs)


def _ensure_db_schema():
    """
    启动时自动完成：
    1. 数据库不存在则创建
    2. 表不存在则创建
    不破坏现有数据
    """
    conn=get_db(use_database=False)
    cur=conn.cursor()

    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` "
        "DEFAULT CHARACTER SET utf8mb4 "
        "DEFAULT COLLATE utf8mb4_unicode_ci"
    )
    cur.execute(f"USE `{MYSQL_DB}`")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(255),
            role VARCHAR(20),
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient (
            id INT PRIMARY KEY AUTO_INCREMENT,
            patient_uid VARCHAR(64) UNIQUE,
            name VARCHAR(50),
            age INT,
            gender VARCHAR(10),
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_record (
            id INT PRIMARY KEY AUTO_INCREMENT,
            patient_id INT,
            image_path VARCHAR(255),
            result_path VARCHAR(255),
            lvef FLOAT,
            edv FLOAT,
            esv FLOAT,
            algorithm VARCHAR(50) DEFAULT 'biplane_simpson',
            view_mode VARCHAR(20) DEFAULT 'biplane',
            annulus_strategy VARCHAR(20) DEFAULT 'auto',
            create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patient(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    conn.close()



# JWT 鉴权
def token_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        token=request.headers.get("Authorization","").replace("Bearer ","")
        if not token:
            return jsonify({"error":"Token missing"}),401
        try:
            # 尝试 JWT 验证
            jwt.decode(token,SECRET_KEY,algorithms=["HS256"])
        except:
            # 兼容旧版 UUID token（本地测试模式）
            # UUID 格式：xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            if len(token) == 36 and token.count('-') == 4:
                # 认为是有效的 UUID token，放行
                pass
            else:
                return jsonify({"error":"Invalid token"}),401
        return f(*args,**kwargs)
    return decorated



def _normalize_ultrasound_image(img:np.ndarray) -> np.ndarray:
    img=np.asarray(img,dtype=np.float32)

    if img.size == 0:
        return img.astype(np.float32)

    finite=np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img,dtype=np.float32)

    vals=img[finite]
    p1,p99=np.percentile(vals,[1,99])

    if p99 <= p1:
        mn,mx=float(vals.min()),float(vals.max())
        if mx <= mn:
            return np.zeros_like(img,dtype=np.float32)
        out=(img - mn)/(mx - mn)
    else:
        out=np.clip((img - p1)/(p99 - p1),0.0,1.0)

    return (out*255.0).astype(np.float32)


def _crop_to_ultrasound_content(frame:np.ndarray,pad:int=8) -> np.ndarray:
    img=np.asarray(frame)
    if img.ndim != 2:
        return img

    thr=max(5,int(np.percentile(img,60)*0.15))
    fg=img > thr
    if fg.sum() < img.size*0.01:
        return img

    ys,xs=np.where(fg)
    y0,y1=max(0,ys.min() - pad),min(img.shape[0],ys.max() + pad + 1)
    x0,x1=max(0,xs.min() - pad),min(img.shape[1],xs.max() + pad + 1)

    cropped=img[y0:y1,x0:x1]
    if cropped.size == 0:
        return img
    return cropped

# DICOM 支持
def _is_dicom(path:str) -> bool:
    ext=os.path.splitext(path.lower())[1]
    if ext in (".dcm",".dicom",".ima"):
        return True
    try:
        with open(path,"rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def load_dicom_to_array(path:str):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    def _postprocess_dcm_pixels(ds):
        arr=apply_voi_lut(ds.pixel_array,ds).astype(np.float32)

        slope=float(getattr(ds,"RescaleSlope",1.0) or 1.0)
        intercept=float(getattr(ds,"RescaleIntercept",0.0) or 0.0)
        arr=arr*slope + intercept

        if str(getattr(ds,"PhotometricInterpretation","")).upper() == "MONOCHROME1":
            arr=arr.max() - arr

        return _normalize_ultrasound_image(arr)

    if os.path.isdir(path):
        dcm_files=sorted(
            [f for f in os.listdir(path) if not f.startswith(".")],
            key=lambda f:int(
                pydicom.dcmread(
                    os.path.join(path,f),stop_before_pixels=True
                ).get("InstanceNumber",0)
            )
        )
        slices=[pydicom.dcmread(os.path.join(path,f)) for f in dcm_files]
        data=np.stack([_postprocess_dcm_pixels(s) for s in slices],axis=-1)
        ps=getattr(slices[0],"PixelSpacing",[1.0,1.0])
    else:
        ds=pydicom.dcmread(path)
        n_frames=int(getattr(ds,"NumberOfFrames",1))
        arr=_postprocess_dcm_pixels(ds)
        if n_frames > 1:
            if arr.ndim == 3:
                data=np.transpose(arr,(1,2,0))
            else:
                data=arr[:,:,np.newaxis]
        else:
            data=arr[:,:,np.newaxis]
        ps=getattr(ds,"PixelSpacing",[1.0,1.0])

    return data.astype(np.float32),(float(ps[1]),float(ps[0]))

def dicom_to_nifti(path:str,out_path:str):
    data,spacing=load_dicom_to_array(path)
    affine=np.diag([spacing[0],spacing[1],1.0,1.0])
    img=nib.Nifti1Image(data,affine)
    img.header.set_zooms((*spacing,1.0))
    nib.save(img,out_path)
    return out_path,spacing


# 视频支持
def _is_video(path:str) -> bool:
    ext=os.path.splitext(path.lower())[1]
    return ext in (".avi",".mp4",".mov")

def video_to_nifti(path:str,out_path:str,max_frames:int=80):
    """
    将视频转换为 NIfTI (H,W,T)。
提取全局统一裁剪框，保证物理空间无形变，确保后续容积/曲线计算精准。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"视频文件不存在:{path}")

    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件:{path}")

    raw_frames=[]
    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            if frame is None:
                continue

            if frame.ndim == 2:
                gray=frame
            elif frame.ndim == 3:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            else:
                continue
            raw_frames.append(gray)
    finally:
        cap.release()

    if not raw_frames:
        raise ValueError("无法读取视频内容，未提取到有效帧")

    #1. 限帧采样：视频太长时均匀采样，减少计算量
    total_frames=len(raw_frames)
    if max_frames is not None and total_frames > max_frames:
        sample_idx=np.linspace(0,total_frames - 1,max_frames).astype(int)
        raw_frames=[raw_frames[i] for i in sample_idx]
        print(f"[INFO] 视频原始帧数={total_frames}，已均匀采样到 {len(raw_frames)} 帧")
    else:
        print(f"[INFO] 视频帧数={total_frames}，无需采样")

    #2. 快速计算【全局统一裁剪框】
    #抽取 几帧做最大值投影，速度极快，且能确保囊括心动周期内所有可能的超声扇形亮区
    #step=max(1,len(raw_frames) // 5)
    step=max(1,len(raw_frames) // 10)
    sample_for_box=np.max(np.stack(raw_frames[::step],axis=0),axis=0)

    pad=8
    thr=max(5,int(np.percentile(sample_for_box,60)*0.15))
    fg=sample_for_box > thr
    if fg.sum() < sample_for_box.size*0.01:
        #如果提取失败，使用全图
        y0,y1,x0,x1=0,sample_for_box.shape[0],0,sample_for_box.shape[1]
    else:
        ys,xs=np.where(fg)
        y0,y1=max(0,ys.min() - pad),min(sample_for_box.shape[0],ys.max() + pad + 1)
        x0,x1=max(0,xs.min() - pad),min(sample_for_box.shape[1],xs.max() + pad + 1)

    #3. 统一裁剪与归一化
    frames=[]
    for gray in raw_frames:
        #所有帧使用同一个绝对坐标裁剪，保证画面无拉伸形变
        cropped=gray[y0:y1,x0:x1]
        normed=_normalize_ultrasound_image(cropped)
        frames.append(normed.astype(np.float32))

    #组合为 NIfTI (由于尺寸严格一致，省去了原本耗时的 cv2.resize)
    data=np.stack(frames,axis=-1)
    spacing=(1.0,1.0)
    affine=np.diag([spacing[0],spacing[1],1.0,1.0])
    img=nib.Nifti1Image(data,affine)
    img.header.set_zooms((*spacing,1.0))
    nib.save(img,out_path)

    return out_path,spacing


def save_upload(fileobj,dest_dir:str,prefix:str=""):
    filename=secure_filename(fileobj.filename or "")
    if not filename:
        raise ValueError("上传文件名为空")

    lower_name=filename.lower()
    if not any(lower_name.endswith(ext) for ext in SUPPORTED_EXT):
        raise ValueError(f"不支持的文件类型:{filename}")

    raw_path=os.path.join(dest_dir,prefix + filename)
    fileobj.save(raw_path)

    if _is_dicom(raw_path):
        nii_path=raw_path + "_converted.nii.gz"
        _,spacing=dicom_to_nifti(raw_path,nii_path)
        return nii_path,spacing,False

    if _is_video(raw_path):
        nii_path=raw_path + "_converted.nii.gz"
        _,spacing=video_to_nifti(raw_path,nii_path)
        return nii_path,spacing,True

    return raw_path,None,False



# 自动检测 annulus 策略
def _detect_annulus_strategy(
    masks:list,
    min_frames:int=3,
    min_pixels_wall:int=30,
    min_pixels_la:int=30,
) -> str:
    """
    自动检测 annulus 策略：
    只有当至少 min_frames 帧中，
    Wall(2) 和 LA(3) 都达到最小像素数时，才启用 wall_la；
    否则使用 polar。
    """
    valid_count=0

    for m in masks:
        arr=np.rint(np.asarray(m)).astype(np.int16)

        n2=int(np.sum(arr == 2))
        n3=int(np.sum(arr == 3))

        if n2 >= min_pixels_wall and n3 >= min_pixels_la:
            valid_count += 1
            if valid_count >= min_frames:
                return "wall_la"

    return "polar"


def _resolve_strategy(requested:str,masks2:list,masks4:list) -> str:
    """将 'auto' 解析为实际策略，其余直接透传。"""
    if requested != "auto":
        return requested
    all_masks=list(masks2) + list(masks4)
    return _detect_annulus_strategy(all_masks) if all_masks else "polar"


# Overlay 图（Simpson 线条叠加）
def _ensure_int_labels(mask_arr):
    arr=np.asarray(mask_arr)
    return arr if arr.dtype.kind in ("i","u") else np.rint(arr).astype(np.int16)


def draw_simpson_lines(
    ax,mask,spacing,n_discs=20,
    band_frac=1.0,min_band_points=3,
    axis_u_override=None,apex_mm=None,annulus_mid_mm=None
):
    dx,dy=float(spacing[0]),float(spacing[1])

    def to_disp(xp,yp):
        return yp,xp

    m=(_ensure_int_labels(mask) == 1)
    coords=np.column_stack(np.where(m))
    if coords.shape[0] < 30:
        return

    pts=np.column_stack([
        coords[:,1].astype(float)*dx,
        coords[:,0].astype(float)*dy
    ])
    mean_pt=pts.mean(axis=0)

    if axis_u_override is not None:
        axis_u=np.asarray(axis_u_override,dtype=float)
    elif apex_mm is not None and annulus_mid_mm is not None:
        axis_u=np.asarray(apex_mm) - np.asarray(annulus_mid_mm)
    else:
        X=pts - mean_pt
        C=(X.T @ X)/max(1,X.shape[0] - 1)
        vals,vecs=np.linalg.eigh(C)
        axis_u=vecs[:,int(np.argmax(vals))]

    axis_u=axis_u/(np.linalg.norm(axis_u) + 1e-12)
    perp_u=np.array([-axis_u[1],axis_u[0]],dtype=float)

    if apex_mm is not None and annulus_mid_mm is not None:
        origin_pt=np.asarray(annulus_mid_mm,dtype=float)
        apex_pt=np.asarray(apex_mm,dtype=float)
        if float(np.dot(apex_pt - origin_pt,axis_u)) < 0:
            axis_u=-axis_u
            perp_u=np.array([-axis_u[1],axis_u[0]],dtype=float)

        centered=pts - origin_pt
        t=centered @ axis_u
        s=centered @ perp_u
        L=float(np.dot(apex_pt - origin_pt,axis_u))
        p1,p2=origin_pt,apex_pt
        tmin_loop=0.0
    else:
        centered=pts - mean_pt
        t=centered @ axis_u
        s=centered @ perp_u
        tmin,tmax=float(np.min(t)),float(np.max(t))
        L=tmax - tmin
        if L <= 1e-6:
            return
        p1=mean_pt + axis_u*tmin
        p2=mean_pt + axis_u*tmax
        tmin_loop=tmin

    if L <= 1e-6:
        return

    ax.plot(
        [to_disp(p1[0]/dx,p1[1]/dy)[0],to_disp(p2[0]/dx,p2[1]/dy)[0]],
        [to_disp(p1[0]/dx,p1[1]/dy)[1],to_disp(p2[0]/dx,p2[1]/dy)[1]],
        color='cyan',linewidth=2
    )

    h_disc=L/n_discs
    centers=tmin_loop + (np.arange(n_discs) + 0.5)*h_disc
    band_half=0.5*band_frac*h_disc
    origin_pt_draw=np.asarray(annulus_mid_mm,float) if annulus_mid_mm is not None else mean_pt

    for c in centers:
        band=(t >= c - band_half) & (t <= c + band_half)
        if np.count_nonzero(band) < min_band_points:
            continue
        smin,smax=float(np.min(s[band])),float(np.max(s[band]))
        if smax <= smin:
            continue

        cpt=origin_pt_draw + axis_u*c
        a=cpt + perp_u*smin
        b=cpt + perp_u*smax

        ax.plot(
            [to_disp(a[0]/dx,a[1]/dy)[0],to_disp(b[0]/dx,b[1]/dy)[0]],
            [to_disp(a[0]/dx,a[1]/dy)[1],to_disp(b[0]/dx,b[1]/dy)[1]],
            color='yellow',linewidth=1.0,alpha=0.75
        )



def _save_single_frame_overlay(
    save_path,
    view_name,
    frame_i,
    mask,
    bg,
    spacing,
    axis_u_override=None,
    apex_mm=None,
    annulus_mid_mm=None,
    n_discs=20,
    band_frac=1.0,
    min_band_points=3,
):
    fig,ax=plt.subplots(1,1,figsize=(6,6))
    fig.patch.set_facecolor('#111111')

    # 正常绘制（心尖朝下）
    if bg is not None:
        ax.imshow(bg.T,cmap='gray',origin='lower')

    int_mask=_ensure_int_labels(mask)

    #默认只显示 LV
    lv_only=(int_mask == 1).astype(np.uint8)
    ax.imshow(
        np.ma.masked_where(lv_only.T == 0,lv_only.T),
        cmap='Reds',alpha=0.50,origin='lower'
    )

    # 左心肌 LVWall(label=2) -> 半透明红色
    lv_wall=(int_mask == 2)
    if np.any(lv_wall):
        wall_rgba=np.zeros((lv_wall.T.shape[0],lv_wall.T.shape[1],4),dtype=np.float32)
        wall_rgba[...,0]=1.0   # R
        wall_rgba[...,1]=0.0   # G
        wall_rgba[...,2]=0.0   # B
        wall_rgba[...,3]=lv_wall.T.astype(np.float32) * 0.35  # alpha
        ax.imshow(wall_rgba,origin='lower')

    # 左心房 LA(label=3) -> 半透明绿色
    la_mask=(int_mask == 3)
    if np.any(la_mask):
        la_rgba=np.zeros((la_mask.T.shape[0],la_mask.T.shape[1],4),dtype=np.float32)
        la_rgba[...,0]=0.0
        la_rgba[...,1]=1.0
        la_rgba[...,2]=0.0
        la_rgba[...,3]=la_mask.T.astype(np.float32) * 0.35
        ax.imshow(la_rgba,origin='lower')
    

    draw_simpson_lines(
        ax,mask,spacing,
        n_discs=n_discs,
        band_frac=band_frac,
        min_band_points=min_band_points,
        axis_u_override=axis_u_override,
        apex_mm=apex_mm,
        annulus_mid_mm=annulus_mid_mm,
    )

    # 整体翻转Y轴，使心尖朝上
    ax.invert_yaxis()

    ax.set_title(
        f"{view_name} frame {frame_i}",
        color='white',fontsize=12,fontweight='bold',pad=8
    )
    ax.axis('off')

    fig.tight_layout(pad=1.0)
    fig.savefig(
        save_path,
        bbox_inches='tight',
        dpi=150,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)

def _interp_series_np(arr, target_len):
    arr = np.asarray(arr, dtype=float)
    src_len = arr.shape[0]

    if src_len == 0:
        return arr
    if src_len == target_len:
        return arr.copy()
    if src_len == 1:
        return np.repeat(arr, target_len, axis=0)

    x_old = np.linspace(0.0, 1.0, src_len)
    x_new = np.linspace(0.0, 1.0, target_len)

    flat = arr.reshape(src_len, -1)
    out = np.zeros((target_len, flat.shape[1]), dtype=float)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(x_new, x_old, flat[:, j])

    return out.reshape((target_len,) + arr.shape[1:])


def _smooth_frame_infos(frame_infos, target_frames=80, smooth_sigma=1.0):
    if not frame_infos:
        return []

    if len(frame_infos) == 1:
        return [frame_infos[0]] * target_frames

    bounds_2ch = np.array([x["bounds_2ch"] for x in frame_infos], dtype=float)
    bounds_4ch = np.array([x["bounds_4ch"] for x in frame_infos], dtype=float)
    h_mm = np.array([x["h_mm"] for x in frame_infos], dtype=float)

    origin_2ch_mm = np.array([
        x["origin_2ch_mm"] if x.get("origin_2ch_mm") is not None else [0.0, 0.0]
        for x in frame_infos
    ], dtype=float)

    axis_u_2ch = np.array([
        x["axis_u_2ch"] if x.get("axis_u_2ch") is not None else [0.0, 1.0]
        for x in frame_infos
    ], dtype=float)

    bounds_2ch = _interp_series_np(bounds_2ch, target_frames)
    bounds_4ch = _interp_series_np(bounds_4ch, target_frames)
    h_mm = _interp_series_np(h_mm, target_frames)
    origin_2ch_mm = _interp_series_np(origin_2ch_mm, target_frames)
    axis_u_2ch = _interp_series_np(axis_u_2ch, target_frames)

    if smooth_sigma > 0:
        radius = max(1, int(round(smooth_sigma * 3)))
        xs = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(xs * xs) / (2.0 * smooth_sigma * smooth_sigma))
        kernel /= kernel.sum()

        def _smooth_last_dims(arr):
            arr = np.asarray(arr, dtype=float)
            flat = arr.reshape(arr.shape[0], -1)
            out = np.zeros_like(flat)
            for col in range(flat.shape[1]):
                padded = np.pad(flat[:, col], (radius, radius), mode="edge")
                out[:, col] = np.convolve(padded, kernel, mode="valid")
            return out.reshape(arr.shape)

        bounds_2ch = _smooth_last_dims(bounds_2ch)
        bounds_4ch = _smooth_last_dims(bounds_4ch)
        h_mm = _smooth_last_dims(h_mm)
        origin_2ch_mm = _smooth_last_dims(origin_2ch_mm)
        axis_u_2ch = _smooth_last_dims(axis_u_2ch)

    norm = np.linalg.norm(axis_u_2ch, axis=1, keepdims=True) + 1e-12
    axis_u_2ch = axis_u_2ch / norm

    out = []
    for i in range(target_frames):
        out.append({
            "bounds_2ch": bounds_2ch[i].tolist(),
            "bounds_4ch": bounds_4ch[i].tolist(),
            "h_mm": float(max(h_mm[i], 1e-6)),
            "origin_2ch_mm": origin_2ch_mm[i].tolist(),
            "axis_u_2ch": axis_u_2ch[i].tolist(),
        })
    return out

# 3D Mesh 生成
def _build_3d_series(calculator,masks2,masks4,spacing2,spacing4,result):
    ed_2ch=int(result['ED_index'])
    es_2ch=int(result['ES_index'])
    ed_4ch=int(result.get('ED_index_4ch',0))
    es_4ch=int(result.get('ES_index_4ch',0))

    T2=len(masks2) if masks2 else 0
    T4=len(masks4) if masks4 else 0
    
    # 判断是单平面还是双平面模式
    has_2ch = T2 > 0
    has_4ch = T4 > 0
    is_singleplane = not (has_2ch and has_4ch)

    aligned_2ch,aligned_4ch=calculator.align_series_indices(
        ed_2ch,es_2ch,max(T2,1),
        ed_4ch,es_4ch,max(T4,1)
    )
    T=len(aligned_2ch)
    if T <= 0:
        return None

    # 安全检查：ED/ES索引
    if has_2ch and ed_2ch >= len(masks2):
        print(f"[WARN] 3D: ed_2ch ({ed_2ch}) out of range (masks2 len={len(masks2)}), using 0")
        ed_2ch = 0
    if has_4ch and ed_4ch >= len(masks4):
        print(f"[WARN] 3D: ed_4ch ({ed_4ch}) out of range (masks4 len={len(masks4)}), using 0")
        ed_4ch = 0
    
    # 双平面：使用两个视图；单平面：使用同一个视图作为参考
    try:
        if has_2ch and has_4ch:
            info_ref=calculator.frame_bounds_and_L(
                masks2[ed_2ch],masks4[ed_4ch],spacing2,spacing4
            )
        elif has_2ch:
            # 只有2CH：用2CH作为参考，构建对称模型
            info_ref=calculator.frame_bounds_and_L(
                masks2[ed_2ch],masks2[ed_2ch],spacing2,spacing2
            )
        elif has_4ch:
            # 只有4CH：用4CH作为参考
            info_ref=calculator.frame_bounds_and_L(
                masks4[ed_4ch],masks4[ed_4ch],spacing4,spacing4
            )
        else:
            return None
    except Exception as e:
        print(f"[WARN] 3D reference frame calculation failed: {e}")
        return None

    origin_ref_2ch=info_ref.get("origin_2ch_mm",None)
    axis_ref_2ch=info_ref.get("axis_u_2ch",None)

    if origin_ref_2ch is None or axis_ref_2ch is None:
        if has_2ch:
            m_ref=masks2[aligned_2ch[0]]
            info_ref0=calculator.frame_bounds_and_L(m_ref,m_ref,spacing2,spacing2)
        elif has_4ch:
            m_ref=masks4[aligned_4ch[0]]
            info_ref0=calculator.frame_bounds_and_L(m_ref,m_ref,spacing4,spacing4)
        else:
            return None
        origin_ref_2ch=info_ref0.get("origin_2ch_mm",None)
        axis_ref_2ch=info_ref0.get("axis_u_2ch",None)
    
    # 最终检查：如果仍然无法获取参考点和轴，使用默认值
    if origin_ref_2ch is None:
        origin_ref_2ch = np.array([0.0, 0.0, 0.0])
        print("[WARN] 3D: origin_ref_2ch is None, using default [0,0,0]")
    if axis_ref_2ch is None:
        axis_ref_2ch = np.array([0.0, -1.0, 0.0])  # 默认朝上
        print("[WARN] 3D: axis_ref_2ch is None, using default [0,-1,0]")

    # 获取参考帧的bounds
    if has_2ch:
        m_ref_0=masks2[aligned_2ch[0]]
        info0=calculator.frame_bounds_and_L(m_ref_0,m_ref_0,spacing2,spacing2)
    elif has_4ch:
        m_ref_0=masks4[aligned_4ch[0]]
        info0=calculator.frame_bounds_and_L(m_ref_0,m_ref_0,spacing4,spacing4)
    else:
        return None
    
    # 单平面模式下，允许只有bounds_2ch或bounds_4ch
    b2ch_ref = info0.get("bounds_2ch")
    b4ch_ref = info0.get("bounds_4ch")
    h_mm_ref = info0.get("h_mm", 0)
    
    if b2ch_ref is None:
        print("[WARN] 3D: bounds_2ch is None, skip 3D generation")
        return None
    
    # 单平面时，如果没有bounds_4ch，用bounds_2ch替代
    if b4ch_ref is None:
        b4ch_ref = b2ch_ref
    
    # 确保h_mm有效
    if h_mm_ref <= 1e-6:
        print(f"[WARN] 3D: h_mm too small ({h_mm_ref}), skip 3D generation")
        return None

    try:
        verts,f0=calculator.generate_3d_mesh_asymmetric(
            b2ch_ref, b4ch_ref, h_mm_ref
        )
        # 检查结果有效性
        if verts is None or len(verts) == 0:
            print("[WARN] 3D mesh generation returned empty vertices")
            return None
        if f0 is None or len(f0) == 0:
            print("[WARN] 3D mesh generation returned empty faces")
            return None
    except Exception as e:
        print(f"[WARN] 3D mesh generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    faces_list=np.asarray(f0,dtype=int).tolist()
    
    
    frame_infos = []
    for t_idx in range(T):
        if has_2ch and t_idx >= len(aligned_2ch):
            print(f"[WARN] 3D: aligned_2ch index out of range: {t_idx} >= {len(aligned_2ch)}")
            continue
        if has_4ch and t_idx >= len(aligned_4ch):
            print(f"[WARN] 3D: aligned_4ch index out of range: {t_idx} >= {len(aligned_4ch)}")
            continue

        idx2 = aligned_2ch[t_idx] if has_2ch else 0
        idx4 = aligned_4ch[t_idx] if has_4ch else 0

        if has_2ch and idx2 >= len(masks2):
            print(f"[WARN] 3D: masks2 index out of range: {idx2} >= {len(masks2)}")
            continue
        if has_4ch and idx4 >= len(masks4):
            print(f"[WARN] 3D: masks4 index out of range: {idx4} >= {len(masks4)}")
            continue

        try:
            if has_2ch and has_4ch:
                m2 = masks2[idx2]
                m4 = masks4[idx4]
                info = calculator.frame_bounds_and_L(m2, m4, spacing2, spacing4)
                b2ch = info.get("bounds_2ch")
                b4ch = info.get("bounds_4ch")
            elif has_2ch:
                m = masks2[idx2]
                info = calculator.frame_bounds_and_L(m, m, spacing2, spacing2)
                b2ch = info.get("bounds_2ch")
                b4ch = b2ch
            elif has_4ch:
                m = masks4[idx4]
                info = calculator.frame_bounds_and_L(m, m, spacing4, spacing4)
                b4ch = info.get("bounds_4ch")
                b2ch = b4ch
            else:
                continue

            h_mm = info.get("h_mm", 0)

            if b2ch is None or b4ch is None or h_mm <= 1e-6:
                continue

            frame_infos.append({
                "bounds_2ch": b2ch,
                "bounds_4ch": b4ch,
                "h_mm": h_mm,
                "origin_2ch_mm": origin_ref_2ch,
                "axis_u_2ch": axis_ref_2ch,
            })
        except Exception as e:
            print(f"[WARN] 3D frame info {t_idx} extraction failed: {e}")
            continue

    if not frame_infos:
        return None

    target_frames = max(len(frame_infos), 80 if len(frame_infos) < 80 else len(frame_infos))
    smooth_infos = _smooth_frame_infos(
        frame_infos,
        target_frames=target_frames,
        smooth_sigma=1.0
    )

    vertices_series = []
    for t_idx, info in enumerate(smooth_infos):
        try:
            v, _ = calculator.generate_3d_mesh_asymmetric(
                info["bounds_2ch"],
                info["bounds_4ch"],
                info["h_mm"],
                origin_2ch_mm=info["origin_2ch_mm"],
                axis_u_2ch=info["axis_u_2ch"]
            )
            vertices_series.append(np.asarray(v, dtype=float).tolist())
        except Exception as e:
            print(f"[WARN] 3D smoothed frame {t_idx} generation failed: {e}")
            vertices_series.append(
                vertices_series[-1] if vertices_series
                else np.zeros((calculator.n*32,3)).tolist()
            )
    """
    vertices_series=[]
    for t_idx in range(T):
        # 安全检查：确保索引在有效范围内
        if has_2ch and t_idx >= len(aligned_2ch):
            print(f"[WARN] 3D: aligned_2ch index out of range: {t_idx} >= {len(aligned_2ch)}")
            continue
        if has_4ch and t_idx >= len(aligned_4ch):
            print(f"[WARN] 3D: aligned_4ch index out of range: {t_idx} >= {len(aligned_4ch)}")
            continue
            
        idx2=aligned_2ch[t_idx] if has_2ch else 0
        idx4=aligned_4ch[t_idx] if has_4ch else 0
        
        # 再检查索引是否在masks范围内
        if has_2ch and idx2 >= len(masks2):
            print(f"[WARN] 3D: masks2 index out of range: {idx2} >= {len(masks2)}")
            continue
        if has_4ch and idx4 >= len(masks4):
            print(f"[WARN] 3D: masks4 index out of range: {idx4} >= {len(masks4)}")
            continue
        
        # 单平面模式：使用同一个视图的 bounds 作为两个平面的输入
        if has_2ch and has_4ch:
            # 双平面模式
            m2=masks2[idx2]
            m4=masks4[idx4]
            info=calculator.frame_bounds_and_L(m2,m4,spacing2,spacing4)
            b2ch=info.get("bounds_2ch")
            b4ch=info.get("bounds_4ch")
        elif has_2ch:
            # 单平面2CH：用2CH bounds 作为两个平面的输入（构建对称椭球）
            m=masks2[idx2]
            info=calculator.frame_bounds_and_L(m,m,spacing2,spacing2)
            b2ch=info.get("bounds_2ch")
            b4ch=b2ch  # 单平面时两个平面用相同的bounds
        elif has_4ch:
            # 单平面4CH
            m=masks4[idx4]
            info=calculator.frame_bounds_and_L(m,m,spacing4,spacing4)
            b4ch=info.get("bounds_4ch")
            b2ch=b4ch  # 单平面时两个平面用相同的bounds
        else:
            vertices_series.append(
                vertices_series[-1] if vertices_series
                else np.zeros((calculator.n*32,3)).tolist()
            )
            continue
        
        h_mm=info.get("h_mm",0)
        
        if (b2ch is None or b4ch is None or h_mm <= 1e-6):
            vertices_series.append(
                vertices_series[-1] if vertices_series
                else np.zeros((calculator.n*32,3)).tolist()
            )
            continue

        try:
            v,_=calculator.generate_3d_mesh_asymmetric(
                b2ch,b4ch,h_mm,
                origin_2ch_mm=origin_ref_2ch,
                axis_u_2ch=axis_ref_2ch
            )
            vertices_series.append(np.asarray(v,dtype=float).tolist())
        except Exception as e:
            print(f"[WARN] 3D frame {t_idx} generation failed: {e}")
            # 使用前一帧的数据，或零填充
            vertices_series.append(
                vertices_series[-1] if vertices_series
                else np.zeros((calculator.n*32,3)).tolist()
            )
    """
    
    return {
        "faces":faces_list,
        "vertices_series":vertices_series,
        "n_frames":len(vertices_series),
        "n_discs":calculator.n,
        "num_theta":32,
    }


def generate_ndjson_response(
    path2ch,path4ch,patient_data:dict,
    algorithm:str="biplane_simpson",
    annulus_strategy:str="auto",
    spacing2_override=None,spacing4_override=None,
    is_video_2ch:bool=False,
    is_video_4ch:bool=False
):
    try:
        yield json.dumps({"progress":5,"status":"文件已保存，读取影像数据..."}) + "\n"
        has2=path2ch is not None and os.path.exists(path2ch)
        has4=path4ch is not None and os.path.exists(path4ch)
        if not has2 and not has4:
            yield json.dumps({"error":"至少需要提供一个视图文件"}) + "\n"
            return
        needs_2ch=algorithm in ("biplane_simpson","singleplane_2ch","area_length_2ch")
        needs_4ch=algorithm in ("biplane_simpson","singleplane_4ch","area_length_4ch")
        if needs_2ch and not has2:
            yield json.dumps({"error":f"算法 [{algorithm}] 需要 2CH 文件"}) + "\n"
            return
        if needs_4ch and not has4:
            yield json.dumps({"error":f"算法 [{algorithm}] 需要 4CH 文件"}) + "\n"
            return
        orig2=nib.load(path2ch) if has2 else None
        orig4=nib.load(path4ch) if has4 else None
        spacing2=spacing2_override or (orig2.header.get_zooms()[:2] if orig2 else (1.0,1.0))
        spacing4=spacing4_override or (orig4.header.get_zooms()[:2] if orig4 else (1.0,1.0))
        yield json.dumps({"progress":15,"status":"AI 图像分割（可能需要几分钟）..."}) + "\n"
        masks2,masks4=[],[]
        if has2:
            _,preds2ch=run_inference(
                path2ch,
                "case_2ch",
                NNUNET_DATASET_2CH,
                is_video=is_video_2ch,
                prefer_onnx=True,
                source_type="video" if is_video_2ch else "camus",
            )
            
            yield json.dumps({"progress":30,"status":"2CH 分割完成..."}) + "\n"
            try:
                spacing2=nib.load(preds2ch[0]).header.get_zooms()[:2]
            except Exception:
                pass
            for p in preds2ch:
                masks2.append(nib.load(p).get_fdata())
        if has4:
            _,preds4ch=run_inference(
                path4ch,
                "case_4ch",
                NNUNET_DATASET_4CH,
                is_video=is_video_4ch,
                prefer_onnx=True,
                source_type="video" if is_video_4ch else "camus",
            )
            
            yield json.dumps({"progress":45,"status":"4CH 分割完成..."}) + "\n"
            try:
                spacing4=nib.load(preds4ch[0]).header.get_zooms()[:2]
            except Exception:
                pass
            for p in preds4ch:
                masks4.append(nib.load(p).get_fdata())
        resolved_strategy=_resolve_strategy(annulus_strategy,masks2,masks4)
        print(f"[INFO] annulus_strategy:请求={annulus_strategy} → 实际={resolved_strategy}")
        yield json.dumps({
            "progress":50,
            "status":f"瓣环定位策略：{resolved_strategy}，开始心功能计算..."
        }) + "\n"
        use_video_rule=is_video_2ch or is_video_4ch
        calculator=BiplaneSimpsonClinical(
            n_discs=20,
            annulus_strategy=resolved_strategy,
            ed_es_mode="robust" if use_video_rule else "simple",
        )
        
        
        yield json.dumps({"progress":55,"status":"计算心功能参数..."}) + "\n"
        all_results=calculator.compute_all_algorithms(
            masks2,masks4,spacing2,spacing4
        )
        result=all_results.get(algorithm)
        if result is None:
            yield json.dumps({"error":f"{algorithm} 计算失败"}) + "\n"
            return
        comparison={}
        for key,r in all_results.items():
            if r is not None:
                comparison[key]={
                    "label":ALGORITHM_LABELS.get(key,key),
                    "EF":round(r["EF"],2),
                    "EDV":round(r["EDV"],2),
                    "ESV":round(r["ESV"],2),
                }
        yield json.dumps({"progress":65,"status":"3D 网格重建..."}) + "\n"
        mesh_3d_series=None
        try:
            mesh_3d_series=_build_3d_series(
                calculator,masks2,masks4,spacing2,spacing4,result
            )
            if mesh_3d_series:
                print(f"[INFO] 3D series:frames={mesh_3d_series['n_frames']}")
        except Exception as e:
            print(f"[WARN] 3D failed:{e}")
            traceback.print_exc()
        yield json.dumps({"progress":80,"status":"生成结果预览图..."}) + "\n"
        overlay_2ch_file=None
        overlay_4ch_file=None
        ts=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            ED_i2=int(result['ED_index'])
            ES_i2=int(result['ES_index'])
            ED_i4=int(result.get('ED_index_4ch',0))
            ES_i4=int(result.get('ES_index_4ch',0))
            orig2_data=orig2.get_fdata() if orig2 else None
            orig4_data=orig4.get_fdata() if orig4 else None
            def _render_view(view_name,masks_v,orig_v,spacing_v,ed_i,es_i,axis_keys):
                fig,axs=plt.subplots(1,2,figsize=(12,6))
                fig.patch.set_facecolor('#111111')
                print(f"[INFO] overlay {view_name}:ED_i={ed_i},ES_i={es_i}")
                print(f"[INFO] overlay {view_name} mask ED area={int(np.sum(_ensure_int_labels(masks_v[ed_i]) == 1))}")
                print(f"[INFO] overlay {view_name} mask ES area={int(np.sum(_ensure_int_labels(masks_v[es_i]) == 1))}")
                for ax,(frame_i,phase) in zip(axs,[(ed_i,"ED"),(es_i,"ES")]):
                    mask=masks_v[frame_i]
                    bg=(orig_v[:,:,frame_i] if orig_v is not None and orig_v.ndim == 3 else orig_v)
                    if bg is not None:
                        ax.imshow(bg.T,cmap='gray',origin='lower')
                    int_mask=_ensure_int_labels(mask)
                    lv_only=(int_mask == 1).astype(np.uint8)
                    ax.imshow(
                        np.ma.masked_where(lv_only.T == 0,lv_only.T),
                        cmap='Reds',alpha=0.5,origin='lower'
                    )
                    lv_wall=(int_mask == 2)
                    if np.any(lv_wall):
                        wall_rgba=np.zeros((lv_wall.T.shape[0],lv_wall.T.shape[1],4),dtype=np.float32)
                        wall_rgba[...,0]=1.0
                        wall_rgba[...,1]=0.0
                        wall_rgba[...,2]=0.0
                        wall_rgba[...,3]=lv_wall.T.astype(np.float32) * 0.35
                        ax.imshow(wall_rgba,origin='lower')
                    la_mask=(int_mask == 3)
                    if np.any(la_mask):
                        la_rgba=np.zeros((la_mask.T.shape[0],la_mask.T.shape[1],4),dtype=np.float32)
                        la_rgba[...,0]=0.0
                        la_rgba[...,1]=1.0
                        la_rgba[...,2]=0.0
                        la_rgba[...,3]=la_mask.T.astype(np.float32) * 0.35
                        ax.imshow(la_rgba,origin='lower')
                    
                    au_key,ap_key,an_key=(
                        axis_keys[:3] if phase == "ED" else axis_keys[3:]
                    )
                    draw_simpson_lines(
                        ax,mask,spacing_v,
                        n_discs=calculator.n,
                        band_frac=calculator.band_frac,
                        min_band_points=calculator.min_band_points,
                        axis_u_override=result.get(au_key),
                        apex_mm=result.get(ap_key),
                        annulus_mid_mm=result.get(an_key),
                    )
                    ax.invert_yaxis()
                    ax.set_title(
                        f"{view_name} {phase} frame {frame_i}",
                        color='white',fontsize=13,fontweight='bold',pad=8
                    )
                    ax.axis('off')
                fig.tight_layout(pad=1.5)
                fname=f"overlay_{view_name.lower()}_{ts}.png"
                fig.savefig(
                    os.path.join(RESULT_FOLDER,fname),
                    bbox_inches='tight',
                    dpi=150,
                    facecolor=fig.get_facecolor()
                )
                plt.close(fig)
                return fname
            
            if masks2:
                overlay_2ch_file=_render_view(
                    "2CH",masks2,orig2_data,spacing2,ED_i2,ES_i2,
                    ("axis_u_2ch_ed","apex_2ch_ed","annulus_mid_2ch_ed",
                     "axis_u_2ch_es","apex_2ch_es","annulus_mid_2ch_es")
                )
            if masks4:
                overlay_4ch_file=_render_view(
                    "4CH",masks4,orig4_data,spacing4,ED_i4,ES_i4,
                    ("axis_u_4ch_ed","apex_4ch_ed","annulus_mid_4ch_ed",
                     "axis_u_4ch_es","apex_4ch_es","annulus_mid_4ch_es")
                )
                
            def _export_video_overlays(view_name,masks_v,orig_v,spacing_v,result,is_video_view):
                if not is_video_view or not masks_v:
                    return None
                ts_dir=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                folder_name=f"overlay_frames_{view_name.lower()}_{ts_dir}"
                folder_path=os.path.join(RESULT_FOLDER,folder_name)
                os.makedirs(folder_path,exist_ok=True)
                print(f"[INFO] 导出视频 overlay:{view_name},共 {len(masks_v)} 帧 -> {folder_path}")
                axis_prefix=view_name.lower()
                for frame_i in range(len(masks_v)):
                    mask=masks_v[frame_i]
                    bg=(orig_v[:,:,frame_i] if orig_v is not None and orig_v.ndim == 3 else orig_v)
                    out_png=os.path.join(folder_path,f"{view_name.lower()}_{frame_i:03d}.png")
                    _save_single_frame_overlay(
                        save_path=out_png,
                        view_name=view_name,
                        frame_i=frame_i,
                        mask=mask,
                        bg=bg,
                        spacing=spacing_v,
                        axis_u_override=None,
                        apex_mm=None,
                        annulus_mid_mm=None,
                        n_discs=calculator.n,
                        band_frac=calculator.band_frac,
                        min_band_points=calculator.min_band_points,
                    )
                import cv2
                def frames_to_video(frame_dir, out_path, fps=20):
                    files = sorted(os.listdir(frame_dir))
                    files = [f for f in files if f.endswith(".png")]
                    first = cv2.imread(os.path.join(frame_dir, files[0]))
                    h, w, _ = first.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    for f in files:
                        img = cv2.imread(os.path.join(frame_dir, f))
                        writer.write(img)
                    writer.release()
                    return out_path
                
                video_filename = f"{view_name.lower()}_{ts_dir}.mp4"
                video_path = os.path.join(RESULT_FOLDER, video_filename)
                frames_to_video(folder_path, video_path)
                return folder_name, video_filename
            
            video_overlay_dirs={}
            if masks2:
                result_2ch=_export_video_overlays(
                    "2CH",masks2,orig2_data,spacing2,result,is_video_2ch
                )
                if result_2ch:
                    folder_2ch, video_2ch = result_2ch
                    video_overlay_dirs["2ch"] = folder_2ch
                    video_overlay_dirs["2ch_video"] = video_2ch
            if masks4:
                result_4ch=_export_video_overlays(
                    "4CH",masks4,orig4_data,spacing4,result,is_video_4ch
                )
                if result_4ch:
                    folder_4ch, video_4ch = result_4ch
                    video_overlay_dirs["4ch"] = folder_4ch
                    video_overlay_dirs["4ch_video"] = video_4ch
                                                    
        except Exception as e:
            print(f"[WARN] Overlay failed:{e}")
            traceback.print_exc()
        yield json.dumps({"progress":90,"status":"保存到数据库..."}) + "\n"
        try:
            patient_uid=patient_data.get("patient_uid") or str(uuid.uuid4())[:8].upper()
            conn=get_db()
            cur=conn.cursor()
            cur.execute("SELECT id FROM patient WHERE patient_uid=%s",(patient_uid,))
            row=cur.fetchone()
            if row:
                patient_id=row[0]
                cur.execute(
                    "UPDATE patient SET name=%s,age=%s,gender=%s WHERE id=%s",
                    (patient_data.get("name"),patient_data.get("age"),patient_data.get("gender"),patient_id)
                )
            else:
                cur.execute(
                    "INSERT INTO patient (patient_uid,name,age,gender) VALUES (%s,%s,%s,%s)",
                    (patient_uid,patient_data.get("name"),patient_data.get("age"),patient_data.get("gender"))
                )
                patient_id=cur.lastrowid
            overlay_db_path=";".join(filter(None,[overlay_2ch_file,overlay_4ch_file]))
            cur.execute(
                """INSERT INTO analysis_record
                   (patient_id,image_path,result_path,lvef,edv,esv,algorithm,view_mode,annulus_strategy)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    patient_id,
                    (path2ch or "") + ";" + (path4ch or ""),
                    overlay_db_path,
                    float(result["EF"]),
                    float(result["EDV"]),
                    float(result["ESV"]),
                    algorithm,
                    "biplane" if (has2 and has4) else ("2ch" if has2 else "4ch"),
                    resolved_strategy
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[WARN] DB save failed:{e}")
            traceback.print_exc()
        yield json.dumps({"progress":100,"status":"分析完成！"}) + "\n"

        # ===== 修复：先保存 mask，再构造返回值 =====
        mask_2ch_path = None
        if masks2:
            try:
                mask_2ch_nii = np.stack(masks2, axis=-1).astype(np.float32)
                #mask_2ch_file = f"mask_2ch_{ts}.nii.gz"
                mask_2ch_file = f"mask_2ch_{ts}.nii"
                mask_2ch_path = os.path.join(RESULT_FOLDER, mask_2ch_file)
                affine = np.eye(4)
                
                nib.save(nib.Nifti1Image(mask_2ch_nii, affine), mask_2ch_path)
                print(f"[INFO] 保存2CH mask: {mask_2ch_path}")
            except Exception as e:
                print(f"[WARN] 保存2CH mask失败: {e}")

        mask_4ch_path = None
        if masks4:
            try:
                mask_4ch_nii = np.stack(masks4, axis=-1).astype(np.float32)
                #mask_4ch_file = f"mask_4ch_{ts}.nii.gz"
                mask_4ch_file = f"mask_4ch_{ts}.nii"
                mask_4ch_path = os.path.join(RESULT_FOLDER, mask_4ch_file)
                affine = np.eye(4)
                nib.save(nib.Nifti1Image(mask_4ch_nii, affine), mask_4ch_path)
                print(f"[INFO] 保存4CH mask: {mask_4ch_path}")
            except Exception as e:
                print(f"[WARN] 保存4CH mask失败: {e}")

        # ===== 构造最终返回（只写一次）=====
        resp_data = {
            "LVEF": round(result["EF"], 2),
            "EDV": round(result["EDV"], 2),
            "ESV": round(result["ESV"], 2),
            "algorithm": algorithm,
            "annulus_strategy": resolved_strategy,
            "comparison": comparison,
            "mesh_3d_series": mesh_3d_series,
            "mask_path_2ch": f"{BASE_URL}/results/{os.path.basename(mask_2ch_path)}" if mask_2ch_path else None,
            "mask_path_4ch": f"{BASE_URL}/results/{os.path.basename(mask_4ch_path)}" if mask_4ch_path else None,
        }

        if overlay_2ch_file:
            resp_data["overlay_2ch_url"]=f"{BASE_URL}/results/{overlay_2ch_file}"
        if overlay_4ch_file:
            resp_data["overlay_4ch_url"]=f"{BASE_URL}/results/{overlay_4ch_file}"
        if video_overlay_dirs:
            resp_data["video_overlay_dirs"] = {}
            for k, v in video_overlay_dirs.items():
                resp_data["video_overlay_dirs"][k] = f"{BASE_URL}/results/{v}"
       
        yield json.dumps({"result":resp_data}) + "\n"
        
        print("[INFO] 分析完成，清理临时文件...")
        for path in [path2ch, path4ch]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[INFO] 已删除临时文件: {path}")
                except Exception as e:
                    print(f"[WARN] 删除临时文件失败 {path}: {e}")
    except Exception as e:
        traceback.print_exc()
        yield json.dumps({"error":str(e)}) + "\n"




# 路由
@app.route('/login',methods=['POST'])
def login():
    data=request.get_json() or {}
    conn=get_db()
    cur=conn.cursor()
    cur.execute(
        "SELECT id FROM user WHERE username=%s AND password=%s",
        (data.get("username"),data.get("password"))
    )
    user=cur.fetchone()
    conn.close()

    if not user:
        return jsonify({"error":"Invalid credentials"}),401

    payload={
        "user_id":user[0],
        "exp":datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }
    token=jwt.encode(payload,SECRET_KEY,algorithm="HS256")
    if isinstance(token,bytes):
        token=token.decode("utf-8")
    return jsonify({"token":token})

"""
@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER,filename)
"""
@app.route('/results/<path:filename>')
def serve_results(filename):
    full_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(full_path):
        abort(404)

    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        with open(full_path, "rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            app.logger.error(f"[BAD FILE] fake .nii.gz detected: {full_path}")
            abort(500, description=f"Invalid gzip NIfTI file: {filename}")

    if lower.endswith(".nii"):
        return send_file(full_path, mimetype="application/octet-stream")
    if lower.endswith(".nii.gz"):
        return send_file(full_path, mimetype="application/gzip")

    return send_from_directory(RESULT_FOLDER, filename)



@app.route('/algorithms',methods=['GET'])
def get_algorithms():
    return jsonify([{"key":k,"label":v} for k,v in ALGORITHM_LABELS.items()])


@app.route('/detect_view',methods=['POST'])
@token_required
def detect_view():
    """
    自动检测上传文件的视图类型（2CH / 4CH / unknown）
    
    请求参数:
        - file: 上传的图像/视频文件（NIfTI, AVI, MP4, PNG, JPG等）
        - threshold: 可选，置信度阈值（默认0.7）
    
    返回:
        {
            "view_type": "2ch" | "4ch" | "unknown",
            "confidence": 0.95,
            "prob_2ch": 0.05,
            "prob_4ch": 0.95,
            "is_reliable": true,
            "threshold": 0.7
        }
    """
    try:
        from inference.view_classifier import CONFIDENCE_THRESHOLD
        
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({"error":"请提供文件"}),400
        
        # 获取阈值参数
        threshold = request.form.get('threshold', CONFIDENCE_THRESHOLD)
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            threshold = CONFIDENCE_THRESHOLD
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"detect_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}")
        file.save(temp_path)
        
        try:
            # 处理视频/DICOM等需要转换的文件
            if _is_video(temp_path):
                nii_path = temp_path + "_converted.nii.gz"
                video_to_nifti(temp_path, nii_path)
                os.remove(temp_path)  # 删除原始视频
                temp_path = nii_path
            elif _is_dicom(temp_path):
                nii_path = temp_path + "_converted.nii.gz"
                dicom_to_nifti(temp_path, nii_path)
                os.remove(temp_path)
                temp_path = nii_path
            
            # 执行视图分类
            result = classify_view(temp_path, threshold=threshold)
            
            # 添加文件信息
            result['filename'] = filename
            result['file_size'] = os.path.getsize(temp_path)
            
            return jsonify(result)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/detect_view_batch',methods=['POST'])
@token_required
def detect_view_batch():
    """
    批量检测视图类型（用于同时上传多个文件）
    
    请求参数:
        - files: 多个上传文件
        - auto_assign: 是否自动分配2CH/4CH（默认true）
        - threshold: 置信度阈值
    
    返回:
        {
            "classifications": [...],
            "assigned": {
                "2ch": "path/to/file1",
                "4ch": "path/to/file2"
            },
            "warnings": [...]
        }
    """
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error":"请提供文件"}),400
        
        auto_assign = request.form.get('auto_assign', 'true').lower() == 'true'
        threshold = float(request.form.get('threshold', 0.7))
        
        # 保存所有文件
        saved_paths = []
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        for i, file in enumerate(files):
            if not file or not file.filename:
                continue
            filename = secure_filename(file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, f"batch_{timestamp}_{i}_{filename}")
            file.save(temp_path)
            saved_paths.append((temp_path, filename))
        
        if not saved_paths:
            return jsonify({"error":"没有有效文件"}),400
        
        try:
            # 转换需要处理的文件
            processed_paths = []
            for path, orig_name in saved_paths:
                try:
                    if _is_video(path):
                        nii_path = path + "_converted.nii.gz"
                        video_to_nifti(path, nii_path)
                        os.remove(path)
                        processed_paths.append((nii_path, orig_name))
                    elif _is_dicom(path):
                        nii_path = path + "_converted.nii.gz"
                        dicom_to_nifti(path, nii_path)
                        os.remove(path)
                        processed_paths.append((nii_path, orig_name))
                    else:
                        processed_paths.append((path, orig_name))
                except Exception as e:
                    print(f"[WARN] Failed to process {orig_name}: {e}")
                    processed_paths.append((path, orig_name))
            
            # 批量分类
            from inference.view_classifier import classify_views_batch
            
            results = []
            for path, orig_name in processed_paths:
                result = classify_view(path, threshold=threshold)
                result['filename'] = orig_name
                result['path'] = path
                results.append(result)
            
            response = {
                "classifications": results,
                "total": len(results),
                "reliable_count": sum(1 for r in results if r.get('is_reliable'))
            }
            
            # 自动分配视图
            if auto_assign and len(results) >= 1:
                ch2_candidates = [r for r in results if r['view_type'] == '2ch']
                ch4_candidates = [r for r in results if r['view_type'] == '4ch']
                
                # 按置信度排序
                ch2_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                ch4_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                
                assigned = {}
                warnings = []
                
                if ch2_candidates:
                    assigned['2ch'] = {
                        'filename': ch2_candidates[0]['filename'],
                        'confidence': ch2_candidates[0]['confidence']
                    }
                if ch4_candidates:
                    assigned['4ch'] = {
                        'filename': ch4_candidates[0]['filename'],
                        'confidence': ch4_candidates[0]['confidence']
                    }
                
                # 生成警告
                if len(results) == 2:
                    types = [r['view_type'] for r in results]
                    if types[0] == types[1]:
                        if types[0] == 'unknown':
                            warnings.append("两个文件都无法被可靠识别")
                        else:
                            warnings.append(f"两个文件都被识别为 {types[0].upper()}")
                
                response['assigned'] = assigned
                response['warnings'] = warnings
            
            return jsonify(response)
            
        finally:
            # 清理临时文件
            for path, _ in processed_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/history',methods=['GET'])
@token_required
def get_history():
    try:
        conn=get_db()
        cur=conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT ar.id,p.patient_uid,p.name,p.age,p.gender,
                   ar.create_time,ar.lvef,ar.edv,ar.esv,ar.result_path,
                   ar.algorithm,ar.view_mode,ar.annulus_strategy
            FROM analysis_record ar
            JOIN patient p ON ar.patient_id=p.id
            ORDER BY ar.create_time DESC
        """)
        records=cur.fetchall()
        conn.close()

        for r in records:
            if r.get('create_time'):
                r['create_time']=r['create_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(records)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/history/trend/<patient_uid>',methods=['GET'])
@token_required
def get_patient_trend(patient_uid):
    try:
        conn=get_db()
        cur=conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT ar.create_time,ar.lvef,ar.edv,ar.esv,
                   ar.algorithm,ar.view_mode,ar.annulus_strategy,ar.result_path,
                   p.name,p.age,p.gender,p.patient_uid
            FROM analysis_record ar
            JOIN patient p ON ar.patient_id=p.id
            WHERE p.patient_uid=%s
            ORDER BY ar.create_time ASC
        """,(patient_uid,))
        rows=cur.fetchall()

        cur.execute(
            "SELECT name,age,gender,patient_uid FROM patient WHERE patient_uid=%s",
            (patient_uid,)
        )
        p_info=cur.fetchone()
        conn.close()

        for r in rows:
            if r.get('create_time'):
                r['create_time']=r['create_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({"patient":p_info,"records":rows})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/patients',methods=['GET'])
@token_required
def get_patients():
    try:
        conn=get_db()
        cur=conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT p.patient_uid,p.name,p.age,p.gender,
                   COUNT(ar.id) AS record_count,MAX(ar.create_time) AS last_time
            FROM patient p
            LEFT JOIN analysis_record ar ON ar.patient_id=p.id
            GROUP BY p.id
            ORDER BY last_time DESC
        """)
        rows=cur.fetchall()
        conn.close()

        for r in rows:
            if r.get('last_time'):
                r['last_time']=r['last_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(rows)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/analyze',methods=['POST'])
@token_required
def analyze():
    try:
        file_2ch=request.files.get('file_2ch')
        file_4ch=request.files.get('file_4ch')
        
        # 支持通用文件上传（不指定视图类型）
        files = request.files.getlist('files')
        
        if not file_2ch and not file_4ch and not files:
            return jsonify({"error":"至少需要提供一个视图文件"}),400

        algorithm=request.form.get("algorithm","biplane_simpson")
        annulus_strategy=request.form.get("annulus_strategy","auto")
        patient_uid=request.form.get("patient_uid","").strip()
        
        # 是否启用自动视图检测
        auto_detect = request.form.get("auto_detect","false").lower() == "true"

        age_raw=request.form.get("age","").strip()
        patient_data={
            "patient_uid":patient_uid,
            "name":request.form.get("name","").strip(),
            "age":int(age_raw) if age_raw else None,
            "gender":request.form.get("gender","").strip(),
        }

        path2ch=path4ch=None
        spacing2_override=spacing4_override=None
        is_video_2ch=False
        is_video_4ch=False
        
        # ========== 自动视图检测模式 ==========
        if auto_detect:
            print("[INFO] Auto view detection enabled")
            
            # 收集所有上传的文件
            all_files = []
            print(f"[AUTO DETECT] file_2ch: {file_2ch}, filename: {file_2ch.filename if file_2ch else None}")
            print(f"[AUTO DETECT] file_4ch: {file_4ch}, filename: {file_4ch.filename if file_4ch else None}")
            print(f"[AUTO DETECT] files list: {len(files)} items")
            
            if file_2ch and file_2ch.filename:
                all_files.append(('user_2ch', file_2ch))
            if file_4ch and file_4ch.filename:
                all_files.append(('user_4ch', file_4ch))
            for f in files:
                if f and f.filename:
                    all_files.append(('auto', f))
            
            print(f"[AUTO DETECT] Total files collected: {len(all_files)}")
            
            if len(all_files) == 0:
                return jsonify({"error":"没有有效文件"}),400
            
            # 保存所有文件
            saved_files = []
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            
            for label, f in all_files:
                filename = secure_filename(f.filename)
                prefix = f"auto_{timestamp}_"
                temp_path = os.path.join(UPLOAD_FOLDER, prefix + filename)
                f.save(temp_path)
                saved_files.append((label, temp_path, filename))
            
            # 转换文件格式
            converted_files = []
            for label, path, orig_name in saved_files:
                try:
                    if _is_video(path):
                        nii_path = path + "_converted.nii.gz"
                        _, spacing = video_to_nifti(path, nii_path)
                        os.remove(path)
                        converted_files.append((label, nii_path, orig_name, spacing, True))
                    elif _is_dicom(path):
                        nii_path = path + "_converted.nii.gz"
                        _, spacing = dicom_to_nifti(path, nii_path)
                        os.remove(path)
                        converted_files.append((label, nii_path, orig_name, spacing, False))
                    else:
                        converted_files.append((label, path, orig_name, None, False))
                except Exception as e:
                    print(f"[WARN] Failed to convert {orig_name}: {e}")
                    converted_files.append((label, path, orig_name, None, False))
            
            # 执行自动分类
            from inference.view_classifier import CONFIDENCE_THRESHOLD
            threshold = float(request.form.get('threshold', CONFIDENCE_THRESHOLD))
            
            classifications = []
            print(f"[AUTO DETECT] Classifying {len(converted_files)} files...")
            for label, path, orig_name, spacing, is_video in converted_files:
                print(f"[AUTO DETECT] Classifying: {orig_name} ({label})")
                result = classify_view(path, threshold=threshold)
                result['path'] = path
                result['filename'] = orig_name
                result['spacing'] = spacing
                result['is_video'] = is_video
                classifications.append(result)
                print(f"[AUTO DETECT]   -> {result['view_class']} ({result['view_type']}) @ {result['confidence']:.2%}")
            
            # 分配视图
            ch2_candidates = [c for c in classifications if c['view_type'] == '2ch']
            ch4_candidates = [c for c in classifications if c['view_type'] == '4ch']
            print(f"[AUTO DETECT] Found {len(ch2_candidates)} 2CH, {len(ch4_candidates)} 4CH")
            
            ch2_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            ch4_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            auto_warnings = []
            
            if ch2_candidates:
                best = ch2_candidates[0]
                path2ch = best['path']
                spacing2_override = best.get('spacing')
                is_video_2ch = best.get('is_video', False)
                print(f"[AUTO DETECT] 2CH assigned: {best['filename']} (conf: {best['confidence']:.2%})")
            
            if ch4_candidates:
                best = ch4_candidates[0]
                path4ch = best['path']
                spacing4_override = best.get('spacing')
                is_video_4ch = best.get('is_video', False)
                print(f"[AUTO DETECT] 4CH assigned: {best['filename']} (conf: {best['confidence']:.2%})")
            
            # 检查问题
            if not path2ch and not path4ch:
                auto_warnings.append("无法可靠识别任何视图（2CH或4CH），请确保上传心脏超声图像")
            elif not path2ch:
                auto_warnings.append("未检测到2CH视图")
            elif not path4ch:
                auto_warnings.append("未检测到4CH视图")
            
            # 清理未使用的文件
            used_paths = {path2ch, path4ch}
            for c in classifications:
                if c['path'] not in used_paths:
                    try:
                        os.remove(c['path'])
                    except:
                        pass
            
            if auto_warnings:
                print(f"[AUTO DETECT] Warnings: {auto_warnings}")
        
        # ========== 传统模式（用户指定视图类型） ==========
        else:
            if file_2ch and file_2ch.filename:
                path2ch,sp2,is_video_2ch=save_upload(file_2ch,UPLOAD_FOLDER,"2ch_")
                if sp2:
                    spacing2_override=sp2

            if file_4ch and file_4ch.filename:
                path4ch,sp4,is_video_4ch=save_upload(file_4ch,UPLOAD_FOLDER,"4ch_")
                if sp4:
                    spacing4_override=sp4
        
        # 检查是否有所需的视图
        needs_2ch = algorithm in ("biplane_simpson","singleplane_2ch","area_length_2ch")
        needs_4ch = algorithm in ("biplane_simpson","singleplane_4ch","area_length_4ch")
        
        # 自动降级：如果双平面算法缺少某个视图，自动切换到单平面
        if algorithm == "biplane_simpson":
            if path2ch and not path4ch:
                algorithm = "singleplane_2ch"
                auto_warnings.append("未检测到4CH视图，自动切换为单平面(2CH)算法")
                print(f"[AUTO SWITCH] biplane_simpson -> singleplane_2ch (缺少4CH)")
            elif path4ch and not path2ch:
                algorithm = "singleplane_4ch"
                auto_warnings.append("未检测到2CH视图，自动切换为单平面(4CH)算法")
                print(f"[AUTO SWITCH] biplane_simpson -> singleplane_4ch (缺少2CH)")
        
        # 检查必需的视图是否存在
        needs_2ch = algorithm in ("biplane_simpson","singleplane_2ch","area_length_2ch")
        needs_4ch = algorithm in ("biplane_simpson","singleplane_4ch","area_length_4ch")
        
        if needs_2ch and not path2ch:
            return jsonify({"error":f"算法 [{algorithm}] 需要 2CH 文件"}),400
        if needs_4ch and not path4ch:
            return jsonify({"error":f"算法 [{algorithm}] 需要 4CH 文件"}),400

        return Response(
            generate_ndjson_response(
                path2ch,path4ch,patient_data,
                algorithm,annulus_strategy,
                spacing2_override,spacing4_override,
                is_video_2ch=is_video_2ch,
                is_video_4ch=is_video_4ch
            ),
            mimetype='application/x-ndjson'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


if __name__ == '__main__':
    _ensure_db_schema()
    app.run(host="0.0.0.0",port=5000,debug=True,use_reloader=False)