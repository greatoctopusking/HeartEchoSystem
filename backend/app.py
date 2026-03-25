"""
Flask 后端 — 心脏超声智能分析系统
修正版说明：
  · 支持 annulus_strategy 参数（"polar" / "wall_la" / "auto"）
  · 支持 AVI / MP4 / MOV 输入
  · 视频自动转换为 NIfTI(H, W, T) 后进入 nnUNet 推理
  · 3D series 统一调用 align_series_indices（全序列重采样）
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
from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

from config import *
from model_infer import run_inference
from biplane_simpson_clinical import BiplaneSimpsonClinical, ALGORITHM_LABELS

# 视频处理
import cv2


UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
BASE_URL = "http://127.0.0.1:5000"

# 修正：补全视频扩展名
SUPPORTED_EXT = {
    ".nii", ".gz",
    ".dcm", ".dicom", ".ima", ".img",
    ".avi", ".mp4", ".mov"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)

LABEL_MAP = {0: "background", 1: "LV", 2: "LVWall", 3: "LA"}


# ──────────────────────────────────────────────────────────────────
#  数据库
# ──────────────────────────────────────────────────────────────────
def get_db(use_database=True):
    conn_kwargs = dict(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=False,
    )
    if use_database:
        conn_kwargs["database"] = MYSQL_DB
    return pymysql.connect(**conn_kwargs)


def _ensure_db_schema():
    """
    启动时自动完成：
    1. 数据库不存在则创建
    2. 表不存在则创建
    不破坏现有数据
    """
    conn = get_db(use_database=False)
    cur = conn.cursor()

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


# ──────────────────────────────────────────────────────────────────
#  JWT 鉴权
# ──────────────────────────────────────────────────────────────────
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

def _normalize_ultrasound_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if img.size == 0:
        return img.astype(np.float32)

    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)

    vals = img[finite]
    p1, p99 = np.percentile(vals, [1, 99])

    if p99 <= p1:
        mn, mx = float(vals.min()), float(vals.max())
        if mx <= mn:
            return np.zeros_like(img, dtype=np.float32)
        out = (img - mn) / (mx - mn)
    else:
        out = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)

    return (out * 255.0).astype(np.float32)


def _crop_to_ultrasound_content(frame: np.ndarray, pad: int = 8) -> np.ndarray:
    img = np.asarray(frame)
    if img.ndim != 2:
        return img

    thr = max(5, int(np.percentile(img, 60) * 0.15))
    fg = img > thr
    if fg.sum() < img.size * 0.01:
        return img

    ys, xs = np.where(fg)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)

    cropped = img[y0:y1, x0:x1]
    if cropped.size == 0:
        return img
    return cropped
# ──────────────────────────────────────────────────────────────────
#  DICOM 支持
# ──────────────────────────────────────────────────────────────────
def _is_dicom(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".dcm", ".dicom", ".ima"):
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def load_dicom_to_array(path: str):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    def _postprocess_dcm_pixels(ds):
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr

        return _normalize_ultrasound_image(arr)

    if os.path.isdir(path):
        dcm_files = sorted(
            [f for f in os.listdir(path) if not f.startswith(".")],
            key=lambda f: int(
                pydicom.dcmread(
                    os.path.join(path, f), stop_before_pixels=True
                ).get("InstanceNumber", 0)
            )
        )
        slices = [pydicom.dcmread(os.path.join(path, f)) for f in dcm_files]
        data = np.stack([_postprocess_dcm_pixels(s) for s in slices], axis=-1)
        ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    else:
        ds = pydicom.dcmread(path)
        n_frames = int(getattr(ds, "NumberOfFrames", 1))
        arr = _postprocess_dcm_pixels(ds)
        if n_frames > 1:
            if arr.ndim == 3:
                data = np.transpose(arr, (1, 2, 0))
            else:
                data = arr[:, :, np.newaxis]
        else:
            data = arr[:, :, np.newaxis]
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])

    return data.astype(np.float32), (float(ps[1]), float(ps[0]))

def dicom_to_nifti(path: str, out_path: str):
    data, spacing = load_dicom_to_array(path)
    affine = np.diag([spacing[0], spacing[1], 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((*spacing, 1.0))
    nib.save(img, out_path)
    return out_path, spacing


# ──────────────────────────────────────────────────────────────────
#  视频支持
# ──────────────────────────────────────────────────────────────────
def _is_video(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in (".avi", ".mp4", ".mov")


def video_to_nifti(path: str, out_path: str, max_frames: int = 80):
    """
    将视频转换为 NIfTI (H, W, T)。
    对普通视频文件默认 spacing=(1.0, 1.0)。

    优化：
    - 若视频帧数过多，则均匀采样到 max_frames 帧，减少 nnUNet 推理耗时。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"视频文件不存在: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                continue

            if frame.ndim == 2:
                gray = frame
            elif frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue

            gray = _crop_to_ultrasound_content(gray)
            gray = _normalize_ultrasound_image(gray)
            frames.append(gray.astype(np.float32))
    finally:
        cap.release()

    if not frames:
        raise ValueError("无法读取视频内容，未提取到有效帧")

    # 限帧：视频太长时，均匀采样
    total_frames = len(frames)
    if max_frames is not None and total_frames > max_frames:
        sample_idx = np.linspace(0, total_frames - 1, max_frames).astype(int)
        frames = [frames[i] for i in sample_idx]
        print(f"[INFO] 视频原始帧数={total_frames}，已均匀采样到 {len(frames)} 帧")
    else:
        print(f"[INFO] 视频帧数={total_frames}，无需采样")

    h0, w0 = frames[0].shape
    aligned_frames = []
    for fr in frames:
        if fr.shape != (h0, w0):
            fr = cv2.resize(fr, (w0, h0), interpolation=cv2.INTER_LINEAR)
        aligned_frames.append(fr.astype(np.float32))

    data = np.stack(aligned_frames, axis=-1)

    spacing = (1.0, 1.0)
    affine = np.diag([spacing[0], spacing[1], 1.0, 1.0])

    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((*spacing, 1.0))
    nib.save(img, out_path)

    return out_path, spacing


def save_upload(fileobj, dest_dir: str, prefix: str = ""):
    filename = secure_filename(fileobj.filename or "")
    if not filename:
        raise ValueError("上传文件名为空")

    # 兼容 .nii.gz 的简单后缀检查
    lower_name = filename.lower()
    if not any(lower_name.endswith(ext) for ext in SUPPORTED_EXT):
        raise ValueError(f"不支持的文件类型: {filename}")

    raw_path = os.path.join(dest_dir, prefix + filename)
    fileobj.save(raw_path)

    if _is_dicom(raw_path):
        nii_path = raw_path + "_converted.nii.gz"
        _, spacing = dicom_to_nifti(raw_path, nii_path)
        return nii_path, spacing

    if _is_video(raw_path):
        nii_path = raw_path + "_converted.nii.gz"
        _, spacing = video_to_nifti(raw_path, nii_path)
        return nii_path, spacing

    return raw_path, None


# ──────────────────────────────────────────────────────────────────
#  自动检测 annulus 策略
# ──────────────────────────────────────────────────────────────────
def _detect_annulus_strategy(
    masks: list,
    min_frames: int = 3,
    min_pixels_wall: int = 30,
    min_pixels_la: int = 30,
) -> str:
    """
    自动检测 annulus 策略：
    只有当至少 min_frames 帧中，
    Wall(2) 和 LA(3) 都达到最小像素数时，才启用 wall_la；
    否则使用 polar。
    """
    valid_count = 0

    for m in masks:
        arr = np.rint(np.asarray(m)).astype(np.int16)

        n2 = int(np.sum(arr == 2))
        n3 = int(np.sum(arr == 3))

        if n2 >= min_pixels_wall and n3 >= min_pixels_la:
            valid_count += 1
            if valid_count >= min_frames:
                return "wall_la"

    return "polar"


def _resolve_strategy(requested: str, masks2: list, masks4: list) -> str:
    """将 'auto' 解析为实际策略，其余直接透传。"""
    if requested != "auto":
        return requested
    all_masks = list(masks2) + list(masks4)
    return _detect_annulus_strategy(all_masks) if all_masks else "polar"


# ──────────────────────────────────────────────────────────────────
#  Overlay 图（Simpson 线条叠加）
# ──────────────────────────────────────────────────────────────────
def _ensure_int_labels(mask_arr):
    arr = np.asarray(mask_arr)
    return arr if arr.dtype.kind in ("i", "u") else np.rint(arr).astype(np.int16)


def draw_simpson_lines(
    ax, mask, spacing, n_discs=20,
    band_frac=1.0, min_band_points=3,
    axis_u_override=None, apex_mm=None, annulus_mid_mm=None
):
    dx, dy = float(spacing[0]), float(spacing[1])

    def to_disp(xp, yp):
        return yp, xp

    m = (_ensure_int_labels(mask) == 1)
    coords = np.column_stack(np.where(m))
    if coords.shape[0] < 30:
        return

    pts = np.column_stack([
        coords[:, 1].astype(float) * dx,
        coords[:, 0].astype(float) * dy
    ])
    mean_pt = pts.mean(axis=0)

    if axis_u_override is not None:
        axis_u = np.asarray(axis_u_override, dtype=float)
    elif apex_mm is not None and annulus_mid_mm is not None:
        axis_u = np.asarray(apex_mm) - np.asarray(annulus_mid_mm)
    else:
        X = pts - mean_pt
        C = (X.T @ X) / max(1, X.shape[0] - 1)
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
        p1, p2 = origin_pt, apex_pt
        tmin_loop = 0.0
    else:
        centered = pts - mean_pt
        t = centered @ axis_u
        s = centered @ perp_u
        tmin, tmax = float(np.min(t)), float(np.max(t))
        L = tmax - tmin
        if L <= 1e-6:
            return
        p1 = mean_pt + axis_u * tmin
        p2 = mean_pt + axis_u * tmax
        tmin_loop = tmin

    if L <= 1e-6:
        return

    ax.plot(
        [to_disp(p1[0] / dx, p1[1] / dy)[0], to_disp(p2[0] / dx, p2[1] / dy)[0]],
        [to_disp(p1[0] / dx, p1[1] / dy)[1], to_disp(p2[0] / dx, p2[1] / dy)[1]],
        color='cyan', linewidth=2
    )

    h_disc = L / n_discs
    centers = tmin_loop + (np.arange(n_discs) + 0.5) * h_disc
    band_half = 0.5 * band_frac * h_disc
    origin_pt_draw = np.asarray(annulus_mid_mm, float) if annulus_mid_mm is not None else mean_pt

    for c in centers:
        band = (t >= c - band_half) & (t <= c + band_half)
        if np.count_nonzero(band) < min_band_points:
            continue
        smin, smax = float(np.min(s[band])), float(np.max(s[band]))
        if smax <= smin:
            continue

        cpt = origin_pt_draw + axis_u * c
        a = cpt + perp_u * smin
        b = cpt + perp_u * smax

        ax.plot(
            [to_disp(a[0] / dx, a[1] / dy)[0], to_disp(b[0] / dx, b[1] / dy)[0]],
            [to_disp(a[0] / dx, a[1] / dy)[1], to_disp(b[0] / dx, b[1] / dy)[1]],
            color='yellow', linewidth=1.0, alpha=0.75
        )


# ──────────────────────────────────────────────────────────────────
#  3D Mesh 生成
# ──────────────────────────────────────────────────────────────────
def _build_3d_series(calculator, masks2, masks4, spacing2, spacing4, result):
    ed_2ch = int(result['ED_index'])
    es_2ch = int(result['ES_index'])
    ed_4ch = int(result.get('ED_index_4ch', 0))
    es_4ch = int(result.get('ES_index_4ch', 0))

    T2 = len(masks2) if masks2 else 0
    T4 = len(masks4) if masks4 else 0

    aligned_2ch, aligned_4ch = calculator.align_series_indices(
        ed_2ch, es_2ch, max(T2, 1),
        ed_4ch, es_4ch, max(T4, 1)
    )
    T = len(aligned_2ch)
    if T <= 0:
        return None

    if masks2 and masks4:
        info_ref = calculator.frame_bounds_and_L(
            masks2[ed_2ch], masks4[ed_4ch], spacing2, spacing4
        )
    else:
        m2_ref = masks2[aligned_2ch[0]] if masks2 else np.zeros((10, 10), dtype=np.int16)
        m4_ref = masks4[aligned_4ch[0]] if masks4 else np.zeros((10, 10), dtype=np.int16)
        info_ref = calculator.frame_bounds_and_L(m2_ref, m4_ref, spacing2, spacing4)

    origin_ref_2ch = info_ref.get("origin_2ch_mm", None)
    axis_ref_2ch = info_ref.get("axis_u_2ch", None)

    if origin_ref_2ch is None or axis_ref_2ch is None:
        m2_ref0 = masks2[aligned_2ch[0]] if masks2 else np.zeros((10, 10), dtype=np.int16)
        m4_ref0 = masks4[aligned_4ch[0]] if masks4 else np.zeros((10, 10), dtype=np.int16)
        info_ref0 = calculator.frame_bounds_and_L(m2_ref0, m4_ref0, spacing2, spacing4)
        origin_ref_2ch = info_ref0.get("origin_2ch_mm", None)
        axis_ref_2ch = info_ref0.get("axis_u_2ch", None)

    m2_0 = masks2[aligned_2ch[0]] if masks2 else np.zeros((10, 10), dtype=np.int16)
    m4_0 = masks4[aligned_4ch[0]] if masks4 else np.zeros((10, 10), dtype=np.int16)
    info0 = calculator.frame_bounds_and_L(m2_0, m4_0, spacing2, spacing4)
    if info0["bounds_2ch"] is None or info0["bounds_4ch"] is None:
        return None

    _, f0 = calculator.generate_3d_mesh_asymmetric(
        info0["bounds_2ch"], info0["bounds_4ch"], info0["h_mm"]
    )
    faces_list = np.asarray(f0, dtype=int).tolist()

    vertices_series = []
    for t_idx in range(T):
        idx2 = aligned_2ch[t_idx] if masks2 else 0
        idx4 = aligned_4ch[t_idx] if masks4 else 0
        m2 = masks2[idx2] if masks2 else np.zeros((10, 10), dtype=np.int16)
        m4 = masks4[idx4] if masks4 else np.zeros((10, 10), dtype=np.int16)

        info = calculator.frame_bounds_and_L(m2, m4, spacing2, spacing4)
        if (info["bounds_2ch"] is None or info["bounds_4ch"] is None
                or info["h_mm"] <= 1e-6):
            vertices_series.append(
                vertices_series[-1] if vertices_series
                else np.zeros((calculator.n * 32, 3)).tolist()
            )
            continue

        v, _ = calculator.generate_3d_mesh_asymmetric(
            info["bounds_2ch"],
            info["bounds_4ch"],
            info["h_mm"],
            origin_2ch_mm=origin_ref_2ch,
            axis_u_2ch=axis_ref_2ch
        )
        vertices_series.append(np.asarray(v, dtype=float).tolist())

    return {
        "faces": faces_list,
        "vertices_series": vertices_series,
        "n_frames": len(vertices_series),
        "n_discs": calculator.n,
        "num_theta": 32,
    }


# ──────────────────────────────────────────────────────────────────
#  流式生成器
# ──────────────────────────────────────────────────────────────────
def generate_ndjson_response(
    path2ch, path4ch, patient_data: dict,
    algorithm: str = "biplane_simpson",
    annulus_strategy: str = "auto",
    spacing2_override=None, spacing4_override=None
):
    try:
        yield json.dumps({"progress": 5, "status": "文件已保存，读取影像数据..."}) + "\n"

        has2 = path2ch is not None and os.path.exists(path2ch)
        has4 = path4ch is not None and os.path.exists(path4ch)
        if not has2 and not has4:
            yield json.dumps({"error": "至少需要提供一个视图文件"}) + "\n"
            return

        needs_2ch = algorithm in ("biplane_simpson", "singleplane_2ch", "area_length_2ch")
        needs_4ch = algorithm in ("biplane_simpson", "singleplane_4ch", "area_length_4ch")
        if needs_2ch and not has2:
            yield json.dumps({"error": f"算法 [{algorithm}] 需要 2CH 文件"}) + "\n"
            return
        if needs_4ch and not has4:
            yield json.dumps({"error": f"算法 [{algorithm}] 需要 4CH 文件"}) + "\n"
            return

        orig2 = nib.load(path2ch) if has2 else None
        orig4 = nib.load(path4ch) if has4 else None
        spacing2 = spacing2_override or (orig2.header.get_zooms()[:2] if orig2 else (1.0, 1.0))
        spacing4 = spacing4_override or (orig4.header.get_zooms()[:2] if orig4 else (1.0, 1.0))

        yield json.dumps({"progress": 15, "status": "AI 图像分割（可能需要几分钟）..."}) + "\n"

        masks2, masks4 = [], []

        if has2:
            _, preds2ch = run_inference(path2ch, "case_2ch", NNUNET_DATASET_2CH)
            yield json.dumps({"progress": 30, "status": "2CH 分割完成..."}) + "\n"
            try:
                spacing2 = nib.load(preds2ch[0]).header.get_zooms()[:2]
            except Exception:
                pass
            for p in preds2ch:
                masks2.append(nib.load(p).get_fdata())

        if has4:
            _, preds4ch = run_inference(path4ch, "case_4ch", NNUNET_DATASET_4CH)
            yield json.dumps({"progress": 45, "status": "4CH 分割完成..."}) + "\n"
            try:
                spacing4 = nib.load(preds4ch[0]).header.get_zooms()[:2]
            except Exception:
                pass
            for p in preds4ch:
                masks4.append(nib.load(p).get_fdata())

        resolved_strategy = _resolve_strategy(annulus_strategy, masks2, masks4)
        print(f"[INFO] annulus_strategy: 请求={annulus_strategy} → 实际={resolved_strategy}")
        yield json.dumps({
            "progress": 50,
            "status": f"瓣环定位策略：{resolved_strategy}，开始心功能计算..."
        }) + "\n"

        calculator = BiplaneSimpsonClinical(
            n_discs=20,
            annulus_strategy=resolved_strategy,
        )

        yield json.dumps({"progress": 55, "status": "计算心功能参数..."}) + "\n"

        # 统一入口：只算一次所有可用算法
        all_results = calculator.compute_all_algorithms(
            masks2, masks4, spacing2, spacing4
        )

        result = all_results.get(algorithm)
        if result is None:
            yield json.dumps({"error": f"{algorithm} 计算失败"}) + "\n"
            return

        comparison = {}
        for key, r in all_results.items():
            if r is not None:
                comparison[key] = {
                    "label": ALGORITHM_LABELS.get(key, key),
                    "EF": round(r["EF"], 2),
                    "EDV": round(r["EDV"], 2),
                    "ESV": round(r["ESV"], 2),
                }

        yield json.dumps({"progress": 65, "status": "3D 网格重建..."}) + "\n"
        mesh_3d_series = None
        try:
            mesh_3d_series = _build_3d_series(
                calculator, masks2, masks4, spacing2, spacing4, result
            )
            if mesh_3d_series:
                print(f"[INFO] 3D series: frames={mesh_3d_series['n_frames']}")
        except Exception as e:
            print(f"[WARN] 3D failed: {e}")
            traceback.print_exc()

        yield json.dumps({"progress": 80, "status": "生成结果预览图..."}) + "\n"
        overlay_2ch_file = None
        overlay_4ch_file = None
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        try:
            ED_i2 = int(result['ED_index'])
            ES_i2 = int(result['ES_index'])
            ED_i4 = int(result.get('ED_index_4ch', 0))
            ES_i4 = int(result.get('ES_index_4ch', 0))
            orig2_data = orig2.get_fdata() if orig2 else None
            orig4_data = orig4.get_fdata() if orig4 else None

            #def _render_view(view_name, masks_v, orig_v, spacing_v, ed_i, es_i, axis_keys):
            def _render_view(view_name, masks_v, orig_v, spacing_v, ed_i, es_i, axis_keys):
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.patch.set_facecolor('#111111')

                print(f"[INFO] overlay {view_name}: ED_i={ed_i}, ES_i={es_i}")
                print(f"[INFO] overlay {view_name} mask ED area={int(np.sum(_ensure_int_labels(masks_v[ed_i]) == 1))}")
                print(f"[INFO] overlay {view_name} mask ES area={int(np.sum(_ensure_int_labels(masks_v[es_i]) == 1))}")

                for ax, (frame_i, phase) in zip(axs, [(ed_i, "ED"), (es_i, "ES")]):
                    mask = masks_v[frame_i]
                    bg = (orig_v[:, :, frame_i] if orig_v is not None and orig_v.ndim == 3 else orig_v)

                    if bg is not None:
                        ax.imshow(bg.T, cmap='gray', origin='lower')

                    int_mask = _ensure_int_labels(mask)
                    lv_only = (int_mask == 1).astype(np.uint8)
                    ax.imshow(
                        np.ma.masked_where(lv_only.T == 0, lv_only.T),
                        cmap='Reds', alpha=0.5, origin='lower'
                    )

                    au_key, ap_key, an_key = (
                        axis_keys[:3] if phase == "ED" else axis_keys[3:]
                    )

                    draw_simpson_lines(
                        ax, mask, spacing_v,
                        n_discs=calculator.n,
                        band_frac=calculator.band_frac,
                        min_band_points=calculator.min_band_points,
                        axis_u_override=result.get(au_key),
                        apex_mm=result.get(ap_key),
                        annulus_mid_mm=result.get(an_key),
                    )

                    ax.set_title(
                        f"{view_name} {phase} frame {frame_i}",
                        color='white', fontsize=13, fontweight='bold', pad=8
                    )
                    ax.axis('off')

                fig.tight_layout(pad=1.5)
                fname = f"overlay_{view_name.lower()}_{ts}.png"
                fig.savefig(
                    os.path.join(RESULT_FOLDER, fname),
                    bbox_inches='tight',
                    dpi=150,
                    facecolor=fig.get_facecolor()
                )
                plt.close(fig)
                return fname

            if masks2:
                overlay_2ch_file = _render_view(
                    "2CH", masks2, orig2_data, spacing2, ED_i2, ES_i2,
                    ("axis_u_2ch_ed", "apex_2ch_ed", "annulus_mid_2ch_ed",
                     "axis_u_2ch_es", "apex_2ch_es", "annulus_mid_2ch_es")
                )
            if masks4:
                overlay_4ch_file = _render_view(
                    "4CH", masks4, orig4_data, spacing4, ED_i4, ES_i4,
                    ("axis_u_4ch_ed", "apex_4ch_ed", "annulus_mid_4ch_ed",
                     "axis_u_4ch_es", "apex_4ch_es", "annulus_mid_4ch_es")
                )
        except Exception as e:
            print(f"[WARN] Overlay failed: {e}")
            traceback.print_exc()

        yield json.dumps({"progress": 90, "status": "保存到数据库..."}) + "\n"
        try:
            patient_uid = patient_data.get("patient_uid") or str(uuid.uuid4())[:8].upper()
            conn = get_db()
            cur = conn.cursor()

            cur.execute("SELECT id FROM patient WHERE patient_uid=%s", (patient_uid,))
            row = cur.fetchone()

            if row:
                patient_id = row[0]
                cur.execute(
                    "UPDATE patient SET name=%s, age=%s, gender=%s WHERE id=%s",
                    (patient_data.get("name"), patient_data.get("age"), patient_data.get("gender"), patient_id)
                )
            else:
                cur.execute(
                    "INSERT INTO patient (patient_uid, name, age, gender) VALUES (%s,%s,%s,%s)",
                    (patient_uid, patient_data.get("name"), patient_data.get("age"), patient_data.get("gender"))
                )
                patient_id = cur.lastrowid

            overlay_db_path = ";".join(filter(None, [overlay_2ch_file, overlay_4ch_file]))
            cur.execute(
                """INSERT INTO analysis_record
                   (patient_id, image_path, result_path, lvef, edv, esv, algorithm, view_mode, annulus_strategy)
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
            print(f"[WARN] DB save failed: {e}")
            traceback.print_exc()

        yield json.dumps({"progress": 100, "status": "分析完成！"}) + "\n"

        resp_data = {
            "LVEF": round(result["EF"], 2),
            "EDV": round(result["EDV"], 2),
            "ESV": round(result["ESV"], 2),
            "algorithm": algorithm,
            "annulus_strategy": resolved_strategy,
            "comparison": comparison,
            "mesh_3d_series": mesh_3d_series,
        }
        if overlay_2ch_file:
            resp_data["overlay_2ch_url"] = f"{BASE_URL}/results/{overlay_2ch_file}"
        if overlay_4ch_file:
            resp_data["overlay_4ch_url"] = f"{BASE_URL}/results/{overlay_4ch_file}"

        yield json.dumps({"result": resp_data}) + "\n"

    except Exception as e:
        traceback.print_exc()
        yield json.dumps({"error": str(e)}) + "\n"


# ──────────────────────────────────────────────────────────────────
#  路由
# ──────────────────────────────────────────────────────────────────
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM user WHERE username=%s AND password=%s",
        (data.get("username"), data.get("password"))
    )
    user = cur.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    payload = {
        "user_id": user[0],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return jsonify({"token": token})


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/algorithms', methods=['GET'])
def get_algorithms():
    return jsonify([{"key": k, "label": v} for k, v in ALGORITHM_LABELS.items()])


@app.route('/history', methods=['GET'])
@token_required
def get_history():
    try:
        conn = get_db()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT ar.id, p.patient_uid, p.name, p.age, p.gender,
                   ar.create_time, ar.lvef, ar.edv, ar.esv, ar.result_path,
                   ar.algorithm, ar.view_mode, ar.annulus_strategy
            FROM analysis_record ar
            JOIN patient p ON ar.patient_id = p.id
            ORDER BY ar.create_time DESC
        """)
        records = cur.fetchall()
        conn.close()

        for r in records:
            if r.get('create_time'):
                r['create_time'] = r['create_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(records)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/history/trend/<patient_uid>', methods=['GET'])
@token_required
def get_patient_trend(patient_uid):
    try:
        conn = get_db()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT ar.create_time, ar.lvef, ar.edv, ar.esv,
                   ar.algorithm, ar.view_mode, ar.annulus_strategy, ar.result_path,
                   p.name, p.age, p.gender, p.patient_uid
            FROM analysis_record ar
            JOIN patient p ON ar.patient_id = p.id
            WHERE p.patient_uid = %s
            ORDER BY ar.create_time ASC
        """, (patient_uid,))
        rows = cur.fetchall()

        cur.execute(
            "SELECT name, age, gender, patient_uid FROM patient WHERE patient_uid=%s",
            (patient_uid,)
        )
        p_info = cur.fetchone()
        conn.close()

        for r in rows:
            if r.get('create_time'):
                r['create_time'] = r['create_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({"patient": p_info, "records": rows})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/patients', methods=['GET'])
@token_required
def get_patients():
    try:
        conn = get_db()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("""
            SELECT p.patient_uid, p.name, p.age, p.gender,
                   COUNT(ar.id) AS record_count, MAX(ar.create_time) AS last_time
            FROM patient p
            LEFT JOIN analysis_record ar ON ar.patient_id = p.id
            GROUP BY p.id
            ORDER BY last_time DESC
        """)
        rows = cur.fetchall()
        conn.close()

        for r in rows:
            if r.get('last_time'):
                r['last_time'] = r['last_time'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(rows)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
@token_required
def analyze():
    try:
        file_2ch = request.files.get('file_2ch')
        file_4ch = request.files.get('file_4ch')
        if not file_2ch and not file_4ch:
            return jsonify({"error": "至少需要提供一个视图文件"}), 400

        algorithm = request.form.get("algorithm", "biplane_simpson")
        annulus_strategy = request.form.get("annulus_strategy", "auto")
        patient_uid = request.form.get("patient_uid", "").strip()

        path2ch = path4ch = None
        spacing2_override = spacing4_override = None

        if file_2ch and file_2ch.filename:
            path2ch, sp2 = save_upload(file_2ch, UPLOAD_FOLDER, "2ch_")
            if sp2:
                spacing2_override = sp2

        if file_4ch and file_4ch.filename:
            path4ch, sp4 = save_upload(file_4ch, UPLOAD_FOLDER, "4ch_")
            if sp4:
                spacing4_override = sp4

        patient_data = {
            'patient_uid': patient_uid,
            'name': request.form.get("name"),
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
        }

        return Response(
            generate_ndjson_response(
                path2ch, path4ch, patient_data,
                algorithm, annulus_strategy,
                spacing2_override, spacing4_override
            ),
            mimetype='application/x-ndjson'
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ensure_db_schema()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)