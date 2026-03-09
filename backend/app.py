import os
import datetime
import traceback
import numpy as np
import nibabel as nib
import pymysql
import jwt
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import *
from model_infer import run_inference
from biplane_simpson_clinical import BiplaneSimpsonClinical

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
BASE_URL = "http://127.0.0.1:5000"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)


LABEL_MAP = {
    0: "background",
    1: "LV",
    2: "LVWall",
    3: "LA",
}


def get_db():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4"
    )


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


def _ensure_int_labels(mask_arr: np.ndarray) -> np.ndarray:
    """
    nnUNet 输出一般是整数标签，但 nibabel get_fdata() 会变 float。
    这里统一转 int，避免 (arr == 1) 之类出现浮点误差。
    """
    arr = np.asarray(mask_arr)
    if arr.dtype.kind in ("i", "u"):
        return arr
    return np.rint(arr).astype(np.int16)


def draw_simpson_lines(ax, mask, spacing, n_discs=20, band_frac=1.6, min_band_points=3, assume_imshow_transposed=True ,axis_u_override=None
                       ,  apex_mm=None, annulus_mid_mm=None):
    """
    在 overlay 上画：
      - LV 长轴（裁剪到 LV 边界）
      - Simpson 分层真实直径线（每层端点为该层 band 内 LV 点的 s_min/s_max -> 自动裁剪在 LV 内）
    关键：考虑你 imshow 使用了 bg.T (转置) + origin='lower' 的坐标变换
    """
    dx, dy = float(spacing[0]), float(spacing[1])
    def to_disp(x_pix, y_pix):
        if assume_imshow_transposed:
            return (y_pix, x_pix)
        else:
            return (x_pix, y_pix)

    # 只用 LV 腔：label=1
    m = (_ensure_int_labels(mask) == 1)
    coords = np.column_stack(np.where(m))
    if coords.shape[0] < 30:
        return

    # coords: (row=y, col=x)
    ys = coords[:, 0].astype(float)
    xs = coords[:, 1].astype(float)

    # 转到物理 mm 坐标：X=col*dx, Y=row*dy
    pts = np.column_stack([xs * dx, ys * dy])

    # ===== PCA 主轴 =====
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

    # 投影到长轴/短轴坐标
        # ===== 选择坐标原点：有 annulus/apex 就用 annulus_mid 做原点（新版），否则退回 mean_pt（旧版）=====
    if apex_mm is not None and annulus_mid_mm is not None:
        origin_pt = np.asarray(annulus_mid_mm, dtype=float)
        apex_pt = np.asarray(apex_mm, dtype=float)

        # 强制 axis_u 朝向 apex（避免方向翻转）
        if float(np.dot(apex_pt - origin_pt, axis_u)) < 0:
            axis_u = -axis_u
            perp_u = np.array([-axis_u[1], axis_u[0]], dtype=float)

        centered = pts - origin_pt
        t = centered @ axis_u
        s = centered @ perp_u

        # 长轴长度：瓣环->心尖（用给定点）
        L = float(np.dot(apex_pt - origin_pt, axis_u))
        if L <= 1e-6:
            return

        # 分层范围：0..L（只取腔体内投影，防止 band 在 base 外乱飘）
        tmin, tmax = 0.0, L

    else:
        # 旧版：以质心为原点
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

    # ===== 画长轴线 =====
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

# ===== 画每层真实直径线 =====
    for c in centers:
        band = (t >= c - band_half) & (t <= c + band_half)
        if np.count_nonzero(band) < int(min_band_points):
            continue

        smin = float(np.min(s[band]))
        smax = float(np.max(s[band]))
        if smax <= smin:
            continue

        # 该层中心点：必须用同一个 origin_pt（否则会整体漂移到腔体外）
        center_pt = origin_pt + axis_u * c

        a = center_pt + perp_u * smin
        b = center_pt + perp_u * smax

        ax_x1_pix, ax_y1_pix = a[0] / dx, a[1] / dy
        ax_x2_pix, ax_y2_pix = b[0] / dx, b[1] / dy

        XA1, YA1 = to_disp(ax_x1_pix, ax_y1_pix)
        XA2, YA2 = to_disp(ax_x2_pix, ax_y2_pix)

        ax.plot([XA1, XA2], [YA1, YA2], color='yellow', linewidth=1.0, alpha=0.75)


def compute_area_curve_mm2(masks, spacing_xy, label_id: int) -> np.ndarray:
    """
    计算某个 label 在序列每一帧的面积曲线（mm^2）
    masks: list[np.ndarray] 每一帧 2D mask
    spacing_xy: (dx, dy) 单位 mm
    """
    dx, dy = float(spacing_xy[0]), float(spacing_xy[1])
    pix_area = dx * dy  # mm^2 per pixel

    areas = []
    for m in masks:
        mi = _ensure_int_labels(m)
        cnt = np.count_nonzero(mi == int(label_id))
        areas.append(cnt * pix_area)
    return np.asarray(areas, dtype=float)

def print_area_report(view_name: str, masks, spacing_xy, ed_idx: int, es_idx: int):
    """
    只打印 mm^2，并且所有数值保留两位小数
    """
    dx, dy = float(spacing_xy[0]), float(spacing_xy[1])
    pix_area = dx * dy

    print(f"\n========== [AREA REPORT] {view_name} ==========")
    print(f"[INFO] spacing dx={dx:.6f} mm, dy={dy:.6f} mm, pixel_area={pix_area:.6f} mm^2")
    print(f"[INFO] total frames = {len(masks)}")
    print(f"[INFO] ED index = {ed_idx}, ES index = {es_idx}")

    # 控制 numpy 打印：每个元素两位小数
    fmt2 = lambda x: f"{x:.2f}"

    for lab in [0, 1, 2, 3]:
        curve_mm2 = compute_area_curve_mm2(masks, spacing_xy, lab)
        name = LABEL_MAP.get(lab, f"label_{lab}")

        curve_str = np.array2string(
            curve_mm2,
            precision=2,
            separator=' ',
            max_line_width=120,
            formatter={'float_kind': fmt2}
        )

        print(f"\n[{view_name}] label={lab} ({name}) area_curve_mm2:\n{curve_str}")

        if 0 <= ed_idx < len(curve_mm2) and 0 <= es_idx < len(curve_mm2):
            print(
                f"[{view_name}] {name} ED_area = {curve_mm2[ed_idx]:.2f} mm^2, "
                f"ES_area = {curve_mm2[es_idx]:.2f} mm^2"
            )

    print("========== [AREA REPORT END] ==========\n")
    sys.stdout.flush()



@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM user WHERE username=%s AND password=%s",
        (data.get("username"), data.get("password"))
    )
    user = cursor.fetchone()
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


@app.route('/analyze', methods=['POST'])
@token_required
def analyze():
    try:
        file_2ch = request.files.get('file_2ch')
        file_4ch = request.files.get('file_4ch')

        if not file_2ch or not file_4ch:
            return jsonify({"error": "Need both 2CH and 4CH"}), 400

        path2ch = os.path.join(UPLOAD_FOLDER, secure_filename(file_2ch.filename))
        path4ch = os.path.join(UPLOAD_FOLDER, secure_filename(file_4ch.filename))

        file_2ch.save(path2ch)
        file_4ch.save(path4ch)

        # ========================== 从原图读 spacing ==========================
        orig2 = nib.load(path2ch)
        orig4 = nib.load(path4ch)
        spacing2 = orig2.header.get_zooms()[:2]
        spacing4 = orig4.header.get_zooms()[:2]

        print(f"\n\n[INFO] spacing2={spacing2}, spacing4={spacing4}")
        sys.stdout.flush()

        # ========================== nnUNet 推理 ==========================
        _, preds2ch = run_inference(path2ch, "case_2ch", NNUNET_DATASET_2CH)
        _, preds4ch = run_inference(path4ch, "case_4ch", NNUNET_DATASET_4CH)




                # ========================== ✅ CHECK & FIX spacing by pred header ==========================
        try:
            pred2_0 = nib.load(preds2ch[0])
            pred4_0 = nib.load(preds4ch[0])

            print("[CHECK] orig2 spacing:", orig2.header.get_zooms()[:2])
            print("[CHECK] pred2 spacing:", pred2_0.header.get_zooms()[:2])
            print("[CHECK] orig4 spacing:", orig4.header.get_zooms()[:2])
            print("[CHECK] pred4 spacing:", pred4_0.header.get_zooms()[:2])

            # ✅ 最直接修复：体积计算用 pred spacing
            spacing2 = pred2_0.header.get_zooms()[:2]
            spacing4 = pred4_0.header.get_zooms()[:2]

            print(f"[FIX] use pred spacing for volume: spacing2={spacing2}, spacing4={spacing4}")
            sys.stdout.flush()
        except Exception as e:
            print("[WARNING] pred spacing check failed, keep orig spacing:", e)
            traceback.print_exc()
            sys.stdout.flush()





        # ========================== 读取 masks ==========================
        masks2 = []
        masks4 = []

        for p in preds2ch:
            img = nib.load(p)
            data = img.get_fdata()
            masks2.append(data)

        for p in preds4ch:
            img = nib.load(p)
            data = img.get_fdata()
            masks4.append(data)

        # ========================== Clinical Biplane Simpson ==========================
        calculator = BiplaneSimpsonClinical(n_discs=20)

        result = calculator.compute_ed_es_from_series(
            masks2,
            masks4,
            spacing2,
            spacing4
        )
      
        
        # ========================== ✅ 3D 非对称重建：全序列（每帧一个 vertices） ==========================
        mesh_3d_series = None
        try:
            # ==========================
            # 核心保证：LVEF 计算用原始索引，完全不做对齐/插值
            # 仅 3D 序列展示时使用对齐后的索引
            # ==========================
            # 1. 读取原始 ED/ES 索引（LVEF 计算的核心，一丝不动）
            ed_2ch_original = int(result['ED_index'])
            es_2ch_original = int(result['ES_index'])
            ed_4ch_original = int(result.get('ED_index_4ch', 0))
            es_4ch_original = int(result.get('ES_index_4ch', 0))
            
            # 2. 仅为 3D 序列调用对齐函数（不影响任何 LVEF 计算逻辑）
            aligned_indices_2ch, aligned_indices_4ch = calculator.align_series_indices(
                ed_2ch_original, es_2ch_original,
                ed_4ch_original, es_4ch_original
            )
            T = len(aligned_indices_2ch)

            # ===== ✅ 固定全序列参考系：用 ED 帧的 origin/axis（避免随帧抖动导致“绕轴转”）=====
            info_ref = calculator.frame_bounds_and_L(
                masks2[ed_2ch_original],
                masks4[ed_4ch_original],
                spacing2,
                spacing4
            )
            origin_ref_2ch = info_ref.get("origin_2ch_mm", None)
            axis_ref_2ch   = info_ref.get("axis_u_2ch", None)

            # 兜底：如果 ED 帧也拿不到参考系，就用对齐序列第0帧的（尽量不改变你现有流程）
            if origin_ref_2ch is None or axis_ref_2ch is None:
                info_ref0 = calculator.frame_bounds_and_L(
                    masks2[aligned_indices_2ch[0]],
                    masks4[aligned_indices_4ch[0]],
                    spacing2,
                    spacing4
                )
                origin_ref_2ch = info_ref0.get("origin_2ch_mm", None)
                axis_ref_2ch   = info_ref0.get("axis_u_2ch", None)


            if T <= 0:
                raise RuntimeError("Empty mask series after alignment")

            # faces 只需要生成一次（拓扑固定：n_discs=20, num_theta=32）
            # 我们先用第 0 帧生成 faces
            info0 = calculator.frame_bounds_and_L(
                masks2[aligned_indices_2ch[0]],
                masks4[aligned_indices_4ch[0]],
                spacing2,
                spacing4
            )
            v0, f0 = calculator.generate_3d_mesh_asymmetric(
                info0["bounds_2ch"], info0["bounds_4ch"], info0["h_mm"]
            )
            faces_list = np.asarray(f0, dtype=int).tolist()

            vertices_series = []
            for t in range(T):
                idx2 = aligned_indices_2ch[t]
                idx4 = aligned_indices_4ch[t]

                info = calculator.frame_bounds_and_L(
                    masks2[idx2], masks4[idx4], spacing2, spacing4
                )

                # 某些帧分割太差可能 bounds None 或 h=0，做个兜底：用上一帧
                if (info["bounds_2ch"] is None) or (info["bounds_4ch"] is None) or (info["h_mm"] <= 1e-6):
                    if len(vertices_series) > 0:
                        vertices_series.append(vertices_series[-1])
                        continue
                    else:
                        # 第一帧就坏：直接跳过
                        v_bad = np.zeros((calculator.n * 32, 3), dtype=float)
                        vertices_series.append(v_bad.tolist())
                        continue
               
                # 在app.py的mesh_3d_series生成逻辑中
                """
                v, _ = calculator.generate_3d_mesh_asymmetric(
                    info["bounds_2ch"], 
                    info["bounds_4ch"], 
                    info["h_mm"],
                    origin_2ch_mm=info["origin_2ch_mm"],  # 新增
                    axis_u_2ch=info["axis_u_2ch"],        # 新增
                    origin_4ch_mm=info["origin_4ch_mm"],  # 新增
                    axis_u_4ch=info["axis_u_4ch"]         # 新增
                )
                """
                v, _ = calculator.generate_3d_mesh_asymmetric(
                info["bounds_2ch"], 
                info["bounds_4ch"], 
                info["h_mm"],
                origin_2ch_mm=origin_ref_2ch,  # ✅ 固定：不要用每帧的
                axis_u_2ch=axis_ref_2ch        # ✅ 固定：不要用每帧的
                # origin_4ch_mm / axis_u_4ch 你原本也没在 generate_3d_mesh_asymmetric 里用到，不动也行
            )
                vertices_series.append(np.asarray(v, dtype=float).tolist())

            mesh_3d_series = {
                "faces": faces_list,
                "vertices_series": vertices_series,
                "n_frames": int(len(vertices_series)),
                "n_discs": int(calculator.n),
                "num_theta": 32
            }

            print(f"[INFO] 3D mesh series generated: frames={mesh_3d_series['n_frames']}, verts/frame={len(vertices_series[0])}")
            sys.stdout.flush()

        except Exception as e:
            print(f"[WARN] 3D series generation failed: {e}")
            traceback.print_exc()
            mesh_3d_series = None
        
        # ========================== 计算独立的 ED/ES 帧索引 ==========================
        curve = np.array(result["curve"])
        T2 = len(masks2)
        T4 = len(masks4)

        ED_index_2ch = int(result['ED_index'])
        ES_index_2ch = int(result['ES_index'])
        ED_index_4ch = int(result.get('ED_index_4ch', 0))
        ES_index_4ch = int(result.get('ES_index_4ch', 0))

        best_shift = result.get("best_shift", 0)

        print(f"[INFO] 2CH: ED frame index = {ED_index_2ch}, ES frame index = {ES_index_2ch}")
        print(f"[INFO] 4CH: ED frame index = {ED_index_4ch}, ES frame index = {ES_index_4ch}")
        print(f"[INFO] best_shift = {best_shift}")
        print(f"[INFO] EDV={result['EDV']:.2f}, ESV={result['ESV']:.2f}, EF={result['EF']:.2f}")

        # ========================== ✅ 计算并打印四类面积（仅终端输出） ==========================
        # 2CH 用 2CH 的 ED/ES；4CH 用 4CH 的 ED/ES（已经按 best_shift 修正过）
        try:
            print_area_report("2CH", masks2, spacing2, ED_index_2ch, ES_index_2ch)
            print_area_report("4CH", masks4, spacing4, ED_index_4ch, ES_index_4ch)
        except Exception as e:
            print("[WARNING] Area report failed:", e)
            traceback.print_exc()

        # ========================== 生成 overlay (显示 ED 和 ES 帧) ==========================
        overlay_file = None
        try:
            orig2_data = orig2.get_fdata()
            orig4_data = orig4.get_fdata()

            # ED 帧
            mask2_ed = masks2[ED_index_2ch]
            mask4_ed = masks4[ED_index_4ch]
            bg2_ed = orig2_data[:, :, ED_index_2ch] if orig2_data.ndim == 3 else orig2_data
            bg4_ed = orig4_data[:, :, ED_index_4ch] if orig4_data.ndim == 3 else orig4_data

            # ES 帧
            mask2_es = masks2[ES_index_2ch]
            mask4_es = masks4[ES_index_4ch]
            bg2_es = orig2_data[:, :, ES_index_2ch] if orig2_data.ndim == 3 else orig2_data
            bg4_es = orig4_data[:, :, ES_index_4ch] if orig4_data.ndim == 3 else orig4_data

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            # ED row
            axs[0, 0].imshow(bg2_ed.T, cmap='gray', origin='lower')
            axs[0, 0].imshow(np.ma.masked_where(_ensure_int_labels(mask2_ed).T == 0, _ensure_int_labels(mask2_ed).T),
                             cmap='Reds', alpha=0.5, origin='lower')
            draw_simpson_lines(
                axs[0, 0],
                mask2_ed,
                spacing2,
                #n_discs=20
                n_discs=calculator.n,
                band_frac=calculator.band_frac,
                min_band_points=calculator.min_band_points,
                axis_u_override=result.get("axis_u_2ch_ed"),
                apex_mm=result.get("apex_2ch_ed"),
                annulus_mid_mm=result.get("annulus_mid_2ch_ed"),
            )

            axs[0, 0].set_title(f"2CH ED frame {ED_index_2ch}")
            axs[0, 0].axis('off')

            axs[0, 1].imshow(bg4_ed.T, cmap='gray', origin='lower')
            axs[0, 1].imshow(np.ma.masked_where(_ensure_int_labels(mask4_ed).T == 0, _ensure_int_labels(mask4_ed).T),
                             cmap='Reds', alpha=0.5, origin='lower')
            draw_simpson_lines(
                axs[0, 1],
                mask4_ed,
                spacing4,
                #n_discs=20
                n_discs=calculator.n,
                band_frac=calculator.band_frac,
                min_band_points=calculator.min_band_points,
                axis_u_override=result.get("axis_u_4ch_ed"),

                apex_mm=result.get("apex_4ch_ed"),
                annulus_mid_mm=result.get("annulus_mid_4ch_ed"),

            )

            axs[0, 1].set_title(f"4CH ED frame {ED_index_4ch}")
            axs[0, 1].axis('off')

            # ES row
            axs[1, 0].imshow(bg2_es.T, cmap='gray', origin='lower')
            axs[1, 0].imshow(np.ma.masked_where(_ensure_int_labels(mask2_es).T == 0, _ensure_int_labels(mask2_es).T),
                             cmap='Reds', alpha=0.5, origin='lower')
            draw_simpson_lines(
                axs[1, 0],
                mask2_es,
                spacing2,
                #n_discs=20
                n_discs=calculator.n,
                band_frac=calculator.band_frac,
                min_band_points=calculator.min_band_points,
                axis_u_override=result.get("axis_u_2ch_es"),
                apex_mm=result.get("apex_2ch_es"),
                annulus_mid_mm=result.get("annulus_mid_2ch_es"),
            )

            axs[1, 0].set_title(f"2CH ES frame {ES_index_2ch}")
            axs[1, 0].axis('off')

            axs[1, 1].imshow(bg4_es.T, cmap='gray', origin='lower')
            axs[1, 1].imshow(np.ma.masked_where(_ensure_int_labels(mask4_es).T == 0, _ensure_int_labels(mask4_es).T),
                             cmap='Reds', alpha=0.5, origin='lower')
            draw_simpson_lines(
                axs[1, 1],
                mask4_es,
                spacing4,
                #n_discs=20
                n_discs=calculator.n,
                band_frac=calculator.band_frac,
                min_band_points=calculator.min_band_points,
                axis_u_override=result.get("axis_u_4ch_es"),apex_mm=result.get("apex_4ch_es"),
                annulus_mid_mm=result.get("annulus_mid_4ch_es")
            )

            axs[1, 1].set_title(f"4CH ES frame {ES_index_4ch}")
            axs[1, 1].axis('off')

            overlay_file = f"overlay_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            overlay_path = os.path.join(RESULT_FOLDER, overlay_file)
            fig.savefig(overlay_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"[INFO] Overlay of ED&ES of 2CH&4CH sequence saved: {overlay_path}")

        except Exception as e:
            print("Overlay generation failed:", e)
            traceback.print_exc()

        # ========================== 写入数据库 ==========================
        try:
            conn = get_db()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO patient (name, age, gender) VALUES (%s, %s, %s)",
                (
                    request.form.get("name"),
                    request.form.get("age"),
                    request.form.get("gender")
                )
            )
            patient_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO analysis_record
                (patient_id, image_path, result_path, lvef, edv, esv)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    patient_id,
                    path2ch + ";" + path4ch,
                    overlay_file if overlay_file else "",
                    float(result["EF"]),
                    float(result["EDV"]),
                    float(result["ESV"]),
                )
            )

            conn.commit()
            conn.close()
            print("[INFO] Database record saved successfully.")

        except Exception as e:
            print("Database save failed:", e)
            traceback.print_exc()

        # ========================== 返回结果（✅ 不再覆盖！） ==========================
        response = {
            #"LVEF": round(result["EF"]-11.429, 2),#-11.429是因为经验之差
            "LVEF": round(result["EF"], 2),
            "EDV": round(result["EDV"], 2),
            "ESV": round(result["ESV"], 2),
            #"mesh_3d": three_d_payload
             "mesh_3d_series": mesh_3d_series
        }

        if overlay_file:
            response["overlay_url"] = f"{BASE_URL}/results/{overlay_file}"

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)