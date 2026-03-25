"""
BiplaneSimpsonClinical — 双平面 Simpson 法 LVEF 计算
=====================================================

两套可选方案（通过 annulus_strategy 参数切换）：

  策略 A  "polar"   （原 Doc2）
    瓣环：对 LV 边缘做极坐标变换，取 r(θ) 前三峰，
          距离最近的两个峰 = 左右瓣环点。
    心尖：第三个峰（排除已选瓣环峰），兜底为距 annulus_mid 最远的腔体点。
    适用：仅有 cavity(1) 标签的纯腔体分割。

  策略 B  "wall_la" （原 Doc3）
    瓣环：基于 Wall(2)＋LA(3) 接触带 + PCA 强制左右分区，
          各区最外点 = 左右瓣环点。
    心尖：直接取距 annulus_mid 最远的 cavity 点。
    适用：含 Wall(2) 和 LA(3) 多类别标签的分割结果。

主要参数
--------
  n_discs          : 辛普森圆盘数量（默认 20）
  cavity_label     : LV 腔体标签值（默认 1）
  annulus_strategy : "polar" 或 "wall_la"
  band_frac        : 切片宽度系数（polar 策略建议 1.0；wall_la 策略建议 0.2）
  min_band_points  : 切片最少像素数（默认 3）
  robust_width     : True=95%-5% 百分位宽度；False=max-min（默认 False）
  enforce_apex_zero: 边界外直径置 0（默认 True）
  keep_largest_cc  : 保留最大连通域（默认 True，当前为 pass-through）
"""

import numpy as np

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

# ──────────────────────────────────────────────────────────────────────────────
#  算法标签（供前后端共用）
# ──────────────────────────────────────────────────────────────────────────────
ALGORITHM_LABELS = {
    "biplane_simpson":  "双平面辛普森 (2CH + 4CH)  [金标准]",
    "singleplane_2ch":  "单平面辛普森 — 2CH only",
    "singleplane_4ch":  "单平面辛普森 — 4CH only",
    "area_length_2ch":  "面积-长度法  — 2CH only",
    "area_length_4ch":  "面积-长度法  — 4CH only",
}


# ──────────────────────────────────────────────────────────────────────────────
class BiplaneSimpsonClinical:

    # ═══════════════════════════════════════════════════════════════════════════
    #  初始化
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(
        self,
        n_discs: int = 20,
        cavity_label: int = 1,
        annulus_strategy: str = "polar",   # "polar" 或 "wall_la"
        band_frac: float = None,           # None = 根据策略自动设置
        min_band_points: int = 3,
        robust_width: bool = False,
        enforce_apex_zero: bool = True,
        keep_largest_cc: bool = True,
    ):
        if n_discs < 5:
            raise ValueError("n_discs 至少为 5（推荐 20）")
        if annulus_strategy not in ("polar", "wall_la"):
            raise ValueError("annulus_strategy 须为 'polar' 或 'wall_la'")

        self.n = int(n_discs)
        self.cavity_label = int(cavity_label)
        self.annulus_strategy = annulus_strategy

        # band_frac 默认：polar → 1.0；wall_la → 0.2
        if band_frac is None:
            band_frac = 1.0 if annulus_strategy == "polar" else 0.2
        self.band_frac = float(band_frac)

        self.min_band_points = int(min_band_points)
        self.robust_width = bool(robust_width)
        self.enforce_apex_zero = bool(enforce_apex_zero)
        self.keep_largest_cc = bool(keep_largest_cc)


    def _fmt_curve(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return []
        return arr.astype(int).tolist()


    def _find_ed_es_robust(self, area_px, view_name=""):
        area_px = np.asarray(area_px, dtype=float)

        n = len(area_px)

        if n == 0:
            return -1, -1

        if n <= 4:
            return int(np.argmax(area_px)), int(np.argmin(area_px))

        med = np.median(area_px)

        mad = np.median(np.abs(area_px - med))

        if mad < 1e-6:
            return int(np.argmax(area_px)), int(np.argmin(area_px))

        robust_z = 0.6745 * (area_px - med) / mad

        valid = np.abs(robust_z) <= 3.5

        if np.sum(valid) < max(3, n // 2):
            return int(np.argmax(area_px)), int(np.argmin(area_px))

        valid_idx = np.where(valid)[0]

        valid_vals = area_px[valid]

        ed_i = int(valid_idx[np.argmax(valid_vals)])

        es_i = int(valid_idx[np.argmin(valid_vals)])

        dropped_idx = np.where(~valid)[0].tolist()

        if dropped_idx:
            print(f"[INFO] {view_name} 异常面积帧已剔除: {dropped_idx}")

        return ed_i, es_i
    # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 1：单帧体积
    # ═══════════════════════════════════════════════════════════════════════════
    def volume_single_frame(
        self, mask2, mask4, spacing2, spacing4,
        return_debug=False, verbose=False
    ):
        """
        双平面 Simpson：V = Σ (π/4 · d2ᵢ · d4ᵢ · h)
        """
        d2, L2, dbg2 = self._diameters_from_mask(mask2, spacing2, return_debug=True)
        d4, L4, dbg4 = self._diameters_from_mask(mask4, spacing4, return_debug=True)

        L = 0.5 * (L2 + L4)
        h = L / self.n if L > 1e-6 else 0.0
        disc_vols = (np.pi / 4.0) * d2 * d4 * h
        vol_mm3 = float(np.sum(disc_vols))
        vol_mL = vol_mm3 / 1000.0

        if verbose:
            print("\n" + "=" * 80)
            print("[DEBUG] 单帧容积 — 辛普森双平面")
            print(f"[DEBUG]  L2={L2:.4f}mm  L4={L4:.4f}mm  Lavg={L:.4f}mm  n={self.n}  h={h:.4f}mm")
            print(f"[DEBUG]  d2={d2.tolist()}")
            print(f"[DEBUG]  d4={d4.tolist()}")
            for i in range(self.n):
                print(
                    f"[DEBUG]  圆盘{i+1:2d}: π/4 × {d2[i]:.4f} × {d4[i]:.4f} × {h:.4f}"
                    f" = {disc_vols[i]:.4f} mm³"
                )
            print(f"[DEBUG]  vol_mm3={vol_mm3:.4f}  vol_mL={vol_mL:.4f}")
            print("=" * 80 + "\n")

        if return_debug:
            return vol_mL, {
                "view2": dbg2, "view4": dbg4,
                "L2_mm": float(L2), "L4_mm": float(L4),
                "L_avg_mm": float(L), "h_mm": float(h),
                "d2_mm": d2.tolist(), "d4_mm": d4.tolist(),
                "vol_mm3": float(vol_mm3), "vol_mL": float(vol_mL),
                "disc_volumes_mm3": disc_vols.tolist(),
            }
        return vol_mL

    # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 2：序列 ED/ES/EF
    # ═══════════════════════════════════════════════════════════════════════════
    def compute_ed_es_from_series(self, masks2, masks4, spacing2, spacing4):
        """
        对 2CH/4CH 序列各自独立找 ED/ES（像素面积最大/最小帧），
        分别计算 EDV、ESV，最终求 EF。
        """

        T2, T4 = len(masks2), len(masks4)
        if T2 == 0 or T4 == 0:
            return dict(EDV=0.0, ESV=0.0, EF=0.0,
                        ED_index=-1, ES_index=-1, curve=[])

        area2 = np.array([np.sum(self._to_cavity_mask(m)) for m in masks2], float)
        area4 = np.array([np.sum(self._to_cavity_mask(m)) for m in masks4], float)

        print(f"[INFO] 2CH LV 像素曲线: {self._fmt_curve(area2)}")
        print(f"[INFO] 4CH LV 像素曲线: {self._fmt_curve(area4)}")

        ed2, es2 = self._find_ed_es_robust(area2, view_name="2CH")
        ed4, es4 = self._find_ed_es_robust(area4, view_name="4CH")
        print(f"[INFO] 2CH: ED={ed2}, area={int(area2[ed2])}")
        print(f"[INFO] 2CH: ES={es2}, area={int(area2[es2])}")
        print(f"[INFO] 4CH: ED={ed4}, area={int(area4[ed4])}")
        print(f"[INFO] 4CH: ES={es4}, area={int(area4[es4])}")

        print("\n[DEBUG] ===== ED FRAME =====")
        EDV, dbg_ed = self.volume_single_frame(
            masks2[ed2], masks4[ed4], spacing2, spacing4,
            return_debug=True, verbose=True)

        print("\n[DEBUG] ===== ES FRAME =====")
        ESV, dbg_es = self.volume_single_frame(
            masks2[es2], masks4[es4], spacing2, spacing4,
            return_debug=True, verbose=True)

        EF = float((EDV - ESV) / EDV * 100.0) if EDV > 1e-6 else 0.0
        print(f"[INFO] EDV={EDV:.2f}  ESV={ESV:.2f}  EF={EF:.2f}")

        return self._build_result(EDV, ESV, EF, ed2, es2, ed4, es4, dbg_ed, dbg_es)  
 # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 3：序列对齐（两种模式）
    # ═══════════════════════════════════════════════════════════════════════════
    def align_series_indices_full(self, ed_2ch, es_2ch, T2, ed_4ch, es_4ch, T4):
        """
        模式 A（原 Doc1）：传入全序列长度，对整段序列均匀重采样。
        用途：3D 动画帧对齐。
        返回：(idx2_list, idx4_list)，长度均为 max(T2, T4)。
        """
        T = max(T2, T4)
        if T <= 1:
            return [0], [0]

        def resample(T_src, T_total):
            return [
                min(int(round(i / (T_total - 1) * (T_src - 1))), T_src - 1)
                for i in range(T_total)
            ]

        return resample(T2, T), resample(T4, T)

    def align_series_indices_edES(self, ed_2ch, es_2ch, ed_4ch, es_4ch):
        """
        模式 B（原 Doc3）：仅对 ED→ES 段做线性插值对齐。
        用途：EDV/ESV 对应帧配对。
        返回：(idx2_list, idx4_list)，以帧数较短的视图为基准。
        """
        len2 = es_2ch - ed_2ch + 1
        len4 = es_4ch - ed_4ch + 1

        if len2 <= len4:
            base_ed, base_es = ed_2ch, es_2ch
            other_ed, other_es = ed_4ch, es_4ch
            base_view = "2CH"
        else:
            base_ed, base_es = ed_4ch, es_4ch
            other_ed, other_es = ed_2ch, es_2ch
            base_view = "4CH"

        base_len = base_es - base_ed + 1
        base_idx = list(range(base_ed, base_es + 1))

        if base_len == 1:
            other_idx = [other_ed]
        else:
            t = np.linspace(0, 1, base_len)
            other_idx = np.round(other_ed + t * (other_es - other_ed)).astype(int).tolist()

        idx2 = base_idx if base_view == "2CH" else other_idx
        idx4 = other_idx if base_view == "2CH" else base_idx

        print(f"[INFO] 序列对齐（ED→ES），基准视图={base_view}")
        print(f"[INFO] 2CH 帧: {idx2}")
        print(f"[INFO] 4CH 帧: {idx4}")
        return idx2, idx4

    def align_series_indices(self, ed_2ch, es_2ch, T2, ed_4ch, es_4ch, T4):
        """向后兼容别名 → align_series_indices_full（用于 3D 动画）"""
        return self.align_series_indices_full(ed_2ch, es_2ch, T2, ed_4ch, es_4ch, T4)

    # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 4b：单视图 ED/ES（单平面 Simpson & 面积-长度法）
    # ═══════════════════════════════════════════════════════════════════════════
    def compute_ed_es_single_view(
    self, masks, spacing, method: str = "singleplane", view_name: str = "2CH"
):
        """
        仅用单视图序列计算 LVEF。

        method:
          "singleplane" — 单平面 Simpson：假设每个截面为圆形，
                          V = Σ (π/4 · dᵢ² · h)
          "area_length" — 面积-长度法：V = 8A² / (3π·L)
                          A = LV 腔体面积(mm²)，L = 长轴长度(mm)

        返回与 compute_ed_es_from_series 相同的结构（对侧视图字段置 None）。
        """

        if not masks:
            return dict(EDV=0.0, ESV=0.0, EF=0.0, ED_index=-1, ES_index=-1, curve=[])

        area_px = np.array([np.sum(self._to_cavity_mask(m)) for m in masks], float)
        print(f"[INFO] {view_name} 像素曲线: {self._fmt_curve(area_px)}")

        ed_i, es_i = self._find_ed_es_robust(area_px, view_name=view_name)
        print(f"[INFO] {view_name}: ED={ed_i}, area={int(area_px[ed_i])}")
        print(f"[INFO] {view_name}: ES={es_i}, area={int(area_px[es_i])}")

        if method == "singleplane":
            EDV = self._singleplane_vol(masks[ed_i], spacing, verbose=True)
            ESV = self._singleplane_vol(masks[es_i], spacing, verbose=True)
        elif method == "area_length":
            EDV = self._area_length_vol(masks[ed_i], spacing, verbose=True)
            ESV = self._area_length_vol(masks[es_i], spacing, verbose=True)
        else:
            raise ValueError(f"未知 method: {method}")

        EF = float((EDV - ESV) / EDV * 100.0) if EDV > 1e-6 else 0.0
        print(f"[INFO] EDV={EDV:.2f}  ESV={ESV:.2f}  EF={EF:.2f}")

        _, _, dbg_ed = self._diameters_from_mask(masks[ed_i], spacing, return_debug=True)
        _, _, dbg_es = self._diameters_from_mask(masks[es_i], spacing, return_debug=True)

        other_view = "4ch" if view_name == "2CH" else "2ch"
        cur_view = view_name.lower()

        return dict(
            EDV=EDV, ESV=ESV, EF=EF, curve=[],

            ED_index=(ed_i if view_name == "2CH" else 0),
            ES_index=(es_i if view_name == "2CH" else 0),

            ED_index_4ch=(ed_i if view_name == "4CH" else 0),
            ES_index_4ch=(es_i if view_name == "4CH" else 0),

            **{f"axis_u_{cur_view}_ed": dbg_ed.get("axis_u")},
            **{f"axis_u_{cur_view}_es": dbg_es.get("axis_u")},
            **{f"apex_{cur_view}_ed": dbg_ed.get("apex_mm")},
            **{f"annulus_mid_{cur_view}_ed": dbg_ed.get("annulus_mid_mm")},
            **{f"apex_{cur_view}_es": dbg_es.get("apex_mm")},
            **{f"annulus_mid_{cur_view}_es": dbg_es.get("annulus_mid_mm")},

            **{f"axis_u_{other_view}_ed": None},
            **{f"axis_u_{other_view}_es": None},
            **{f"apex_{other_view}_ed": None},
            **{f"annulus_mid_{other_view}_ed": None},
            **{f"apex_{other_view}_es": None},
            **{f"annulus_mid_{other_view}_es": None},

            bounds_2ch_ed=(dbg_ed.get("bounds_mm") if view_name == "2CH" else None),
            bounds_4ch_ed=(dbg_ed.get("bounds_mm") if view_name == "4CH" else None),
            h_mm_ed=dbg_ed.get("h_mm", 0.0),
        )
    # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 4c：所有算法并行计算（供前端对比面板）
    # ═══════════════════════════════════════════════════════════════════════════
    def compute_all_algorithms(self, masks2, masks4, spacing2, spacing4) -> dict:
        """
        对可用视图跑所有算法，返回 {algo_key: result_dict or None}。
        任何算法失败时不抛出，返回 None。
        """
        has2 = bool(masks2)
        has4 = bool(masks4)
        results = {}

        def _safe(fn):
            try:
                return fn()
            except Exception as e:
                print(f"[WARN] algo failed: {e}")
                return None

        results["biplane_simpson"] = (
            _safe(lambda: self.compute_ed_es_from_series(masks2, masks4, spacing2, spacing4))
            if has2 and has4 else None
        )
        results["singleplane_2ch"] = (
            _safe(lambda: self.compute_ed_es_single_view(masks2, spacing2, "singleplane", "2CH"))
            if has2 else None
        )
        results["singleplane_4ch"] = (
            _safe(lambda: self.compute_ed_es_single_view(masks4, spacing4, "singleplane", "4CH"))
            if has4 else None
        )
        results["area_length_2ch"] = (
            _safe(lambda: self.compute_ed_es_single_view(masks2, spacing2, "area_length", "2CH"))
            if has2 else None
        )
        results["area_length_4ch"] = (
            _safe(lambda: self.compute_ed_es_single_view(masks4, spacing4, "area_length", "4CH"))
            if has4 else None
        )

        return results

    def frame_bounds_and_L(self, mask2, mask4, spacing2, spacing4):
        _, L2, dbg2 = self._diameters_from_mask(mask2, spacing2, return_debug=True)
        _, L4, dbg4 = self._diameters_from_mask(mask4, spacing4, return_debug=True)
        L = 0.5 * (float(L2) + float(L4))
        h = L / float(self.n) if L > 1e-6 else 0.0
        return {
            "bounds_2ch": dbg2.get("bounds_mm"),
            "bounds_4ch": dbg4.get("bounds_mm"),
            "L2_mm": float(L2), "L4_mm": float(L4),
            "L_avg_mm": float(L), "h_mm": float(h),
            "origin_2ch_mm": dbg2.get("annulus_mid_mm"),
            "axis_u_2ch": dbg2.get("axis_u"),
            "origin_4ch_mm": dbg4.get("annulus_mid_mm"),
            "axis_u_4ch": dbg4.get("axis_u"),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    #  公开接口 5：3D 非对称网格（封闭，含底盖/顶盖）
    # ═══════════════════════════════════════════════════════════════════════════
    def generate_3d_mesh_asymmetric(
        self, bounds_2ch, bounds_4ch, h,
        origin_2ch_mm=None, axis_u_2ch=None,
        origin_4ch_mm=None, axis_u_4ch=None,
    ):
        """
        每层以 bounds_4ch[i] 定义椭圆的 X 半径，bounds_2ch[i] 定义 Y 半径，
        生成 n_discs × num_theta 个顶点，拼接侧壁三角面片，
        并在底部和顶部加扇形盖，保证网格封闭。

        若传入 origin_2ch_mm / axis_u_2ch，则将局部坐标映射到全局 3D 坐标系。
        """
        num_theta = 32
        n_discs = len(bounds_2ch)

        def to3(v):
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size >= 3:
                return v[:3].copy()
            if v.size == 2:
                return np.array([v[0], v[1], 0.0])
            return np.zeros(3)

        use_global = (origin_2ch_mm is not None and axis_u_2ch is not None)
        if use_global:
            origin = to3(origin_2ch_mm)
            ax = to3(axis_u_2ch)
            ax /= np.linalg.norm(ax) + 1e-12
            basis_p1 = np.array([-ax[1], ax[0], 0.0])
            basis_p1 /= np.linalg.norm(basis_p1) + 1e-12
            basis_p2 = np.cross(ax, basis_p1)
            basis_p2 /= np.linalg.norm(basis_p2) + 1e-12

        vertices = []
        for i in range(n_discs):
            z_loc = i * h
            x_min, x_max = bounds_4ch[i]
            y_min, y_max = bounds_2ch[i]
            cx, rx = (x_max + x_min) / 2.0, max(0.0, (x_max - x_min) / 2.0)
            cy, ry = (y_max + y_min) / 2.0, max(0.0, (y_max - y_min) / 2.0)

            for theta in np.linspace(0, 2 * np.pi, num_theta, endpoint=False):
                vx = cx + rx * np.cos(theta)
                vy = cy + ry * np.sin(theta)
                if use_global:
                    pt = origin + vx * basis_p1 + vy * basis_p2 + z_loc * ax
                    vertices.append(pt.tolist())
                else:
                    vertices.append([vx, vy, z_loc])

        vertices = np.array(vertices, dtype=np.float32)

        # 侧壁面片
        faces = []
        for i in range(n_discs - 1):
            for j in range(num_theta):
                p1i = i * num_theta + j
                p2i = i * num_theta + (j + 1) % num_theta
                p3i = (i + 1) * num_theta + (j + 1) % num_theta
                p4i = (i + 1) * num_theta + j
                faces += [[3, p1i, p2i, p3i], [3, p1i, p3i, p4i]]

        # 底盖
        base_center = np.mean(vertices[:num_theta], axis=0)
        bc_idx = len(vertices)
        vertices = np.vstack([vertices, base_center.astype(np.float32)])
        for j in range(num_theta):
            faces.append([3, bc_idx, (j + 1) % num_theta, j])

        # 顶盖
        top_start = (n_discs - 1) * num_theta
        top_center = np.mean(vertices[top_start:top_start + num_theta], axis=0)
        tc_idx = len(vertices)
        vertices = np.vstack([vertices, top_center.astype(np.float32)])
        for j in range(num_theta):
            faces.append([3, tc_idx, top_start + j, top_start + (j + 1) % num_theta])

        return vertices, np.array(faces, dtype=np.int32).flatten()

    # ═══════════════════════════════════════════════════════════════════════════
    #  核心：直径数组计算
    # ═══════════════════════════════════════════════════════════════════════════
    def _diameters_from_mask(self, mask, spacing, return_debug=False):
        m = self._to_cavity_mask(mask)
        if self.keep_largest_cc:
            m = self._keep_largest_component(m)

        dbg = {}
        if m.sum() < 30:
            if return_debug:
                dbg["reason"] = "too_few_pixels"
                return np.zeros(self.n, float), 0.0, dbg
            return np.zeros(self.n, float), 0.0, {}

        dx, dy = float(spacing[0]), float(spacing[1])
        coords = np.column_stack(np.where(m))
        pts = np.column_stack([coords[:, 1] * dx, coords[:, 0] * dy]).astype(float)

        axis_u, apex_mm, annulus_mid_mm = self._axis_and_points(mask, spacing)
        if axis_u is None:
            axis_u = self._pca_first_component(pts)
            apex_mm = annulus_mid_mm = None

        perp_u = np.array([-axis_u[1], axis_u[0]], float)

        if annulus_mid_mm is not None and apex_mm is not None:
            origin_pt = np.asarray(annulus_mid_mm, float)
            apex_pt = np.asarray(apex_mm, float)
            if np.dot(apex_pt - origin_pt, axis_u) < 0:
                axis_u = -axis_u
                perp_u = np.array([-axis_u[1], axis_u[0]], float)

            centered = pts - origin_pt
            t = centered @ axis_u
            s = centered @ perp_u
            L = float(np.dot(apex_pt - origin_pt, axis_u))
            if L <= 1e-6:
                if return_debug:
                    dbg["reason"] = "degenerate_axis"
                    return np.zeros(self.n, float), 0.0, dbg
                return np.zeros(self.n, float), 0.0, {}

            in_range = (t >= 0.0) & (t <= L)
            if np.count_nonzero(in_range) < 30:
                if return_debug:
                    dbg["reason"] = "too_few_in_range"
                    return np.zeros(self.n, float), 0.0, dbg
                return np.zeros(self.n, float), 0.0, {}

            t, s = t[in_range], s[in_range]
            tmin, tmax = 0.0, L
        else:
            mean_pt = pts.mean(axis=0)
            centered = pts - mean_pt
            t = centered @ axis_u
            s = centered @ perp_u
            tmin, tmax = float(t.min()), float(t.max())
            L = tmax - tmin
            if L <= 1e-6:
                if return_debug:
                    dbg["reason"] = "degenerate_axis_pca"
                    return np.zeros(self.n, float), 0.0, dbg
                return np.zeros(self.n, float), 0.0, {}

        h = L / self.n
        centers = tmin + (np.arange(self.n) + 0.5) * h
        band_half = 0.5 * self.band_frac * h

        diam = np.zeros(self.n, float)
        valid = np.zeros(self.n, bool)
        bounds = np.zeros((self.n, 2), float)
        bounds_valid = np.zeros(self.n, bool)

        for i, c in enumerate(centers):
            band = (t >= c - band_half) & (t <= c + band_half)
            if np.count_nonzero(band) < self.min_band_points:
                continue
            smin, smax = float(s[band].min()), float(s[band].max())
            if smax > smin:
                bounds[i] = [smin, smax]
                bounds_valid[i] = True
            w = (self._robust_width_val(s[band])
                 if self.robust_width
                 else float(smax - smin))
            diam[i] = max(0.0, w)
            valid[i] = diam[i] > 0.0

        # 仅插值内部缺失，边界外置零
        n_valid = int(valid.sum())
        if n_valid >= 2 and not valid.all():
            x = np.arange(self.n)
            fv, lv = int(x[valid].min()), int(x[valid].max())
            inside = (~valid) & (x > fv) & (x < lv)
            if inside.any():
                diam[inside] = np.interp(x[inside], x[valid], diam[valid])
            if self.enforce_apex_zero:
                diam[x > lv] = 0.0
                diam[x < fv] = 0.0
        elif n_valid < 2:
            diam[:] = 0.0

        if return_debug:
            dbg.update({
                "L_mm": float(L), "h_mm": float(h),
                "tmin_tmax": [tmin, tmax],
                "band_half_mm": float(band_half),
                "diameters_mm": diam.tolist(),
                "valid_mask": valid.tolist(),
                "axis_u": axis_u.tolist(),
                "apex_mm": (apex_mm.tolist() if apex_mm is not None else None),
                "annulus_mid_mm": (annulus_mid_mm.tolist() if annulus_mid_mm is not None else None),
                "bounds_mm": bounds.tolist(),
                "bounds_valid": bounds_valid.tolist(),
            })
            return diam, float(L), dbg
        return diam, float(L), {}

    # ═══════════════════════════════════════════════════════════════════════════
    #  策略路由：统一入口
    # ═══════════════════════════════════════════════════════════════════════════
    def _axis_and_points(self, mask, spacing):
        """根据 annulus_strategy 选择对应算法，返回 (axis_u, apex_mm, annulus_mid_mm)"""
        if self.annulus_strategy == "polar":
            return self._axis_and_points_polar(mask, spacing)
        else:
            return self._axis_and_points_wall_la(mask, spacing)

    # ─────────────────────────────────────────────────────────────────────
    #  策略 A：极坐标法（polar）
    # ─────────────────────────────────────────────────────────────────────
    def _axis_and_points_polar(self, mask, spacing):
        """
        1. 以 LV cavity 质心为极点，提取边缘点的 r(θ)。
        2. 找 r(θ) 前三峰：最近两个 = 左右瓣环；第三个 = 心尖。
        3. axis_u = annulus_mid → apex 的单位向量。
        """
        annulus_r, annulus_l, annulus_mid = self._annulus_by_polar(mask, spacing)
        cav_pts = self._cavity_points_mm(mask, spacing)
        if cav_pts is None:
            return None, None, None

        if annulus_mid is None:
            annulus_mid = cav_pts.mean(axis=0).astype(float)
        annulus_mid = np.asarray(annulus_mid, float)

        apex_mm = None
        if annulus_l is not None and annulus_r is not None:
            edge_pts = self._edge_points_mm(mask, spacing)
            if edge_pts is not None:
                centroid = cav_pts.mean(axis=0)
                centered = edge_pts - centroid
                r = np.hypot(centered[:, 0], centered[:, 1])
                theta = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)
                sort_idx = np.argsort(theta)
                theta_s, r_s = theta[sort_idx], r[sort_idx]

                peaks = self._find_peaks(r_s, min_d=10, thr=0.08)
                if len(peaks) >= 3:
                    top3 = sorted(peaks, key=lambda p: r_s[p], reverse=True)[:3]
                    for pk in top3:
                        t_ = theta_s[pk]
                        rv = r_s[pk]
                        pt = np.array([centroid[0] + rv * np.cos(t_),
                                       centroid[1] + rv * np.sin(t_)])
                        if (np.linalg.norm(pt - annulus_l) > 10 and
                                np.linalg.norm(pt - annulus_r) > 10):
                            apex_mm = pt
                            break

        if apex_mm is None:
            diff = cav_pts - annulus_mid
            apex_mm = cav_pts[np.argmax((diff * diff).sum(axis=1))].astype(float)

        axis_u = apex_mm - annulus_mid
        nrm = float(np.linalg.norm(axis_u))
        if nrm <= 1e-12:
            return None, None, None
        return (axis_u / nrm).astype(float), apex_mm, annulus_mid

    def _annulus_by_polar(self, mask, spacing):
        """极坐标法：找 r(θ) 前三峰，取最近两个 = 左右瓣环。"""
        edge_pts = self._edge_points_mm(mask, spacing)
        cav_pts = self._cavity_points_mm(mask, spacing)
        if edge_pts is None or cav_pts is None:
            return None, None, None

        centroid = cav_pts.mean(axis=0)
        centered = edge_pts - centroid
        r = np.hypot(centered[:, 0], centered[:, 1])
        theta = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)
        sort_idx = np.argsort(theta)
        theta_s, r_s = theta[sort_idx], r[sort_idx]

        peaks = self._find_peaks(r_s, min_d=10, thr=0.08)
        if len(peaks) < 2:
            return None, None, centroid.astype(float)

        top3 = sorted(peaks, key=lambda p: r_s[p], reverse=True)[:3]
        # 亚像素精化
        refined = []
        for pk in top3:
            if 0 < pk < len(r_s) - 1:
                y0, y1, y2 = r_s[pk - 1], r_s[pk], r_s[pk + 1]
                a = (y0 + y2 - 2 * y1) / 2
                delta = (0.5 * (y0 - y2) / a) if abs(a) > 1e-10 else 0
                refined.append(pk + delta)
            else:
                refined.append(float(pk))

        pts = []
        for pk in refined:
            idx = int(pk)
            t_ = theta_s[idx]
            rv = r_s[idx]
            pts.append(np.array([centroid[0] + rv * np.cos(t_),
                                  centroid[1] + rv * np.sin(t_)]))

        if len(pts) >= 2:
            min_d, pair = float('inf'), (0, 1)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_d:
                        min_d, pair = d, (i, j)
            al, ar = pts[pair[0]], pts[pair[1]]
            mid = 0.5 * (al + ar)
            return ar.astype(float), al.astype(float), mid.astype(float)

        return None, None, centroid.astype(float)

    # ─────────────────────────────────────────────────────────────────────
    #  策略 B：Wall-LA 接触带法（wall_la）
    # ─────────────────────────────────────────────────────────────────────
    def _axis_and_points_wall_la(self, mask, spacing):
        """
        1. 利用 Wall(2)+LA(3) 接触带 + PCA 强制左右分区找瓣环。
        2. apex = 距 annulus_mid 最远的 cavity 点。
        3. axis_u = annulus_mid → apex。
        """
        _, _, annulus_mid = self._annulus_by_wall_la(mask, spacing)
        cav_pts = self._cavity_points_mm(mask, spacing)
        if cav_pts is None:
            return None, None, None

        if annulus_mid is None:
            annulus_mid = cav_pts.mean(axis=0).astype(float)
        annulus_mid = np.asarray(annulus_mid, float)

        diff = cav_pts - annulus_mid
        apex_mm = cav_pts[np.argmax((diff * diff).sum(axis=1))].astype(float)

        axis_u = apex_mm - annulus_mid
        nrm = float(np.linalg.norm(axis_u))
        if nrm <= 1e-12:
            return None, None, None
        return (axis_u / nrm).astype(float), apex_mm, annulus_mid

    def _annulus_by_wall_la(self, mask, spacing):
        """
        Wall(2)+LA(3) 接触带 + PCA 强制分区：
          - 找 LV_all(1|2) 与 LA(3) 扩张区的交集作为候选。
          - PCA 求长轴方向，垂直方向强制分左右区。
          - 各区取最外点为瓣环。
          - 若两点间距 < 15mm，则强制取 cavity 横截面极值点。
        """
        try:
            from scipy.ndimage import binary_dilation
        except Exception:
            binary_dilation = None

        arr = np.rint(np.asarray(mask)).astype(np.int16)
        lv = (arr == 1)
        wall = (arr == 2)
        la = (arr == 3)
        lv_all = lv | wall

        if lv.sum() < 30:
            return None, None, None

        dx, dy = float(spacing[0]), float(spacing[1])
        cav_coords = np.column_stack(np.where(lv))
        cav_pts = np.column_stack([cav_coords[:, 1] * dx, cav_coords[:, 0] * dy]).astype(float)
        mean_c = cav_pts.mean(axis=0)

        X = cav_pts - mean_c
        C = (X.T @ X) / max(1, X.shape[0] - 1)
        vals, vecs = np.linalg.eigh(C)
        v_long = vecs[:, np.argmax(vals)]
        v_perp = np.array([-v_long[1], v_long[0]])

        la_zone = binary_dilation(la, iterations=5) if binary_dilation else la
        cand_mask = lv_all & la_zone
        if cand_mask.sum() < 5:
            cand_mask = lv

        cand_coords = np.column_stack(np.where(cand_mask))
        cand_pts = np.column_stack([cand_coords[:, 1] * dx, cand_coords[:, 0] * dy]).astype(float)
        proj_perp = (cand_pts - mean_c) @ v_perp

        def best_outer(pts_sub, direction):
            if len(pts_sub) == 0:
                return None
            return pts_sub[np.argmax(pts_sub @ direction)]

        pL = best_outer(cand_pts[proj_perp < 0], -v_perp)
        pR = best_outer(cand_pts[proj_perp >= 0], v_perp)

        # 兜底
        if pL is None or pR is None:
            proj_all = (cav_pts - mean_c) @ v_perp
            if pL is None:
                sub = cav_pts[proj_all < 0]
                pL = sub[np.argmin(sub @ v_long)] if len(sub) > 0 else cav_pts[proj_all.argmin()]
            if pR is None:
                sub = cav_pts[proj_all >= 0]
                pR = sub[np.argmin(sub @ v_long)] if len(sub) > 0 else cav_pts[proj_all.argmax()]

        # 距离检查
        if np.linalg.norm(pL - pR) < 15.0:
            proj_all = (cav_pts - mean_c) @ v_perp
            pL = cav_pts[np.argmin(proj_all)]
            pR = cav_pts[np.argmax(proj_all)]

        mid = 0.5 * (pL + pR)
        return pR.astype(float), pL.astype(float), mid.astype(float)

    # ═══════════════════════════════════════════════════════════════════════════
    #  内部工具函数
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_result(self, EDV, ESV, EF, ed2, es2, ed4, es4, dbg_ed, dbg_es):
        def _g(dbg, view, key):
            v = dbg.get(view, {})
            val = v.get(key) if isinstance(v, dict) else None
            return np.asarray(val).tolist() if val is not None else None

        return dict(
            EDV=EDV, ESV=ESV, EF=EF, curve=[],
            ED_index=ed2, ES_index=es2,
            ED_index_4ch=ed4, ES_index_4ch=es4,
            axis_u_2ch_ed=_g(dbg_ed, "view2", "axis_u"),
            axis_u_4ch_ed=_g(dbg_ed, "view4", "axis_u"),
            axis_u_2ch_es=_g(dbg_es, "view2", "axis_u"),
            axis_u_4ch_es=_g(dbg_es, "view4", "axis_u"),
            apex_2ch_ed=_g(dbg_ed, "view2", "apex_mm"),
            annulus_mid_2ch_ed=_g(dbg_ed, "view2", "annulus_mid_mm"),
            apex_4ch_ed=_g(dbg_ed, "view4", "apex_mm"),
            annulus_mid_4ch_ed=_g(dbg_ed, "view4", "annulus_mid_mm"),
            apex_2ch_es=_g(dbg_es, "view2", "apex_mm"),
            annulus_mid_2ch_es=_g(dbg_es, "view2", "annulus_mid_mm"),
            apex_4ch_es=_g(dbg_es, "view4", "apex_mm"),
            annulus_mid_4ch_es=_g(dbg_es, "view4", "annulus_mid_mm"),
            bounds_2ch_ed=_g(dbg_ed, "view2", "bounds_mm"),
            bounds_4ch_ed=_g(dbg_ed, "view4", "bounds_mm"),
            h_mm_ed=dbg_ed.get("h_mm", 0.0),
        )

    # ── 单平面 Simpson 体积 ───────────────────────────────────────────────────
    def _singleplane_vol(self, mask, spacing, verbose=False) -> float:
        """单平面 Simpson：V = Σ (π/4 · dᵢ² · h)"""
        d, L, dbg = self._diameters_from_mask(mask, spacing, return_debug=True)
        h = L / self.n if L > 1e-6 else 0.0
        disc_vols = (np.pi / 4.0) * d * d * h
        vol_mm3 = float(np.sum(disc_vols))
        vol_mL = vol_mm3 / 1000.0
        if verbose:
            print(f"[DEBUG] 单平面 Simpson: L={L:.2f}mm  h={h:.4f}mm  V={vol_mL:.2f}mL")
        return vol_mL

    # ── 面积-长度法体积 ──────────────────────────────────────────────────────
    def _area_length_vol(self, mask, spacing, verbose=False) -> float:
        """
        面积-长度法：V = 8A² / (3π·L)
        A = LV 腔体面积(mm²)，L = 长轴长度(mm)
        """
        m = self._to_cavity_mask(mask)
        dx, dy = float(spacing[0]), float(spacing[1])
        A_mm2 = float(m.sum()) * dx * dy         # 像素数 × 像素面积

        _, L, _ = self._diameters_from_mask(mask, spacing, return_debug=True)
        if L <= 1e-6 or A_mm2 <= 1e-6:
            return 0.0

        vol_mm3 = (8.0 * A_mm2 * A_mm2) / (3.0 * np.pi * L)
        vol_mL = vol_mm3 / 1000.0
        if verbose:
            print(f"[DEBUG] 面积-长度法: A={A_mm2:.1f}mm²  L={L:.2f}mm  V={vol_mL:.2f}mL")
        return vol_mL

    def _to_cavity_mask(self, mask):
        arr = np.asarray(mask)
        if arr.dtype.kind not in ("i", "u", "b"):
            arr = np.rint(arr).astype(np.int16)
        return arr == self.cavity_label

    def _keep_largest_component(self, m):
        """保留最大连通域（当前为 pass-through，可按需启用）。
        注意：若要启用，需确认 scipy.ndimage.label 已成功导入（见文件顶部 cc_label）。
        """
        return m
        # 如需启用：
        # if cc_label is None: return m
        # lab, n = cc_label(m.astype(np.uint8))
        # if n <= 1: return m
        # counts = np.bincount(lab.ravel()); counts[0] = 0
        # return lab == int(np.argmax(counts))

    def _pca_first_component(self, pts):
        X = pts - pts.mean(axis=0)
        C = (X.T @ X) / max(1, X.shape[0] - 1)
        vals, vecs = np.linalg.eigh(C)
        u = vecs[:, np.argmax(vals)]
        return (u / (np.linalg.norm(u) + 1e-12)).astype(float)

    def _robust_width_val(self, s_vals):
        s_vals = np.asarray(s_vals, float)
        if s_vals.size < 10:
            return 0.0
        return max(0.0, float(np.percentile(s_vals, 95)) - float(np.percentile(s_vals, 5)))

    def _find_peaks(self, data, min_d=10, thr=0.08):
        """简单极大值检测。"""
        peaks = []
        threshold = thr * (data.max() - data.min()) + data.min()
        for i in range(min_d, len(data) - min_d):
            lo, hi = max(0, i - min_d), min(len(data), i + min_d)
            if data[i] >= threshold and data[i] >= data[lo:hi].max():
                peaks.append(i)
        return peaks

    def _edge_points_mm(self, mask, spacing):
        """提取 LV cavity(1) 的边缘点（优先 cv2 轮廓，兜底 erosion）。"""
        m = self._to_cavity_mask(mask)
        if m.sum() < 30:
            return None
        dx, dy = float(spacing[0]), float(spacing[1])
        try:
            import cv2
            lv_u8 = m.astype(np.uint8)
            contours, _ = cv2.findContours(lv_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                pts = cnt[:, 0, :].astype(float)
                return np.column_stack([pts[:, 0] * dx, pts[:, 1] * dy])
        except Exception:
            pass
        try:
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(m, iterations=2)
            edge = m ^ eroded
        except Exception:
            edge = m
        coords = np.column_stack(np.where(edge))
        return np.column_stack([coords[:, 1] * dx, coords[:, 0] * dy]).astype(float)

    def _cavity_points_mm(self, mask, spacing):
        """返回 LV cavity(1) 全部像素的 mm 坐标。"""
        m = self._to_cavity_mask(mask)
        if self.keep_largest_cc:
            m = self._keep_largest_component(m)
        if m.sum() < 30:
            return None
        dx, dy = float(spacing[0]), float(spacing[1])
        coords = np.column_stack(np.where(m))
        return np.column_stack([coords[:, 1] * dx, coords[:, 0] * dy]).astype(float)