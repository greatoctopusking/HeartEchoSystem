import numpy as np

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

class BiplaneSimpsonClinical:
    def __init__(
        self,
        n_discs: int = 20,
        cavity_label: int = 1,
        band_frac: float = 1,          # ✅ 放?宽 band，避免前期直径下降过快
        min_band_points: int = 3,        # ✅ apex 像素少，降低阈值
        robust_width: bool = False,      # True: percentile width; False: max-min,不确定用没用到
        enforce_apex_zero: bool = True,  # ✅ 不做边界外延，尾部缺失直接置0
        keep_largest_cc: bool = True,
    ):
        if n_discs < 5:
            raise ValueError("n_discs too small; use >= 5 (typical 20).")

        self.n = int(n_discs)
        self.cavity_label = int(cavity_label)
        self.band_frac = float(band_frac)
        self.min_band_points = int(min_band_points)
        self.robust_width = bool(robust_width)
        self.enforce_apex_zero = bool(enforce_apex_zero)
        self.keep_largest_cc = bool(keep_largest_cc)



    # ===============================================================
    # 单帧容积（✅ 只在 verbose=True 时打印逐圆盘；曲线阶段不刷屏）
    # ===============================================================
    def volume_single_frame(self, mask2, mask4, spacing2, spacing4, return_debug=False, verbose=False):
        # 提取直径数组与长轴
        d2, L2, dbg2 = self._diameters_from_mask(mask2, spacing2, return_debug=True)
        d4, L4, dbg4 = self._diameters_from_mask(mask4, spacing4, return_debug=True)

        # 长轴平均与盘高
        L = 0.5 * (L2 + L4)
        h = L / self.n if L > 1e-6 else 0.0

        # Simpson 双平面：Σ(π/4 * d2 * d4 * h)
        disc_volumes = (np.pi / 4.0) * d2 * d4 * h
        vol_mm3 = float(np.sum(disc_volumes))
        vol_mL = float(vol_mm3 / 1000.0)

        # ✅ 只在 ED/ES 调试时打印
        if verbose:
            print("\n" + "=" * 80)
            print(f"[DEBUG] 单帧容积计算 - 辛普森双平面公式")
            print(f"[DEBUG] 基础参数：L2={L2:.4f}mm, L4={L4:.4f}mm, Lavg={L:.4f}mm, n_discs={self.n}, h={h:.4f}mm")

            # 这里按你之前习惯：全数组打印（避免尾部异常看不到）
            print(f"[DEBUG] d2数组（2CH各圆盘直径）: {d2}... (长度={len(d2)})")
            print(f"[DEBUG] d4数组（4CH各圆盘直径）: {d4}... (长度={len(d4)})")

            print(f"[DEBUG] 逐圆盘容积计算（π/4 × d2 × d4 × h）：")
            for i in range(self.n):
                print(
                    f"[DEBUG] 圆盘{i+1:2d}: π/4 × {d2[i]:.4f} × {d4[i]:.4f} × {h:.4f} = {disc_volumes[i]:.4f} mm³"
                )

            print(f"[DEBUG] 总容积汇总：")
            print(f"[DEBUG] vol_mm3 = Σ(π/4 × d2 × d4 × h) = {vol_mm3:.4f} mm³")
            print(f"[DEBUG] vol_mL  = vol_mm3 / 1000 = {vol_mL:.4f} mL")
            print("=" * 80 + "\n")

        if return_debug:
            return vol_mL, {
                "view2": dbg2,
                "view4": dbg4,
                "L2_mm": float(L2),
                "L4_mm": float(L4),
                "L_avg_mm": float(L),
                "h_mm": float(h),
                "d2_mm": d2.astype(float).tolist(),
                "d4_mm": d4.astype(float).tolist(),
                "vol_mm3": float(vol_mm3),
                "vol_mL": float(vol_mL),
                "disc_volumes_mm3": disc_volumes.astype(float).tolist(),
            }

        return vol_mL

    # ===============================================================
    # 序列 ED/ES（保持你原来的输出：像素曲线、shift、volume curve、最终ED/ES/EF）
    # 额外：✅ 只对 ED/ES 两帧打印逐圆盘 debug
    # ===============================================================
    def compute_ed_es_from_series(self, masks2, masks4, spacing2, spacing4):
        T2 = len(masks2)
        T4 = len(masks4)

        if T2 == 0 or T4 == 0:
            return dict(EDV=0.0, ESV=0.0, EF=0.0,
                        ED_index=-1, ES_index=-1, curve=[])

        # ===================== 像素数曲线 =====================
        area2_LV = np.array([np.sum(self._to_cavity_mask(m)) for m in masks2], dtype=float)
        area4_LV = np.array([np.sum(self._to_cavity_mask(m)) for m in masks4], dtype=float)

        print(f"[INFO] 2CH - LV (1) 像素数曲线: {area2_LV}")
        print(f"[INFO] 4CH - LV (1) 像素数曲线: {area4_LV}")

        # ===================== 各自找 ED / ES =====================
        ED_index_2ch = int(np.argmax(area2_LV))
        ES_index_2ch = int(np.argmin(area2_LV))

        ED_index_4ch = int(np.argmax(area4_LV))
        ES_index_4ch = int(np.argmin(area4_LV))

        print(f"[INFO] 2CH自身ED帧: {ED_index_2ch}, ES帧: {ES_index_2ch}")
        print(f"[INFO] 4CH自身ED帧: {ED_index_4ch}, ES帧: {ES_index_4ch}")

        # ===================== 直接计算 EDV / ESV（无对齐） =====================

        print("\n[DEBUG] ===================== ED FRAME (2CH/4CH) =====================")
        EDV, dbg_ed = self.volume_single_frame(
            masks2[ED_index_2ch],
            masks4[ED_index_4ch],
            spacing2,
            spacing4,
            return_debug=True,
            verbose=True
        )

        print("\n[DEBUG] ===================== ES FRAME (2CH/4CH) =====================")
        ESV , dbg_es= self.volume_single_frame(
            masks2[ES_index_2ch],
            masks4[ES_index_4ch],
            spacing2,
            spacing4,
            return_debug=True,
            verbose=True
        )

        EF = float((EDV - ESV) / EDV * 100.0) if EDV > 1e-6 else 0.0

        print(f"[INFO] 最终 - 2CH: ED frame index = {ED_index_2ch}, ES frame index = {ES_index_2ch}")
        print(f"[INFO] 最终 - 4CH: ED frame index = {ED_index_4ch}, ES frame index = {ES_index_4ch}")
        print(f"[INFO] EDV={EDV:.2f}, ESV={ESV:.2f}, EF={EF:.2f}")

        return dict(
            EDV=EDV,
            ESV=ESV,
            EF=EF,
            ED_index=ED_index_2ch,
            ES_index=ES_index_2ch,
            curve=[],  # 不再使用体积曲线
            ED_index_4ch=ED_index_4ch,
            ES_index_4ch=ES_index_4ch,
            axis_u_2ch_ed=dbg_ed["view2"].get("axis_u"),
            axis_u_4ch_ed=dbg_ed["view4"].get("axis_u"),
            axis_u_2ch_es=dbg_es["view2"].get("axis_u"),
            axis_u_4ch_es=dbg_es["view4"].get("axis_u"),

            apex_2ch_ed=dbg_ed["view2"].get("apex_mm"),
            annulus_mid_2ch_ed=dbg_ed["view2"].get("annulus_mid_mm"),
            apex_4ch_ed=dbg_ed["view4"].get("apex_mm"),
            annulus_mid_4ch_ed=dbg_ed["view4"].get("annulus_mid_mm"),

            apex_2ch_es=dbg_es["view2"].get("apex_mm"),
            annulus_mid_2ch_es=dbg_es["view2"].get("annulus_mid_mm"),
            apex_4ch_es=dbg_es["view4"].get("apex_mm"),
            annulus_mid_4ch_es=dbg_es["view4"].get("annulus_mid_mm"),

            # ====== ✅ 3D 必需（只带 ED 一次即可）======
            bounds_2ch_ed=dbg_ed["view2"].get("bounds_mm"),
            bounds_4ch_ed=dbg_ed["view4"].get("bounds_mm"),
            h_mm_ed=dbg_ed.get("h_mm"),
        )

    # ===============================================================
    # 直径计算（✅ 修复 centers 对齐 + 只插值内部缺失、边界不外延）
    # ===============================================================
    def _diameters_from_mask(self, mask, spacing, return_debug=False):
        m = self._to_cavity_mask(mask)
        if self.keep_largest_cc:
            m = self._keep_largest_component(m)

        dbg = {}
        if m.sum() < 30:
            if return_debug:
                dbg["reason"] = "too_few_pixels_after_filter"
                return np.zeros(self.n, dtype=float), 0.0, dbg
            return np.zeros(self.n, dtype=float), 0.0, {}

        dx, dy = float(spacing[0]), float(spacing[1])
        coords = np.column_stack(np.where(m))
        ys = coords[:, 0].astype(float)
        xs = coords[:, 1].astype(float)
        pts = np.column_stack([xs * dx, ys * dy])

        # 每帧用“annulus_mid -> apex”定义长轴（新版）
        axis_u, apex_mm, annulus_mid_mm = self._axis_and_points_from_mask_annulus_apex(mask, spacing)
        if axis_u is None:
            axis_u = self._pca_first_component(pts)
            apex_mm, annulus_mid_mm = None, None

        perp_u = np.array([-axis_u[1], axis_u[0]], dtype=float)

        # ===== 统一坐标系：优先用 annulus_mid 作为原点（与 overlay 一致）=====
        if annulus_mid_mm is not None and apex_mm is not None:
            origin_pt = np.asarray(annulus_mid_mm, dtype=float)
            apex_pt = np.asarray(apex_mm, dtype=float)

            # 保证 axis_u 指向 apex
            if float(np.dot(apex_pt - origin_pt, axis_u)) < 0:
                axis_u = -axis_u
                perp_u = np.array([-axis_u[1], axis_u[0]], dtype=float)

            centered = pts - origin_pt
            t = centered @ axis_u
            s = centered @ perp_u

            # 定义长轴长度：瓣环->心尖（0..L）
            L = float(np.dot(apex_pt - origin_pt, axis_u))
            if L <= 1e-6:
                if return_debug:
                    dbg["reason"] = "degenerate_axis"
                    return np.zeros(self.n, dtype=float), 0.0, dbg
                return np.zeros(self.n, dtype=float), 0.0, {}

            # 只保留投影落在 [0, L] 的点，避免 base 外/噪声污染
            in_range = (t >= 0.0) & (t <= L)
            if np.count_nonzero(in_range) < 30:
                if return_debug:
                    dbg["reason"] = "too_few_pixels_in_0L"
                    return np.zeros(self.n, dtype=float), 0.0, dbg
                return np.zeros(self.n, dtype=float), 0.0, {}

            t = t[in_range]
            s = s[in_range]

            tmin, tmax = 0.0, L
            h = L / self.n
        else:
            # 兜底：旧版（质心为原点）
            mean_pt = pts.mean(axis=0)
            centered = pts - mean_pt
            t = centered @ axis_u
            s = centered @ perp_u

            tmin, tmax = float(np.min(t)), float(np.max(t))
            L = tmax - tmin
            if L <= 1e-6:
                if return_debug:
                    dbg["reason"] = "degenerate_axis"
                    return np.zeros(self.n, dtype=float), 0.0, dbg
                return np.zeros(self.n, dtype=float), 0.0, {}
            h = L / self.n

        # ===== 分层中心与 band 半宽（centers 必须在 for 循环前定义）=====
        h = L / self.n
        centers = tmin + (np.arange(self.n) + 0.5) * h
        band_half = 0.5 * self.band_frac * h

        diam = np.zeros(self.n, dtype=float)
        valid = np.zeros(self.n, dtype=bool)

        # ====== ✅ 新增：每层边界（给 3D 用），不改变你原直径逻辑 ======
        bounds = np.zeros((self.n, 2), dtype=float)
        bounds_valid = np.zeros(self.n, dtype=bool)

        #for i, c in enumerate(centers):
        for i, c in enumerate(centers):
            band = (t >= c - band_half) & (t <= c + band_half)
            if np.count_nonzero(band) < self.min_band_points:
                continue

            # bounds（smin/smax）
            smin = float(np.min(s[band]))
            smax = float(np.max(s[band]))
            if smax > smin:
                bounds[i, 0] = smin
                bounds[i, 1] = smax
                bounds_valid[i] = True

            if self.robust_width:
                w = self._robust_width_val(s[band])
            else:
                w = float(np.max(s[band]) - np.min(s[band]))

            diam[i] = max(0.0, w)
            valid[i] = diam[i] > 0.0

        # ✅ 只插值内部缺失；边界不外延（避免尾部“全一样”）
        n_valid = int(np.count_nonzero(valid))
        if n_valid >= 2 and not np.all(valid):
            x = np.arange(self.n)
            first_valid = int(np.min(np.where(valid)))
            last_valid = int(np.max(np.where(valid)))

            inside = (~valid) & (x > first_valid) & (x < last_valid)
            #只补 inside，不补边界，就不会产生“尾部常数外延”。
            if np.any(inside):
                diam[inside] = np.interp(x[inside], x[valid], diam[valid])

            if self.enforce_apex_zero:
                diam[x > last_valid] = 0.0
                diam[x < first_valid] = 0.0

        elif n_valid < 2:
            diam[:] = 0.0

        if return_debug:
            dbg.update({
                "L_mm": float(L),
                "h_mm": float(h),
                "tmin_tmax": [tmin, tmax],
                "band_half_mm": float(band_half),
                "diameters_mm": diam.astype(float).tolist(),
                "valid_mask": valid.astype(bool).tolist(),
                "axis_u": axis_u.astype(float).tolist(),
                "apex_mm": apex_mm.astype(float).tolist() if apex_mm is not None else None,
                "annulus_mid_mm": annulus_mid_mm.astype(float).tolist() if annulus_mid_mm is not None else None,

                # ====== ✅ 新增输出：给 app.py 生成 3D mesh 用 ======
                "bounds_mm": bounds.astype(float).tolist(),
                "bounds_valid": bounds_valid.astype(bool).tolist(),
            })
            return diam, float(L), dbg

        return diam, float(L), {}


    def frame_bounds_and_L(self, mask2, mask4, spacing2, spacing4):
        d2, L2, dbg2 = self._diameters_from_mask(mask2, spacing2, return_debug=True)
        d4, L4, dbg4 = self._diameters_from_mask(mask4, spacing4, return_debug=True)

        L = 0.5 * (float(L2) + float(L4))
        h = L / float(self.n) if L > 1e-6 else 0.0

        return {
            "bounds_2ch": dbg2.get("bounds_mm"),
            "bounds_4ch": dbg4.get("bounds_mm"),
            "L2_mm": float(L2),
            "L4_mm": float(L4),
            "L_avg_mm": float(L),
            "h_mm": float(h),
            # ========== 新增：传递原始坐标信息 ==========
            "origin_2ch_mm": dbg2.get("annulus_mid_mm"),  # 瓣环中点（原始2D坐标）
            "axis_u_2ch": dbg2.get("axis_u"),            # 长轴方向（原始2D向量）
            "origin_4ch_mm": dbg4.get("annulus_mid_mm"),
            "axis_u_4ch": dbg4.get("axis_u"),
        }


    # ===============================================================
    # 工具函数
    # ===============================================================
    def _to_cavity_mask(self, mask):
        arr = np.asarray(mask)
        if arr.dtype.kind not in ("i", "u", "b"):
            arr = np.rint(arr).astype(np.int16)
        return (arr == self.cavity_label)

    def _keep_largest_component(self, m):

        return m

    def _keep_largest_component1(self, m):
        if cc_label is None:
            return m
        lab, n = cc_label(m.astype(np.uint8))
        if n <= 1:
            return m
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = int(np.argmax(counts))
        m2 = (lab == keep)

        before = int(np.sum(m))
        after = int(np.sum(m2))
        if before > 0 and after < 0.8 * before:
            print("[WARN] keep_largest_component removed too much:", before, "->", after)

        return m2

    def _pca_first_component(self, pts):
        X = pts - pts.mean(axis=0)
        C = (X.T @ X) / max(1, X.shape[0] - 1)
        vals, vecs = np.linalg.eigh(C)
        u = vecs[:, int(np.argmax(vals))]
        u = u / (np.linalg.norm(u) + 1e-12)
        return u.astype(float)

    def _robust_width_val(self, s_vals):
        s_vals = np.asarray(s_vals, dtype=float)
        if s_vals.size < 10:
            return 0.0
        lo = float(np.percentile(s_vals, 5))
        hi = float(np.percentile(s_vals, 95))
        return max(0.0, hi - lo)

    def _axis_from_mask_annulus_apex(self, mask, spacing):
        axis_u, _, _ = self._axis_and_points_from_mask_annulus_apex(mask, spacing)
        return axis_u

    def _axis_and_points_from_mask_annulus_apex(self, mask, spacing):
        """
        新算法：基于边缘和极坐标
        1) 用极坐标法找瓣环左右两点 -> annulus_mid
        2) apex = 排除瓣环点后的剩余峰值点
        3) axis_u = annulus_mid -> apex 的单位向量
        返回: (axis_u, apex_mm, annulus_mid_mm)
        """
        annulus_right_mm, annulus_left_mm, annulus_mid_mm = self._annulus_points_from_wall_la(mask, spacing)
        
        cav_pts = self._cavity_points_mm(mask, spacing)
        if cav_pts is None:
            return None, None, None
        
        if annulus_mid_mm is None:
            annulus_mid_mm = cav_pts.mean(axis=0).astype(float)
        
        annulus_mid_mm = np.asarray(annulus_mid_mm, dtype=float)
        
        apex_mm = None
        
        if annulus_left_mm is not None and annulus_right_mm is not None:
            try:
                import cv2
            except Exception:
                cv2 = None
            
            try:
                from scipy.ndimage import binary_erosion
            except Exception:
                binary_erosion = None
            
            arr = np.rint(np.asarray(mask)).astype(np.int16)
            lv = (arr == 1)
            dx, dy = float(spacing[0]), float(spacing[1])
            
            cav_coords = np.column_stack(np.where(lv))
            cav_pts_mm = cav_coords[:, [1, 0]] * [dx, dy]
            centroid_mm = cav_pts_mm.mean(axis=0)
            
            edge_pts_mm = None
            if cv2 is not None:
                lv_uint8 = lv.astype(np.uint8)
                contours, _ = cv2.findContours(lv_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    edge_pts = largest_contour[:, 0, :].astype(float)
                    edge_pts_mm = edge_pts * [dx, dy]
            
            if edge_pts_mm is None:
                if binary_erosion is not None:
                    eroded = binary_erosion(lv, iterations=2)
                    edge = lv ^ eroded
                else:
                    edge = lv
                edge_coords = np.column_stack(np.where(edge))
                edge_pts_mm = edge_coords[:, [1, 0]] * [dx, dy]
            
            centered_pts = edge_pts_mm - centroid_mm
            r = np.sqrt(centered_pts[:, 0]**2 + centered_pts[:, 1]**2)
            theta = np.arctan2(centered_pts[:, 1], centered_pts[:, 0])
            theta = np.mod(theta, 2 * np.pi)
            
            sort_idx = np.argsort(theta)
            theta_sorted = theta[sort_idx]
            r_sorted = r[sort_idx]
            
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
            
            if len(top3_peaks) >= 3:
                apex_peak_idx = None
                for i, peak_idx in enumerate(top3_peaks):
                    idx = int(peak_idx)
                    t = theta_sorted[idx]
                    r_val = r_sorted[idx]
                    peak_pt = np.array([
                        centroid_mm[0] + r_val * np.cos(t),
                        centroid_mm[1] + r_val * np.sin(t)
                    ])
                    dist_to_left = np.linalg.norm(peak_pt - annulus_left_mm)
                    dist_to_right = np.linalg.norm(peak_pt - annulus_right_mm)
                    if dist_to_left > 10 and dist_to_right > 10:
                        apex_peak_idx = i
                        break
                
                if apex_peak_idx is not None:
                    apex_idx = int(top3_peaks[apex_peak_idx])
                    t = theta_sorted[apex_idx]
                    r_val = r_sorted[apex_idx]
                    apex_mm = np.array([
                        centroid_mm[0] + r_val * np.cos(t),
                        centroid_mm[1] + r_val * np.sin(t)
                    ])
        
        if apex_mm is None:
            diff = cav_pts - annulus_mid_mm
            dist2 = np.sum(diff * diff, axis=1)
            apex_mm = cav_pts[int(np.argmax(dist2))].astype(float)
        
        axis_u = apex_mm - annulus_mid_mm
        nrm = float(np.linalg.norm(axis_u))
        if nrm <= 1e-12:
            return None, None, None

        axis_u = axis_u / nrm
        return axis_u.astype(float), apex_mm, annulus_mid_mm

    def _annulus_points_from_wall_la(self, mask, spacing):
        """
        新算法：基于边缘和极坐标
        1. 找到质心位置
        2. 提取LV的边缘
        3. 以质心为中心作坐标系，r(theta)
        4. 找到3个峰值，取距离最近的2个作为瓣环点
        
        返回: (annulus_right_mm, annulus_left_mm, annulus_mid_mm)
        """
        try:
            import cv2
        except Exception:
            cv2 = None
        
        try:
            from scipy.ndimage import binary_erosion
        except Exception:
            binary_erosion = None
        
        arr = np.rint(np.asarray(mask)).astype(np.int16)
        lv = (arr == 1)
        
        if np.count_nonzero(lv) < 30:
            return None, None, None
        
        dx, dy = float(spacing[0]), float(spacing[1])
        
        cav_coords = np.column_stack(np.where(lv))
        cav_pts_mm = cav_coords[:, [1, 0]] * [dx, dy]
        centroid_mm = cav_pts_mm.mean(axis=0)
        
        edge_pts_mm = None
        if cv2 is not None:
            lv_uint8 = lv.astype(np.uint8)
            contours, _ = cv2.findContours(lv_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                edge_pts = largest_contour[:, 0, :].astype(float)
                edge_pts_mm = edge_pts * [dx, dy]
        
        if edge_pts_mm is None:
            if binary_erosion is not None:
                eroded = binary_erosion(lv, iterations=2)
                edge = lv ^ eroded
            else:
                edge = lv
            edge_coords = np.column_stack(np.where(edge))
            edge_pts_mm = edge_coords[:, [1, 0]] * [dx, dy]
        
        centered_pts = edge_pts_mm - centroid_mm
        r = np.sqrt(centered_pts[:, 0]**2 + centered_pts[:, 1]**2)
        theta = np.arctan2(centered_pts[:, 1], centered_pts[:, 0])
        theta = np.mod(theta, 2 * np.pi)
        
        sort_idx = np.argsort(theta)
        theta_sorted = theta[sort_idx]
        r_sorted = r[sort_idx]
        
        window_size = 15
        r_smoothed = np.convolve(r_sorted, np.ones(window_size)/window_size, mode='same')
        
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
        
        top3_peaks = refined_peaks
        
        annulus_left_mm = None
        annulus_right_mm = None
        
        if len(top3_peaks) >= 2:
            peak_points = []
            for peak_idx in top3_peaks:
                idx = int(peak_idx)
                t = theta_sorted[idx]
                r_val = r_sorted[idx]
                x = centroid_mm[0] + r_val * np.cos(t)
                y = centroid_mm[1] + r_val * np.sin(t)
                peak_points.append(np.array([x, y]))
            
            min_dist = float('inf')
            min_pair = (0, 1)
            for i in range(len(peak_points)):
                for j in range(i+1, len(peak_points)):
                    dist = np.linalg.norm(peak_points[i] - peak_points[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (i, j)
            
            annulus_left_mm = peak_points[min_pair[0]]
            annulus_right_mm = peak_points[min_pair[1]]
        
        if annulus_left_mm is not None and annulus_right_mm is not None:
            annulus_mid_mm = 0.5 * (annulus_left_mm + annulus_right_mm)
        else:
            annulus_mid_mm = centroid_mm
        
        return annulus_right_mm, annulus_left_mm, annulus_mid_mm


    def _cavity_points_mm(self, mask, spacing):
        """返回 LV cavity(1) 的点集 (N,2) mm 坐标；失败返回 None"""
        m = self._to_cavity_mask(mask)
        if self.keep_largest_cc:
            m = self._keep_largest_component(m)
        if np.count_nonzero(m) < 30:
            return None

        dx, dy = float(spacing[0]), float(spacing[1])
        coords = np.column_stack(np.where(m))
        ys = coords[:, 0].astype(float)
        xs = coords[:, 1].astype(float)
        pts = np.column_stack([xs * dx, ys * dy])
        return pts

    def _cavity_boundary_points_mm(self, mask, spacing):
        m = self._to_cavity_mask(mask)
        if np.count_nonzero(m) < 30:
            return None

        import cv2

        m_uint = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        # 取最大轮廓
        cnt = max(contours, key=lambda x: len(x))
        cnt = cnt[:, 0, :]  # (N,2) col,row

        dx, dy = float(spacing[0]), float(spacing[1])
        xs = cnt[:, 0].astype(float)
        ys = cnt[:, 1].astype(float)

        pts = np.column_stack([xs * dx, ys * dy])
        return pts



    def generate_3d_mesh_asymmetric(
        self, bounds_2ch, bounds_4ch, h,
        origin_2ch_mm=None, axis_u_2ch=None,
        origin_4ch_mm=None, axis_u_4ch=None
    ):
        print("[DBG] origin_2ch_mm type/shape:", type(origin_2ch_mm), np.asarray(origin_2ch_mm).shape)
        print("[DBG] axis_u_2ch type/shape:", type(axis_u_2ch), np.asarray(axis_u_2ch).shape)
        vertices = []
        num_theta = 32
        n_discs = len(bounds_2ch)

        def to3(v2):
            """把 [x,y] 或 (2,) 变成 [x,y,0]；已经是3维就原样返回"""
            v = np.asarray(v2, dtype=float).reshape(-1)
            if v.size >= 3:
                return np.array([v[0], v[1], v[2]], dtype=float)
            elif v.size == 2:
                return np.array([v[0], v[1], 0.0], dtype=float)
            else:
                return np.array([0.0, 0.0, 0.0], dtype=float)

        # ========= 构建“全局坐标系”基底（全部 3D）=========
        use_global = (origin_2ch_mm is not None and axis_u_2ch is not None)


        # 初始化变量，避免LSP警告（实际使用受use_global控制）
        origin = None
        axis_u = None
        perp_u1 = None
        perp_u2 = None

        if use_global:
            origin = to3(origin_2ch_mm)

            axis_u = to3(axis_u_2ch)
            axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)

            perp_u1 = np.array([-axis_u[1], axis_u[0], 0.0], dtype=float)
            perp_u1 = perp_u1 / (np.linalg.norm(perp_u1) + 1e-12)

            perp_u2 = np.cross(axis_u, perp_u1)
            perp_u2 = perp_u2 / (np.linalg.norm(perp_u2) + 1e-12)

        # ========= 逐层生成 =========
        for i in range(n_discs):
            z_local = i * h

            x_min, x_max = bounds_4ch[i]
            y_min, y_max = bounds_2ch[i]
            cx, rx = (x_max + x_min) / 2.0, (x_max - x_min) / 2.0
            cy, ry = (y_max + y_min) / 2.0, (y_max - y_min) / 2.0

            if rx <= 0 or ry <= 0:
                rx = ry = 0.0

            for theta in np.linspace(0, 2 * np.pi, num_theta, endpoint=False):
                vx_local = cx + rx * np.cos(theta)
                vy_local = cy + ry * np.sin(theta)

                if use_global:
                    v_global = (
                        origin
                        + vx_local * perp_u1
                        + vy_local * perp_u2
                        + z_local * axis_u
                    )
                    vertices.append(v_global.tolist())
                else:
                    vertices.append([vx_local, vy_local, z_local])

        vertices = np.array(vertices, dtype=np.float32)

        faces = []
        for i in range(n_discs - 1):
            for j in range(num_theta):
                p1 = i * num_theta + j
                p2 = i * num_theta + (j + 1) % num_theta
                p3 = (i + 1) * num_theta + (j + 1) % num_theta
                p4 = (i + 1) * num_theta + j
                faces.append([3, p1, p2, p3])
                faces.append([3, p1, p3, p4])

        return vertices, np.array(faces, dtype=np.int32).flatten()

