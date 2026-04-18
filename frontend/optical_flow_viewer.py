# optical_flow_viewer.py
import os
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSlider
)

try:
    import nibabel as nib
except ImportError:
    nib = None


# -----------------------------
# 基础工具
# -----------------------------
def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    """将任意范围数组归一化到 0-255 uint8"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [1, 99])

    if hi <= lo:
        mn, mx = float(vals.min()), float(vals.max())
        if mx <= mn:
            return np.zeros_like(arr, dtype=np.uint8)
        out = (arr - mn) / (mx - mn)
    else:
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


def _largest_contour(binary_mask: np.ndarray):
    """取最大轮廓"""
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _contour_band_mask(binary_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """生成轮廓带 mask，只允许在 LV 轮廓附近取点"""
    h, w = binary_mask.shape[:2]
    band = np.zeros((h, w), dtype=np.uint8)
    cnt = _largest_contour(binary_mask)
    if cnt is None:
        return band
    cv2.drawContours(band, [cnt], -1, 255, thickness=thickness)
    return band


def _draw_lv_contour(canvas_bgr: np.ndarray, binary_mask: np.ndarray):
    """画 LV 轮廓"""
    cnt = _largest_contour(binary_mask)
    if cnt is not None:
        cv2.drawContours(canvas_bgr, [cnt], -1, (0, 255, 255), 1)  # 黄色轮廓


def _to_qpixmap(bgr_img: np.ndarray) -> QPixmap:
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -----------------------------
# 数据读取
# -----------------------------
def _load_frames(path: str, max_frames: int = 120):
    """
    读取原始图像帧
    支持:
      - 视频: avi/mp4/mov
      - NIfTI: (H, W, T) 或 squeeze 后为 (H, W, T)
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    lower = path.lower()
    frames = []

    if lower.endswith((".avi", ".mp4", ".mov")):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)

        cap.release()

    elif lower.endswith((".nii", ".nii.gz")):
        if nib is None:
            raise ImportError("请先安装 nibabel: pip install nibabel")

        img = nib.load(path)
        data = np.asarray(img.get_fdata())
        print(f"[OpticalFlow] 图像原始 shape: {data.shape}")

        if data.ndim == 3:
            T = data.shape[-1]
            frames = [_normalize_to_u8(data[:, :, i]) for i in range(T)]

        elif data.ndim == 4:
            squeezed = np.squeeze(data)
            if squeezed.ndim != 3:
                raise ValueError(f"不支持的 4D 数据维度: {data.shape}")
            T = squeezed.shape[-1]
            frames = [_normalize_to_u8(squeezed[:, :, i]) for i in range(T)]

        else:
            raise ValueError(f"不支持的 NIfTI 维度: {data.shape}")

    else:
        raise ValueError("仅支持 AVI/MP4/MOV/NII/NII.GZ")

    if not frames:
        raise ValueError("未读取到有效图像帧")

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    return frames


def _resolve_nifti_path(mask_path: str) -> str:
    import os

    if os.path.exists(mask_path):
        return mask_path

    if mask_path.endswith(".nii.gz"):
        alt = mask_path[:-3]   # 去掉 .gz -> .nii
        if os.path.exists(alt):
            return alt

    elif mask_path.endswith(".nii"):
        alt = mask_path + ".gz"   # .nii -> .nii.gz
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(f"mask 文件不存在: {mask_path}")

def _safe_nib_load(path: str):
    import os
    import nibabel as nib

    # 1. 先尝试原路径
    try:
        return nib.load(path)
    except Exception as e:
        print(f"[WARN] 直接加载失败: {path} -> {e}")

    # 2. 如果是 .nii.gz，尝试 .nii
    if path.endswith(".nii.gz"):
        alt = path[:-3]
        if os.path.exists(alt):
            print(f"[FIX] 使用未压缩版本: {alt}")
            return nib.load(alt)

    # 3. 如果是 .nii，尝试 .nii.gz
    if path.endswith(".nii"):
        alt = path + ".gz"
        if os.path.exists(alt):
            print(f"[FIX] 使用压缩版本: {alt}")
            return nib.load(alt)

    raise ValueError(f"无法读取 NIfTI 文件: {path}")

def _load_mask_frames(mask_path: str, max_frames: int = 120, lv_label: int = 1):
    if not mask_path:
        raise ValueError("未提供 mask_path")
    if nib is None:
        raise ImportError("请先安装 nibabel: pip install nibabel")

    mask_path = _resolve_nifti_path(mask_path)

    lower = mask_path.lower()
    if not lower.endswith((".nii", ".nii.gz")):
        raise ValueError("mask_path 目前仅支持 NIfTI (.nii/.nii.gz)")

    #img = nib.load(mask_path)
    img = _safe_nib_load(mask_path)
    
    data = np.asarray(img.get_fdata())
    print(f"[OpticalFlow] mask 原始 shape: {data.shape}")

    #print(f"[OpticalFlow] mask 原始 shape: {data.shape}")

    masks = []

    if data.ndim == 3:
        T = data.shape[-1]
        for i in range(T):
            frame = np.rint(data[:, :, i]).astype(np.int16)
            masks.append(frame == lv_label)

    elif data.ndim == 4:
        squeezed = np.squeeze(data)
        if squeezed.ndim != 3:
            raise ValueError(f"不支持的 mask 4D 维度: {data.shape}")
        T = squeezed.shape[-1]
        for i in range(T):
            frame = np.rint(squeezed[:, :, i]).astype(np.int16)
            masks.append(frame == lv_label)

    else:
        raise ValueError(f"不支持的 mask 维度: {data.shape}")

    if not masks:
        raise ValueError("未读取到有效 mask 帧")

    if len(masks) > max_frames:
        idx = np.linspace(0, len(masks) - 1, max_frames).astype(int)
        masks = [masks[i] for i in idx]

    return masks


# -----------------------------
# 轮廓点初始化与补点
# -----------------------------
def _detect_contour_trackable_points(
    gray: np.ndarray,
    lv_mask: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 4,
    contour_thickness: int = 2
):
    """
    只在 LV 轮廓带上找可追踪点
    返回 shape=(N,1,2) 的 float32 点集
    """
    band_mask = _contour_band_mask(lv_mask, thickness=contour_thickness)

    pts = cv2.goodFeaturesToTrack(
        gray,
        mask=band_mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False
    )

    if pts is None:
        return None

    return np.asarray(pts, dtype=np.float32)


def _supplement_points(
    gray: np.ndarray,
    lv_mask: np.ndarray,
    existing_pts,
    target_points: int = 120,
    min_distance: float = 5.0
):
    """
    当现有点过少时，在当前 LV 轮廓上补充新的可追踪点
    """
    extra = _detect_contour_trackable_points(
        gray=gray,
        lv_mask=lv_mask,
        max_corners=target_points,
        quality_level=0.01,
        min_distance=int(max(3, min_distance)),
        contour_thickness=2
    )

    if extra is None:
        return existing_pts

    if existing_pts is None or len(existing_pts) == 0:
        return extra

    base = existing_pts.reshape(-1, 2)
    add_list = []

    for pt in extra.reshape(-1, 2):
        d = np.sqrt(np.sum((base - pt[None, :]) ** 2, axis=1))
        if d.min() >= min_distance:
            add_list.append(pt)

    if not add_list:
        return existing_pts

    add_arr = np.asarray(add_list, dtype=np.float32).reshape(-1, 1, 2)
    out = np.concatenate([existing_pts, add_arr], axis=0)

    if len(out) > target_points:
        out = out[:target_points]

    return out


# -----------------------------
# 核心：基于 LV 轮廓的光流追踪
# -----------------------------
def _draw_optical_flow_on_lv_contour(frames_gray, lv_masks):
    """
    基于 LV 轮廓的 LK 光流追踪
    返回:
      rendered_frames: 每帧带轮廓和箭头的 BGR 图
    """
    if len(frames_gray) < 2:
        return [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames_gray]

    if len(frames_gray) != len(lv_masks):
        n = min(len(frames_gray), len(lv_masks))
        frames_gray = frames_gray[:n]
        lv_masks = lv_masks[:n]
        print(f"[OpticalFlow] 图像帧数与 mask 帧数不一致，已截断为 {n} 帧")

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    rendered_frames = []

    gray_prev = frames_gray[0]
    mask_prev = lv_masks[0].astype(np.uint8)

    p_prev = _detect_contour_trackable_points(
        gray_prev,
        mask_prev,
        max_corners=180,
        quality_level=0.01,
        min_distance=4,
        contour_thickness=2
    )

    first_canvas = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR)
    _draw_lv_contour(first_canvas, mask_prev)

    if p_prev is not None:
        for pt in p_prev.reshape(-1, 2):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(first_canvas, (x, y), 2, (0, 255, 0), -1)

    rendered_frames.append(first_canvas)

    if p_prev is None or len(p_prev) == 0:
        print("[OpticalFlow] 首帧 LV 轮廓上未检测到可追踪点")
        for i in range(1, len(frames_gray)):
            canvas = cv2.cvtColor(frames_gray[i], cv2.COLOR_GRAY2BGR)
            _draw_lv_contour(canvas, lv_masks[i].astype(np.uint8))
            rendered_frames.append(canvas)
        return rendered_frames

    for i in range(1, len(frames_gray)):
        gray_curr = frames_gray[i]
        mask_curr = lv_masks[i].astype(np.uint8)
        canvas = cv2.cvtColor(gray_curr, cv2.COLOR_GRAY2BGR)
        _draw_lv_contour(canvas, mask_curr)

        if p_prev is None or len(p_prev) == 0:
            p_prev = _detect_contour_trackable_points(gray_curr, mask_curr)
            if p_prev is not None:
                for pt in p_prev.reshape(-1, 2):
                    x, y = int(round(pt[0])), int(round(pt[1]))
                    cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
            rendered_frames.append(canvas)
            gray_prev = gray_curr
            continue

        # 前向 LK
        p_curr, st1, err1 = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, p_prev, None, **lk_params
        )

        valid = (
            p_curr is not None and
            st1 is not None and
            err1 is not None
        )

        if not valid:
            p_prev = _detect_contour_trackable_points(gray_curr, mask_curr)
            rendered_frames.append(canvas)
            gray_prev = gray_curr
            continue

        # 后向 LK：做 forward-backward consistency 过滤
        p_back, st2, err2 = cv2.calcOpticalFlowPyrLK(
            gray_curr, gray_prev, p_curr, None, **lk_params
        )

        valid = (
            p_back is not None and
            st2 is not None and
            err2 is not None
        )

        if not valid:
            p_prev = _detect_contour_trackable_points(gray_curr, mask_curr)
            rendered_frames.append(canvas)
            gray_prev = gray_curr
            continue

        good_prev = p_prev.reshape(-1, 2)
        good_curr = p_curr.reshape(-1, 2)
        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        err1 = err1.reshape(-1)
        fb_err = np.linalg.norm(good_prev - p_back.reshape(-1, 2), axis=1)

        # 当前帧轮廓带，用于保证点仍贴近 LV 轮廓
        band_curr = _contour_band_mask(mask_curr, thickness=3)
        h, w = band_curr.shape[:2]

        keep_idx = []
        for k, (old_pt, new_pt) in enumerate(zip(good_prev, good_curr)):
            if not st1[k] or not st2[k]:
                continue

            x_new, y_new = float(new_pt[0]), float(new_pt[1])
            x_old, y_old = float(old_pt[0]), float(old_pt[1])

            # 边界
            if x_new < 0 or y_new < 0 or x_new >= w or y_new >= h:
                continue

            # 前后向误差
            if fb_err[k] > 1.5:
                continue

            # LK 自带误差
            if err1[k] > 25:
                continue

            # 位移太小/太大都过滤
            disp = np.hypot(x_new - x_old, y_new - y_old)
            if disp < 0.2 or disp > 30:
                continue

            # 必须贴近当前 LV 轮廓
            xi = int(round(x_new))
            yi = int(round(y_new))
            xi = max(0, min(w - 1, xi))
            yi = max(0, min(h - 1, yi))
            if band_curr[yi, xi] == 0:
                continue

            keep_idx.append(k)

        if len(keep_idx) == 0:
            p_prev = _detect_contour_trackable_points(gray_curr, mask_curr)
            if p_prev is not None:
                for pt in p_prev.reshape(-1, 2):
                    x, y = int(round(pt[0])), int(round(pt[1]))
                    cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
            rendered_frames.append(canvas)
            gray_prev = gray_curr
            continue

        keep_idx = np.asarray(keep_idx, dtype=np.int32)
        good_prev = good_prev[keep_idx]
        good_curr = good_curr[keep_idx]

        # 绘制箭头与点
        for old_pt, new_pt in zip(good_prev, good_curr):
            x_old, y_old = old_pt
            x_new, y_new = new_pt
            speed = np.hypot(x_new - x_old, y_new - y_old)

            if speed < 1.5:
                color = (255, 0, 0)    # 蓝：低速
            elif speed < 4.0:
                color = (0, 255, 0)    # 绿：中速
            else:
                color = (0, 0, 255)    # 红：高速

            cv2.arrowedLine(
                canvas,
                (int(round(x_old)), int(round(y_old))),
                (int(round(x_new)), int(round(y_new))),
                color,
                2,
                tipLength=0.28
            )
            cv2.circle(
                canvas,
                (int(round(x_new)), int(round(y_new))),
                2,
                color,
                -1
            )

        # 当前保留下来的点
        p_prev = good_curr.reshape(-1, 1, 2).astype(np.float32)

        # 点太少则补点，但仍只从 LV 轮廓上补
        if len(p_prev) < 40:
            p_prev = _supplement_points(
                gray=gray_curr,
                lv_mask=mask_curr,
                existing_pts=p_prev,
                target_points=120,
                min_distance=5.0
            )

        rendered_frames.append(canvas)
        gray_prev = gray_curr

    return rendered_frames


# -----------------------------
# UI
# -----------------------------
class OpticalFlowDialog(QDialog):
    def __init__(self, file_path, mask_path=None, title="LV 轮廓光流追踪", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.mask_path = mask_path
        self.title = title
        self.frames_bgr = []
        self.current_frame_idx = 0
        self.is_playing = False

        self.setWindowTitle(title)
        self.resize(1000, 800)
        self.setMinimumSize(800, 600)

        self.image_label = QLabel("正在加载数据...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #121212;
                color: #cccccc;
                border: 1px solid #333333;
                font-size: 14px;
            }
        """)
        self.image_label.setMinimumHeight(650)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("font-size: 12px; color: #333333; padding: 4px;")

        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 6px 16px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.slider, stretch=1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label, stretch=1)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(btn_layout)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._on_timer_timeout)

        self.play_btn.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self._load_and_process_data()

    def _load_and_process_data(self):
        try:
            frames_gray = _load_frames(self.file_path)
            print(f"[OpticalFlow] 图像帧数: {len(frames_gray)}")

            if not self.mask_path:
                raise ValueError(
                    "这版是基于 LV 轮廓的追踪，必须传入 mask_path（LV 分割结果 NIfTI）"
                )

            lv_masks = _load_mask_frames(self.mask_path)
            print(f"[OpticalFlow] mask 帧数: {len(lv_masks)}")

            self.frames_bgr = _draw_optical_flow_on_lv_contour(frames_gray, lv_masks)

            total_frames = len(self.frames_bgr)
            self.slider.setMaximum(max(0, total_frames - 1))
            self.info_label.setText(
                f"图像: {os.path.basename(self.file_path)} | "
                f"Mask: {os.path.basename(self.mask_path)} | "
                f"总帧数: {total_frames} | 当前帧: 1"
            )

            self._show_frame(0)

        except Exception as e:
            print(f"[OpticalFlow] 加载失败: {e}")
            QMessageBox.critical(self, "加载失败", f"错误信息:\n{str(e)}", QMessageBox.Ok)
            self.close()

    def _show_frame(self, idx):
        if not self.frames_bgr:
            return

        idx = max(0, min(idx, len(self.frames_bgr) - 1))
        self.current_frame_idx = idx

        pixmap = _to_qpixmap(self.frames_bgr[idx])
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.info_label.setText(
            f"图像: {os.path.basename(self.file_path)} | "
            f"Mask: {os.path.basename(self.mask_path) if self.mask_path else '--'} | "
            f"总帧数: {len(self.frames_bgr)} | 当前帧: {idx + 1}"
        )

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _on_timer_timeout(self):
        next_idx = (self.current_frame_idx + 1) % len(self.frames_bgr)
        self._show_frame(next_idx)

    def _toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("▶ 播放")
            self.is_playing = False
        else:
            self.timer.start()
            self.play_btn.setText("⏸ 暂停")
            self.is_playing = True

    def _on_slider_changed(self, value):
        self._show_frame(value)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_frame(self.current_frame_idx)


def launch_optical_flow(parent, file_path, title="LV轮廓光流追踪", mask_path=None):
    """
    对外接口
    兼容旧调用:
        launch_optical_flow(parent, file_path, title)

    推荐新调用:
        launch_optical_flow(parent, file_path, title, mask_path=pred_mask_path)
    """
    dlg = OpticalFlowDialog(
        file_path=file_path,
        mask_path=mask_path,
        title=title,
        parent=parent
    )
    dlg.exec_()