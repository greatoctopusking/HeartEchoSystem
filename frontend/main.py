"""
PyQt5 前端 — 心脏超声智能分析系统
变更说明：
  · 新增"瓣环定位策略"下拉（自动/极坐标/Wall-LA），随表单提交至后端
  · 分析结果展示中追加 annulus_strategy 字段
  · 历史详情弹窗新增"定位策略"行
  · 其余逻辑与原版完全一致
"""
import sys, json, os, uuid, requests
from typing import Optional, TYPE_CHECKING
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QMainWindow, QTabWidget, QWidget,
    QGroupBox, QFormLayout, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QDockWidget, QSlider,
    QSizePolicy, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# [修改] 补充 QPainter, QBrush, QColor 用于登录界面美化
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QColor
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # type: ignore[no-redef]
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter

BASE_URL = "http://127.0.0.1:5000"

NIFTI_FILTER   = "NIfTI Files (*.nii *.nii.gz)"
DICOM_FILTER   = "DICOM Files (*.dcm *.dicom *.ima)"
VIDEO_FILTER   = "Video Files (*.avi *.mp4 *.mov)"
ALL_IMG_FILTER = "影像文件 (*.nii *.nii.gz *.dcm *.dicom *.ima *.avi *.mp4 *.mov)"

ALGORITHM_OPTIONS = [
    ("biplane_simpson",  "双平面辛普森 (2CH + 4CH)  [金标准]"),
    ("singleplane_2ch",  "单平面辛普森 — 2CH only"),
    ("singleplane_4ch",  "单平面辛普森 — 4CH only"),
    ("area_length_2ch",  "面积-长度法  — 2CH only"),
    ("area_length_4ch",  "面积-长度法  — 4CH only"),
]

# 瓣环定位策略选项：(发送给后端的 key, 界面显示文字)
ANNULUS_STRATEGY_OPTIONS = [
    ("auto",     "自动检测（推荐）"),
    ("polar",    "极坐标法（仅腔体标签）"),
    ("wall_la",  "Wall-LA 接触带法（含 2/3 标签）"),
]

ALGO_NEEDS = {
    "biplane_simpson": (True, True),
    "singleplane_2ch": (True, False),
    "singleplane_4ch": (False, True),
    "area_length_2ch": (True, False),
    "area_length_4ch": (False, True),
}


# [新增] 获取资源文件的绝对路径函数
def get_resource_path(relative_path):
    """获取资源文件的绝对路径（支持相对路径）"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ──────────────────────────────────────────────────────────────────
#  美化后的登录对话框
# ──────────────────────────────────────────────────────────────────
class LoginDialog(QDialog):
    def __init__(self, background_image_path=None, background_opacity=0.4, parent=None):
        super().__init__(parent)
        self.bg_pixmap = QPixmap()
        self.content_opacity = background_opacity
        self._build_ui()
        if background_image_path:
            self.set_background_image(background_image_path)

    def set_background_image(self, image_path):
        if os.path.exists(image_path):
            self.bg_pixmap = QPixmap(image_path)
            self.update() 
            return True
        else:
            print(f"警告: 背景图片不存在: {image_path}")
            return False

    def set_content_opacity(self, opacity):
        """设置半透明蒙版的不透明度 (0.0-1.0)"""
        self.content_opacity = opacity
        self.update()

    def _build_ui(self):
        self.setWindowTitle("系统登录")
        self.setFixedSize(700, 500)
        # 让 paintEvent 在窗口上生效
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        main_layout = QHBoxLayout(self)
        main_layout.addStretch(1) # 将登录框推向右侧
        
        # 登录容器
        login_frame = QFrame()
        login_frame.setFixedWidth(350)
        login_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 230);
                border-radius: 15px;
                border: 1px solid #ccc;
            }
        """)
        
        form_layout = QVBoxLayout(login_frame)
        form_layout.setContentsMargins(30, 40, 30, 40)
        form_layout.setSpacing(15)
        
        title_label = QLabel("系统登录")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; border: none; background: transparent;")
        form_layout.addWidget(title_label)
        form_layout.addSpacing(20)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("用户名")
        self.user_input.setMinimumHeight(40)
        self.user_input.setStyleSheet("font-size: 14px; padding-left: 10px; border-radius: 5px; border: 1px solid #ddd;")
        form_layout.addWidget(self.user_input)
        
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("密码")
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd_input.setMinimumHeight(40)
        self.pwd_input.setStyleSheet("font-size: 14px; padding-left: 10px; border-radius: 5px; border: 1px solid #ddd;")
        form_layout.addWidget(self.pwd_input)
        
        form_layout.addSpacing(10)
        
        self.login_btn = QPushButton("登录")
        self.login_btn.setMinimumHeight(45)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:pressed { background-color: #004578; }
        """)
        self.login_btn.clicked.connect(self.login)
        form_layout.addWidget(self.login_btn)
        
        main_layout.addWidget(login_frame)
        main_layout.addSpacing(50)

    def paintEvent(self, event):
        """绘制背景图片及半透明蒙版"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. 绘制背景图
        if not self.bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.bg_pixmap)
        
        # 2. 绘制白色半透明蒙版以增加对比度
        painter.setBrush(QBrush(QColor(255, 255, 255, int(255 * self.content_opacity))))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

    def login(self):
        u = self.user_input.text().strip()
        p = self.pwd_input.text().strip()
        if not u or not p:
            QMessageBox.warning(self, "提示", "请输入用户名和密码")
            return
        try:
            resp = requests.post(f"{BASE_URL}/login", json={"username": u, "password": p})
            if resp.status_code == 200:
                self.token = resp.json().get("token")
                self.accept()
            else:
                QMessageBox.warning(self, "失败", resp.json().get("error", "登录失败"))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接服务器失败: {e}")


# ──────────────────────────────────────────────────────────────────
#  自适应图片 Label
# ──────────────────────────────────────────────────────────────────
class ScaledImageLabel(QLabel):
    def __init__(self, text="等待分析..."):
        super().__init__(text)
        self._raw_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet(
            "background:#000;color:#888;border:1px solid #333;font-size:16px;")

    def set_image(self, pixmap: QPixmap):
        self._raw_pixmap = pixmap
        self._refresh()

    def _refresh(self):
        if self._raw_pixmap and not self._raw_pixmap.isNull():
            self.setPixmap(self._raw_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ──────────────────────────────────────────────────────────────────
#  嵌入式趋势图
# ──────────────────────────────────────────────────────────────────
class TrendCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 4), facecolor='#1e1e1e')
        self.ax_lvef = fig.add_subplot(131)
        self.ax_edv  = fig.add_subplot(132)
        self.ax_esv  = fig.add_subplot(133)
        fig.tight_layout(pad=2.0)
        super().__init__(fig)
        self.figure = fig
        self._style_axes()

    def _style_axes(self):
        for ax, title in [(self.ax_lvef, 'LVEF (%)'),
                           (self.ax_edv,  'EDV (ml)'),
                           (self.ax_esv,  'ESV (ml)')]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='#cccccc', labelsize=8)
            ax.set_title(title, color='#eeeeee', fontsize=9, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_color('#555555')

    def plot_trend(self, records: list):
        for ax in [self.ax_lvef, self.ax_edv, self.ax_esv]:
            ax.cla()
        self._style_axes()
        if not records:
            self.draw()
            return

        records = sorted(records, key=lambda r: r.get('create_time', ''))
        labels   = [r['create_time'][-5:] for r in records]
        x        = list(range(len(records)))
        lvef_v   = [float(r.get('lvef', 0)) for r in records]
        edv_v    = [float(r.get('edv',  0)) for r in records]
        esv_v    = [float(r.get('esv',  0)) for r in records]

        algo_colors = {
            'biplane_simpson': '#4fc3f7',
            'singleplane_2ch': '#81c784',
            'singleplane_4ch': '#a5d6a7',
            'area_length_2ch': '#ffb74d',
            'area_length_4ch': '#ff8a65',
        }
        for ax, vals, ylabel, nr in [
            (self.ax_lvef, lvef_v, 'LVEF (%)', (50, 100)),
            (self.ax_edv,  edv_v,  'EDV (ml)', (60, 150)),
            (self.ax_esv,  esv_v,  'ESV (ml)', (20,  60)),
        ]:
            colors = [algo_colors.get(r.get('algorithm', ''), '#4fc3f7') for r in records]
            for i in range(len(x) - 1):
                ax.plot([x[i], x[i+1]], [vals[i], vals[i+1]],
                        color='#666666', linewidth=1, zorder=1)
            ax.scatter(x, vals, c=colors, s=50, zorder=2,
                       edgecolors='white', linewidths=0.5)
            ax.axhspan(nr[0], nr[1], alpha=0.08, color='green', zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel(ylabel, color='#cccccc', fontsize=8)
            if vals:
                margin = max(5.0, (max(vals) - min(vals)) * 0.2)
                ax.set_ylim(max(0, min(vals) - margin), max(vals) + margin)

        self.figure.tight_layout(pad=1.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────
#  分析工作线程
# ──────────────────────────────────────────────────────────────────
class AnalysisWorker(QThread):
    progress_updated  = pyqtSignal(int, str)
    analysis_finished = pyqtSignal(dict)
    analysis_error    = pyqtSignal(str)

    def __init__(self, token, file_2ch, file_4ch, patient_data,
                 algorithm, annulus_strategy, base_url):
        super().__init__()
        self.token            = token
        self.file_2ch         = file_2ch
        self.file_4ch         = file_4ch
        self.patient_data     = patient_data
        self.algorithm        = algorithm
        self.annulus_strategy = annulus_strategy
        self.base_url         = base_url
        self._running         = True

    def run(self):
        files: dict = {}
        try:
            headers = {"Authorization": f"Bearer {self.token}",
                       "Accept": "application/ndjson"}
            if self.file_2ch:
                files['file_2ch'] = (os.path.basename(self.file_2ch),
                                     open(self.file_2ch, 'rb'))
            if self.file_4ch:
                files['file_4ch'] = (os.path.basename(self.file_4ch),
                                     open(self.file_4ch, 'rb'))

            form_data = dict(self.patient_data)
            form_data['algorithm']        = self.algorithm
            form_data['annulus_strategy'] = self.annulus_strategy   # ← 新增

            resp = requests.post(f"{self.base_url}/analyze",
                                 files=files, data=form_data,
                                 headers=headers, timeout=600, stream=True)
            if resp.status_code != 200:
                self.analysis_error.emit(f"请求失败: {resp.text}")
                return

            final = None
            for line in resp.iter_lines():
                if not self._running:
                    return
                if not line:
                    continue
                try:
                    data = json.loads(line.decode('utf-8').strip())
                    if "error" in data:
                        self.analysis_error.emit(data["error"])
                        return
                    if "progress" in data and "status" in data:
                        self.progress_updated.emit(data["progress"], data["status"])
                    if "result" in data:
                        final = data["result"]
                except Exception as e:
                    print(f"[WARN] parse line: {e}")

            if final:
                self.analysis_finished.emit(final)
            else:
                self.analysis_error.emit("未获取到有效结果")
        except Exception as e:
            if self._running:
                self.analysis_error.emit(str(e))
        finally:
            for f in files.values():
                try:
                    f[1].close()
                except Exception:
                    pass

    def stop(self):
        # [FIX] 不再调用 self.terminate()，该方法可能导致资源泄漏或崩溃。
        #        仅设置标志位并等待线程自然退出（最多 5 秒）。
        self._running = False
        self.wait(5000)


# ──────────────────────────────────────────────────────────────────
#  独立 3D 查看窗口
# ──────────────────────────────────────────────────────────────────
class LV3DViewerWindow(QMainWindow):
    def __init__(self, mesh_series, parent=None):
        super().__init__(parent)
        self.mesh_series = mesh_series
        self.current_idx = 0
        self.play_direction = 1   # 1: 正向播放, -1: 反向播放
        self.surf = None
        self.plotter = None
        self.auto_play_timer = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("左心室 3D 动态重建")
        self.setMinimumSize(900, 700)

        verts_list = self.mesh_series.get("vertices_series", [])
        faces_raw = self.mesh_series.get("faces", [])
        self.n_frames = int(self.mesh_series.get("n_frames", 0))

        if not verts_list or not faces_raw or self.n_frames <= 0:
            QMessageBox.critical(self, "错误", "3D 数据无效")
            return

        import pyvista as pv
        from pyvistaqt import QtInteractor

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        verts0 = np.array(verts_list[0], dtype=np.float32)
        if verts0.ndim == 1:
            verts0 = verts0.reshape(-1, 3)

        faces_np = np.array(faces_raw, dtype=np.int32)
        if faces_np.ndim == 2 and faces_np.shape[1] == 3:
            faces_np = np.hstack([np.full((len(faces_np), 1), 3, dtype=np.int32), faces_np]).flatten()
        else:
            faces_np = faces_np.flatten()

        self.surf = pv.PolyData(verts0, faces_np)

        self.plotter.add_mesh(
            self.surf,
            color="#ff6b6b",
            smooth_shading=True,
            specular=0.6,
            specular_power=20,
            opacity=0.95
        )

        self.plotter.set_background("black")
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        self.vertices_series = verts_list
        self._init_dock()

        self.auto_play_timer = QTimer(self)
        self.auto_play_timer.setInterval(80)
        self.auto_play_timer.timeout.connect(self._next_frame)

    def _init_dock(self):
        dock = QDockWidget("心动周期控制", self)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(15, 8, 15, 8)
        lay.setSpacing(10)

        self.frame_lbl = QLabel(f"帧: 0 / {self.n_frames - 1}")
        self.frame_lbl.setStyleSheet("color:white;font-weight:bold;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.valueChanged.connect(self._update_frame)

        self.speed_text_lbl = QLabel("速度:")
        self.speed_text_lbl.setStyleSheet("color:white;font-weight:bold;")

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.setFixedWidth(120)
        self.speed_slider.valueChanged.connect(self._update_play_speed)

        self.speed_lbl = QLabel("1.0x")
        self.speed_lbl.setStyleSheet("color:white;min-width:45px;")

        self.play_btn = QPushButton(" ▶ 自动播放")
        self.play_btn.setCheckable(True)
        self.play_btn.setStyleSheet(
            "QPushButton{background:#28a745;color:white;padding:5px 12px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:checked{background:#dc3545;}"
        )
        self.play_btn.toggled.connect(self._toggle_play)

        lay.addWidget(self.frame_lbl)
        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.speed_text_lbl)
        lay.addWidget(self.speed_slider)
        lay.addWidget(self.speed_lbl)
        lay.addWidget(self.play_btn)

        dock.setWidget(w)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        self._update_play_speed(self.speed_slider.value())

    def _update_frame(self, idx):
        if self.surf is None or self.plotter is None:
            return
        self.current_idx = idx
        self.surf.points  = np.array(self.vertices_series[idx], dtype=np.float32)
        self.frame_lbl.setText(f"帧: {idx} / {self.n_frames - 1}")
        self.plotter.update()

    def _update_play_speed(self, value):
        """
        value: 1~10
        映射到不同播放速度（timer interval 越小越快）
        """
        speed_map = {
            1: (200, "0.4x"),
            2: (160, "0.6x"),
            3: (120, "0.8x"),
            4: (100, "0.9x"),
            5: (80,  "1.0x"),
            6: (60,  "1.3x"),
            7: (45,  "1.8x"),
            8: (35,  "2.3x"),
            9: (25,  "3.2x"),
            10: (15, "5.0x"),
        }
        interval, text = speed_map.get(value, (80, "1.0x"))
        self.speed_lbl.setText(text)
        if self.auto_play_timer is not None:
            self.auto_play_timer.setInterval(interval)

    def _next_frame0(self):
        self.slider.setValue((self.current_idx + 1) % self.n_frames)

    def _next_frame(self):
        if self.n_frames <= 1:
            return

        next_idx = self.current_idx + self.play_direction

        if next_idx >= self.n_frames:
            self.play_direction = -1
            next_idx = self.n_frames - 2
        elif next_idx < 0:
            self.play_direction = 1
            next_idx = 1

        self.slider.setValue(next_idx)

    def _toggle_play0(self, checked):
        self.play_btn.setText("⏸暂停" if checked else "▶自动播放")
        if self.auto_play_timer is None:
            return
        if checked:
            self.auto_play_timer.start()
        else:
            self.auto_play_timer.stop()

    def _toggle_play(self, checked):
        self.play_btn.setText(" ⏸ 暂停" if checked else " ▶ 自动播放")
        if self.auto_play_timer is None:
            return
        if checked:
            self._update_play_speed(self.speed_slider.value())
            self.auto_play_timer.start()
        else:
            self.auto_play_timer.stop()

    def closeEvent(self, event):
        if self.auto_play_timer:
            self.auto_play_timer.stop()
        if self.plotter:
            self.plotter.close()
        event.accept()


# ──────────────────────────────────────────────────────────────────
#  算法对比面板
# ──────────────────────────────────────────────────────────────────
class ComparisonPanel(QGroupBox):
    ALGO_NAMES = dict(ALGORITHM_OPTIONS)

    def __init__(self, parent=None):
        super().__init__("算法对比", parent)
        self.setFont(QFont("Arial", 12, QFont.Bold))
        lay = QVBoxLayout()
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["算法", "LVEF (%)", "EDV (ml)", "ESV (ml)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("font-size:12px;")
        lay.addWidget(self._table)
        self.setLayout(lay)
        self.setVisible(False)

    def update_comparison(self, comparison: dict, active_algo: str):
        self._table.setRowCount(0)
        self.setVisible(bool(comparison))
        from PyQt5.QtGui import QColor
        for key, data in comparison.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = [
                QTableWidgetItem(data.get("label", key)),
                QTableWidgetItem(f"{data['EF']:.1f}"),
                QTableWidgetItem(f"{data['EDV']:.1f}"),
                QTableWidgetItem(f"{data['ESV']:.1f}"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if key == active_algo:
                    item.setBackground(QColor(0, 120, 212, 60))
                self._table.setItem(row, col, item)


# ──────────────────────────────────────────────────────────────────
#  历史详情弹窗
# ──────────────────────────────────────────────────────────────────
class HistoryDetailDialog(QDialog):
    def __init__(self, record: dict, token: str, parent=None):
        super().__init__(parent)
        self.record = record
        self.token  = token
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle(f"分析详情 - {self.record.get('name', '')}")
        self.setGeometry(200, 150, 900, 780)
        lay = QVBoxLayout()
        lay.setContentsMargins(20, 20, 20, 20)

        info = QGroupBox("基本信息与临床指标")
        info.setFont(QFont("Arial", 13, QFont.Bold))
        fl = QFormLayout()
        fl.setSpacing(10)
        fl.setContentsMargins(16, 16, 16, 16)

        lvef_v = float(self.record.get('lvef', 0))
        status_txt, status_style = self._lvef_status(lvef_v)
        lvef_lbl   = QLabel(f"{lvef_v:.2f} %")
        lvef_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        status_lbl = QLabel(status_txt)
        status_lbl.setStyleSheet(status_style)
        status_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        lvef_row = QHBoxLayout()
        lvef_row.addWidget(lvef_lbl)
        lvef_row.addSpacing(12)
        lvef_row.addWidget(status_lbl)
        lvef_row.addStretch()

        for name, val in [
            ("患者 UID",    self.record.get('patient_uid', '--')),
            ("姓名",        self.record.get('name', '--')),
            ("年龄",        str(self.record.get('age', '--'))),
            ("性别",        self.record.get('gender', '--')),
            ("分析时间",    self.record.get('create_time', '--')),
            ("算法",        self.record.get('algorithm', '--')),
            ("视图模式",    self.record.get('view_mode', '--')),
            ("瓣环定位策略", self._translate_strategy(self.record.get('annulus_strategy', '--'))),
        ]:
            lbl = QLabel(val)
            lbl.setFont(QFont("Arial", 12))
            fl.addRow(f"{name}:", lbl)
        fl.addRow("LVEF:", lvef_row)
        fl.addRow("EDV:", QLabel(f"{float(self.record.get('edv', 0)):.2f} ml"))
        fl.addRow("ESV:", QLabel(f"{float(self.record.get('esv', 0)):.2f} ml"))
        info.setLayout(fl)

        img_group = QGroupBox("历史分析影像")
        img_group.setFont(QFont("Arial", 13, QFont.Bold))
        img_lay = QVBoxLayout()
        self.overlay_lbl = ScaledImageLabel("加载中...")
        self.overlay_lbl.setMinimumHeight(420)
        img_lay.addWidget(self.overlay_lbl)
        img_group.setLayout(img_lay)

        lay.addWidget(info)
        lay.addWidget(img_group, stretch=1)
        self.setLayout(lay)
        self._load_image()

    @staticmethod
    def _translate_strategy(key: str) -> str:
        """将数据库中存储的 annulus_strategy 值翻译为中文。
        'auto' 表示旧记录（迁移前默认值），显示为"自动检测（旧记录）"。
        """
        return {
            "polar":    "极坐标法",
            "wall_la":  "Wall-LA 接触带法",
            "auto":     "自动检测（旧记录）",
        }.get(key, key or "--")

    @staticmethod
    def _lvef_status(lvef):
        if lvef >= 50:
            return "正常", "color:#28a745;"
        if lvef >= 40:
            return "轻度降低", "color:#ffc107;"
        if lvef >= 30:
            return "中度降低", "color:#fd7e14;"
        return "重度降低", "color:#dc3545;"

    def _load_image0(self):
        rpath = self.record.get('result_path', '')
        if not rpath:
            self.overlay_lbl.setText("无影像记录")
            return
        try:
            resp = requests.get(f"{BASE_URL}/results/{rpath}",
                                headers={"Authorization": f"Bearer {self.token}"},
                                timeout=15)
            if resp.status_code == 200:
                pix = QPixmap()
                pix.loadFromData(resp.content)
                self.overlay_lbl.set_image(pix)
            else:
                self.overlay_lbl.setText("图片加载失败")
        except Exception as e:
            self.overlay_lbl.setText(f"图片加载失败: {e}")

    def _load_image(self):
        rpath = self.record.get('result_path', '')
        if not rpath:
            self.overlay_lbl.setText("无影像记录")
            return

        try:
            # 后端现在是 "file1.png;file2.png"
            paths = [p for p in rpath.split(";") if p.strip()]

            if not paths:
                self.overlay_lbl.setText("无影像记录")
                return

            # 默认加载第一张
            first = paths[0]

            resp = requests.get(
                f"{BASE_URL}/results/{first}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=15
            )

            if resp.status_code == 200:
                pix = QPixmap()
                pix.loadFromData(resp.content)
                self.overlay_lbl.set_image(pix)
            else:
                self.overlay_lbl.setText("图片加载失败")

        except Exception as e:
            self.overlay_lbl.setText(f"图片加载失败: {e}")


# ──────────────────────────────────────────────────────────────────
#  主窗口
# ──────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, token: str):
        super().__init__()
        self.token            = token
        self.file_2ch_path: Optional[str] = None
        self.file_4ch_path: Optional[str] = None
        self.mesh_series: Optional[dict]  = None
        self.viewer_window: Optional[LV3DViewerWindow] = None
        self.worker: Optional[AnalysisWorker]          = None
        self._build_ui()

    # ── 整体布局 ──────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("心脏超声智能分析系统-工作台")
        self.setMinimumSize(1300, 900)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 13))
        self.tab_analyze = QWidget()
        self.tab_history = QWidget()
        self.tabs.addTab(self.tab_analyze, "🔬 分析工作台")
        self.tabs.addTab(self.tab_history, "📋 历史记录与趋势")
        self.setCentralWidget(self.tabs)
        self._build_analyze_tab()
        self._build_history_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ── 分析 Tab ──────────────────────────────────────────────────
    def _build_analyze_tab(self):
        main = QHBoxLayout()
        main.setContentsMargins(15, 15, 15, 15)
        main.setSpacing(18)

        # ─── 左侧控制面板 ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(16)

        # 患者信息
        pat_grp = QGroupBox("患者信息")
        pat_grp.setFont(QFont("Arial", 13, QFont.Bold))
        pat_fl = QFormLayout()
        pat_fl.setSpacing(10)
        pat_fl.setContentsMargins(14, 14, 14, 14)

        uid_row = QHBoxLayout()
        self.uid_input = QLineEdit()
        self.uid_input.setPlaceholderText("留空自动生成")
        self.uid_input.setFixedHeight(34)
        uid_auto_btn = QPushButton("生成")
        uid_auto_btn.setFixedSize(52, 34)
        uid_auto_btn.clicked.connect(
            lambda: self.uid_input.setText(str(uuid.uuid4())[:8].upper()))
        uid_row.addWidget(self.uid_input)
        uid_row.addWidget(uid_auto_btn)

        self.name_input   = QLineEdit(); self.name_input.setFixedHeight(34)
        self.age_input    = QLineEdit(); self.age_input.setFixedHeight(34)
        self.gender_combo = QComboBox(); self.gender_combo.setFixedHeight(34)
        self.gender_combo.addItems(["男", "女", "未知"])

        pat_fl.addRow("患者 UID:", uid_row)
        pat_fl.addRow("姓名:",     self.name_input)
        pat_fl.addRow("年龄:",     self.age_input)
        pat_fl.addRow("性别:",     self.gender_combo)
        pat_grp.setLayout(pat_fl)

        # 算法选择 + 瓣环策略（合并在同一 GroupBox）
        algo_grp = QGroupBox("计算算法与瓣环定位策略")
        algo_grp.setFont(QFont("Arial", 13, QFont.Bold))
        algo_lay = QFormLayout()
        algo_lay.setSpacing(10)
        algo_lay.setContentsMargins(14, 14, 14, 14)

        self.algo_combo = QComboBox()
        self.algo_combo.setFixedHeight(34)
        for key, label in ALGORITHM_OPTIONS:
            self.algo_combo.addItem(label, userData=key)
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)

        self.strategy_combo = QComboBox()
        self.strategy_combo.setFixedHeight(34)
        for key, label in ANNULUS_STRATEGY_OPTIONS:
            self.strategy_combo.addItem(label, userData=key)
        # 默认选"自动检测"
        self.strategy_combo.setCurrentIndex(0)

        strategy_hint = QLabel(
            "自动：推理后检测 mask 标签决定策略\n"
            "极坐标：仅腔体(label 1)分割结果\n"
            "Wall-LA：含壁(2)/左房(3)多类别结果")
        strategy_hint.setStyleSheet("color:#888;font-size:11px;")
        strategy_hint.setWordWrap(True)

        algo_lay.addRow("计算算法:", self.algo_combo)
        algo_lay.addRow("瓣环策略:", self.strategy_combo)
        algo_lay.addRow("",          strategy_hint)
        algo_grp.setLayout(algo_lay)

        # 文件上传
        file_grp = QGroupBox("影像文件 (NIfTI / DICOM / Video)")
        file_grp.setFont(QFont("Arial", 13, QFont.Bold))
        file_lay = QVBoxLayout()
        file_lay.setContentsMargins(14, 14, 14, 14)
        file_lay.setSpacing(10)

        self.file2ch_btn   = QPushButton("选择 2CH 文件")
        self.file2ch_label = QLabel("未选择")
        self._style_file_label(self.file2ch_label)
        self.file4ch_btn   = QPushButton("选择 4CH 文件")
        self.file4ch_label = QLabel("未选择")
        self._style_file_label(self.file4ch_label)

        file_lay.addWidget(self.file2ch_btn)
        file_lay.addWidget(self.file2ch_label)
        file_lay.addSpacing(6)
        file_lay.addWidget(self.file4ch_btn)
        file_lay.addWidget(self.file4ch_label)
        file_grp.setLayout(file_lay)

        # 进度
        prog_grp = QGroupBox("分析进度")
        prog_grp.setFont(QFont("Arial", 13, QFont.Bold))
        prog_lay = QVBoxLayout()
        prog_lay.setContentsMargins(14, 14, 14, 14)
        self.status_lbl = QLabel("等待开始...")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.prog_bar = QProgressBar()
        self.prog_bar.setFixedHeight(24)
        prog_lay.addWidget(self.status_lbl)
        prog_lay.addWidget(self.prog_bar)
        prog_grp.setLayout(prog_lay)

        self.upload_btn = QPushButton("🚀开始分析")
        self.upload_btn.setMinimumHeight(54)
        self.upload_btn.setStyleSheet(
            "background:#28a745;color:white;font-size:17px;"
            "font-weight:bold;border-radius:6px;")

        left_lay.addWidget(pat_grp)
        left_lay.addWidget(algo_grp)
        left_lay.addWidget(file_grp)
        left_lay.addWidget(prog_grp)
        left_lay.addWidget(self.upload_btn)
        left_lay.addStretch()

        # ─── 右侧结果区域 ─────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(12)

        img_grp = QGroupBox("分析结果影像")
        img_grp.setFont(QFont("Arial", 13, QFont.Bold))
        img_inner = QVBoxLayout()
        img_inner.setContentsMargins(6, 10, 6, 6)

        self.overlay_tabs = QTabWidget()
        self.overlay_tabs.setFont(QFont("Arial", 12))
        self.overlay_tabs.setStyleSheet("""
            QTabBar::tab { min-width: 100px; padding: 6px 18px; font-size: 13px; }
            QTabBar::tab:selected { font-weight: bold; }
        """)

        self.overlay_2ch = ScaledImageLabel("等待 2CH 分析...")
        self.overlay_4ch = ScaledImageLabel("等待 4CH 分析...")

        self.overlay_tabs.addTab(self.overlay_2ch, "🫀2CH（两腔心）")
        self.overlay_tabs.addTab(self.overlay_4ch, "🫀4CH（四腔心）")

        img_inner.addWidget(self.overlay_tabs)
        img_grp.setLayout(img_inner)

        data_grp = QGroupBox("临床分析指标")
        data_grp.setFont(QFont("Arial", 13, QFont.Bold))
        data_lay = QHBoxLayout()
        data_lay.setContentsMargins(14, 14, 14, 14)

        metrics_lay = QFormLayout()
        metrics_lay.setSpacing(8)
        self.lvef_label         = QLabel("-- %");       self.lvef_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.lvef_status        = QLabel("--");          self.lvef_status.setFont(QFont("Arial", 13, QFont.Bold))
        self.edv_label          = QLabel("-- ml");      self.edv_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.esv_label          = QLabel("-- ml");      self.esv_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.algo_result_lbl    = QLabel("--");          self.algo_result_lbl.setStyleSheet("color:#0078D4;font-size:12px;")
        self.strategy_result_lbl = QLabel("--");         self.strategy_result_lbl.setStyleSheet("color:#888;font-size:11px;")  # ← 新增

        metrics_lay.addRow("LVEF:",       self.lvef_label)
        metrics_lay.addRow("状态:",        self.lvef_status)
        metrics_lay.addRow("EDV:",        self.edv_label)
        metrics_lay.addRow("ESV:",        self.esv_label)
        metrics_lay.addRow("算法:",        self.algo_result_lbl)
        metrics_lay.addRow("定位策略:",    self.strategy_result_lbl)    # ← 新增

        btn_col = QVBoxLayout()
        self.view3d_btn = QPushButton("🏥查看3D模型")
        self.view3d_btn.setEnabled(False)
        self.view3d_btn.setFixedHeight(50)
        self.view3d_btn.setStyleSheet(
            "background:#ffc107;font-weight:bold;font-size:14px;border-radius:5px;")
        btn_col.addWidget(self.view3d_btn)
        btn_col.addStretch()

        data_lay.addLayout(metrics_lay, stretch=2)
        data_lay.addStretch()
        data_lay.addLayout(btn_col)
        data_grp.setLayout(data_lay)

        self.comparison_panel = ComparisonPanel()

        right_lay.addWidget(img_grp, stretch=4)
        right_lay.addWidget(data_grp, stretch=1)
        right_lay.addWidget(self.comparison_panel, stretch=2)

        main.addWidget(left)
        main.addWidget(right, stretch=1)
        self.tab_analyze.setLayout(main)

        # ── 绑定事件 ──
        self.file2ch_btn.clicked.connect(lambda: self._pick_file("2ch"))
        self.file4ch_btn.clicked.connect(lambda: self._pick_file("4ch"))
        self.upload_btn.clicked.connect(self._start_analysis)
        self.view3d_btn.clicked.connect(self._open_3d_viewer)
        self._on_algo_changed(0)

    # ── 历史 Tab ──────────────────────────────────────────────────
    def _build_history_tab(self):
        outer = QHBoxLayout()
        outer.setContentsMargins(15, 15, 15, 15)
        outer.setSpacing(15)

        left = QWidget()
        left.setFixedWidth(300)
        left_lay = QVBoxLayout(left)

        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setFixedHeight(38)
        refresh_btn.setStyleSheet("background:#0078D4;color:white;font-weight:bold;")
        refresh_btn.clicked.connect(self._load_patients)
        left_lay.addWidget(refresh_btn)

        self.patient_table = QTableWidget()
        self.patient_table.setColumnCount(3)
        self.patient_table.setHorizontalHeaderLabels(["UID", "姓名", "记录数"])
        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.patient_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.patient_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.patient_table.cellClicked.connect(self._on_patient_clicked)
        left_lay.addWidget(self.patient_table, stretch=1)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setSpacing(12)

        self.patient_info_lbl = QLabel("← 从左侧选择患者查看趋势")
        self.patient_info_lbl.setStyleSheet(
            "background:#f0f4ff;padding:8px;border-radius:4px;font-size:13px;")
        right_lay.addWidget(self.patient_info_lbl)

        trend_grp = QGroupBox("历史趋势图")
        trend_grp.setFont(QFont("Arial", 12, QFont.Bold))
        trend_inner = QVBoxLayout()
        trend_inner.setContentsMargins(8, 8, 8, 8)
        self.trend_canvas = TrendCanvas()
        self.trend_canvas.setMinimumHeight(200)
        trend_inner.addWidget(self.trend_canvas)
        trend_grp.setLayout(trend_inner)
        right_lay.addWidget(trend_grp, stretch=2)

        rec_grp = QGroupBox("所有记录")
        rec_grp.setFont(QFont("Arial", 12, QFont.Bold))
        rec_inner = QVBoxLayout()
        self.record_table = QTableWidget()
        self.record_table.setColumnCount(9)                             # ← +1 列
        self.record_table.setHorizontalHeaderLabels([
            "时间", "LVEF(%)", "EDV(ml)", "ESV(ml)",
            "算法", "视图", "定位策略", "状态", "操作"])               # ← 新增"定位策略"
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.record_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.record_table.setAlternatingRowColors(True)
        self.record_table.cellDoubleClicked.connect(
            lambda r, _: self._show_record_detail(r))
        rec_inner.addWidget(self.record_table)
        rec_grp.setLayout(rec_inner)
        right_lay.addWidget(rec_grp, stretch=3)

        outer.addWidget(left)
        outer.addWidget(right, stretch=1)
        self.tab_history.setLayout(outer)
        self._current_patient_records = []

    # ──────────────────────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _style_file_label(lbl: QLabel):
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "background:#f8f9fa;border:1px solid #ddd;padding:5px;color:#666;")

    def _on_algo_changed(self, _idx):
        algo = self.algo_combo.currentData()
        needs2, needs4 = ALGO_NEEDS.get(algo, (True, True))
        self.file2ch_btn.setEnabled(needs2)
        self.file4ch_btn.setEnabled(needs4)
        self.file2ch_btn.setStyleSheet("" if needs2 else "background:#cccccc;color:#888;")
        self.file4ch_btn.setStyleSheet("" if needs4 else "background:#cccccc;color:#888;")
        if not needs2:
            self.file_2ch_path = None
            self.file2ch_label.setText("不需要")
        if not needs4:
            self.file_4ch_path = None
            self.file4ch_label.setText("不需要")

    def _pick_file(self, view: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择影像文件",
            "",
            f"{ALL_IMG_FILTER};;{NIFTI_FILTER};;{DICOM_FILTER};;{VIDEO_FILTER}"
        )
        if not path:
            return

        fname = os.path.basename(path)
        style_ok = (
            "color:green;background:#e6fffa;"
            "border:1px solid #c3e6cb;padding:5px;font-weight:bold;"
        )

        if view == "2ch":
            self.file_2ch_path = path
            self.file2ch_label.setText(f"✓ {fname}")
            self.file2ch_label.setStyleSheet(style_ok)
        else:
            self.file_4ch_path = path
            self.file4ch_label.setText(f"✓ {fname}")
            self.file4ch_label.setStyleSheet(style_ok)

    def _start_analysis(self):
        algo = self.algo_combo.currentData()
        needs2, needs4 = ALGO_NEEDS.get(algo, (True, True))

        if needs2 and not self.file_2ch_path:
            QMessageBox.warning(self, "提示", "当前算法需要 2CH 文件，请先选择")
            return
        if needs4 and not self.file_4ch_path:
            QMessageBox.warning(self, "提示", "当前算法需要 4CH 文件，请先选择")
            return
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "提示", "请输入患者姓名")
            return

        self.upload_btn.setEnabled(False)
        self.prog_bar.setValue(0)
        self.overlay_2ch.setText("正在分析中，请稍候...")
        self.overlay_2ch._raw_pixmap = None
        self.overlay_4ch.setText("正在分析中，请稍候...")
        self.overlay_4ch._raw_pixmap = None
        self.comparison_panel.setVisible(False)

        patient_data = {
            'patient_uid': self.uid_input.text().strip(),
            'name':   self.name_input.text().strip(),
            'age':    self.age_input.text().strip(),
            'gender': self.gender_combo.currentText(),
        }

        # ← 读取用户选择的瓣环策略
        annulus_strategy = self.strategy_combo.currentData()

        self.worker = AnalysisWorker(
            self.token,
            self.file_2ch_path if needs2 else None,
            self.file_4ch_path if needs4 else None,
            patient_data, algo, annulus_strategy, BASE_URL
        )
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.analysis_finished.connect(self._on_analysis_finished)
        self.worker.analysis_error.connect(self._on_analysis_error)
        self.worker.start()

    def _update_progress(self, val, txt):
        self.prog_bar.setValue(val)
        self.status_lbl.setText(txt)

    def _on_analysis_finished(self, result):
        self.upload_btn.setEnabled(True)
        self.status_lbl.setText("分析完成！")

        lvef = float(result.get('LVEF', 0))
        self.lvef_label.setText(f"{lvef:.1f} %")
        self.edv_label.setText(f"{result.get('EDV', 0):.1f} ml")
        self.esv_label.setText(f"{result.get('ESV', 0):.1f} ml")

        algo_key   = result.get('algorithm', '')
        algo_label = dict(ALGORITHM_OPTIONS).get(algo_key) or algo_key
        self.algo_result_lbl.setText(algo_label)

        # ← 显示实际使用的瓣环策略
        strat = result.get('annulus_strategy', '--')
        self.strategy_result_lbl.setText(HistoryDetailDialog._translate_strategy(strat))

        # LVEF 状态颜色
        if lvef >= 50:
            self.lvef_status.setText("正常")
            self.lvef_status.setStyleSheet("color:#28a745;font-weight:bold;")
        elif lvef >= 40:
            self.lvef_status.setText("轻度降低")
            self.lvef_status.setStyleSheet("color:#ffc107;font-weight:bold;")
        elif lvef >= 30:
            self.lvef_status.setText("中度降低")
            self.lvef_status.setStyleSheet("color:#fd7e14;font-weight:bold;")
        else:
            self.lvef_status.setText("重度降低")
            self.lvef_status.setStyleSheet("color:#dc3545;font-weight:bold;")

        comparison = result.get('comparison', {})
        if comparison:
            self.comparison_panel.update_comparison(comparison, algo_key)

        def _load_overlay(url, label: ScaledImageLabel):
            try:
                resp = requests.get(url, headers={"Authorization": f"Bearer {self.token}"}, timeout=20)
                if resp.status_code == 200:
                    pix = QPixmap()
                    pix.loadFromData(resp.content)
                    label.set_image(pix)
            except Exception as e:
                print(f"[WARN] overlay load: {e}")

        url_2ch = result.get('overlay_2ch_url')
        url_4ch = result.get('overlay_4ch_url')
        if url_2ch:
            _load_overlay(url_2ch, self.overlay_2ch)
            self.overlay_tabs.setCurrentIndex(0)   # 分析完自动跳到 2CH
        if url_4ch:
            _load_overlay(url_4ch, self.overlay_4ch)
            if not url_2ch:                        # 纯 4CH 模式直接显示 4CH
                self.overlay_tabs.setCurrentIndex(1)

        if result.get('mesh_3d_series'):
            self.mesh_series = result['mesh_3d_series']
            self.view3d_btn.setEnabled(True)

    def _on_analysis_error(self, msg):
        self.upload_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)

    def _open_3d_viewer(self):
        if not self.mesh_series:
            return
        try:
            self.viewer_window = LV3DViewerWindow(self.mesh_series, self)
            self.viewer_window.show()
        except ImportError:
            QMessageBox.critical(self, "错误",
                "请先安装 3D 组件:\npip install pyvista pyvistaqt")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"3D 渲染失败: {e}")

    # ── 历史 Tab 相关 ─────────────────────────────────────────────
    def _on_tab_changed(self, idx):
        if idx == 1:
            self._load_patients()

    def _load_patients(self):
        try:
            resp = requests.get(f"{BASE_URL}/patients",
                                headers={"Authorization": f"Bearer {self.token}"},
                                timeout=10)
            if resp.status_code != 200:
                QMessageBox.critical(self, "错误", resp.text)
                return
            patients = resp.json()
            self.patient_table.setRowCount(0)
            for i, p in enumerate(patients):
                self.patient_table.insertRow(i)
                self.patient_table.setItem(i, 0, QTableWidgetItem(p.get('patient_uid', '')))
                self.patient_table.setItem(i, 1, QTableWidgetItem(p.get('name', '')))
                cnt = QTableWidgetItem(str(p.get('record_count', 0)))
                cnt.setTextAlignment(Qt.AlignCenter)
                self.patient_table.setItem(i, 2, cnt)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载患者列表失败: {e}")

    def _on_patient_clicked(self, row, _col):
        uid_item = self.patient_table.item(row, 0)
        if uid_item:
            self._load_patient_trend(uid_item.text())

    def _load_patient_trend(self, patient_uid: str):
        try:
            resp = requests.get(
                f"{BASE_URL}/history/trend/{patient_uid}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=10)
            if resp.status_code != 200:
                QMessageBox.critical(self, "错误", resp.text)
                return
            data    = resp.json()
            info    = data.get('patient', {})
            records = data.get('records', [])
            self._current_patient_records = records

            self.patient_info_lbl.setText(
                f"  患者 UID: {info.get('patient_uid', '--')}  |  "
                f"姓名: {info.get('name', '--')}  |  "
                f"年龄: {info.get('age', '--')}  |  "
                f"性别: {info.get('gender', '--')}  |  "
                f"共 {len(records)} 条记录"
            )
            self.trend_canvas.plot_trend(records)

            self.record_table.setRowCount(0)
            for i, r in enumerate(records):
                self.record_table.insertRow(i)
                lvef_v  = float(r.get('lvef', 0))
                status, _ = HistoryDetailDialog._lvef_status(lvef_v)
                strategy_display = HistoryDetailDialog._translate_strategy(
                    r.get('annulus_strategy', ''))
                row_data = [
                    r.get('create_time', ''),
                    f"{lvef_v:.1f}",
                    f"{float(r.get('edv', 0)):.1f}",
                    f"{float(r.get('esv', 0)):.1f}",
                    r.get('algorithm', '--'),
                    r.get('view_mode', '--'),
                    strategy_display,
                    status,
                ]
                for col, txt in enumerate(row_data):
                    item = QTableWidgetItem(txt)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.record_table.setItem(i, col, item)

                detail_btn = QPushButton("详情")
                detail_btn.clicked.connect(
                    lambda checked, row_=i: self._show_record_detail(row_))
                self.record_table.setCellWidget(i, 8, detail_btn)  # ← col 8 (原 7)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载趋势失败: {e}")

    def _show_record_detail(self, row: int):
        if row < 0 or row >= len(self._current_patient_records):
            return
        rec      = self._current_patient_records[row]
        uid_item = self.patient_table.item(self.patient_table.currentRow(), 0)
        nm_item  = self.patient_table.item(self.patient_table.currentRow(), 1)
        rec_full = dict(rec)
        rec_full.setdefault('patient_uid', uid_item.text() if uid_item else '')
        rec_full.setdefault('name',        nm_item.text()  if nm_item  else '')
        rec_full.setdefault('age',    '--')
        rec_full.setdefault('gender', '--')
        dlg = HistoryDetailDialog(rec_full, self.token, self)
        dlg.exec_()

    def closeEvent(self, event):
        # [FIX] 关闭主窗口时同时关闭 3D 查看窗口和工作线程
        if self.viewer_window:
            self.viewer_window.close()
        if self.worker:
            self.worker.stop()
        event.accept()



# ──────────────────────────────────────────────────────────────────
#  程序入口
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # [修改] 调用美化后的 LoginDialog，传入背景图路径
    login = LoginDialog(
        background_image_path=get_resource_path("ui_res/login_back.png"), 
        background_opacity=0.4
    )
    
    if login.exec_() == QDialog.Accepted:
        main_win = MainWindow(login.token)
        main_win.show()
        sys.exit(app.exec_())