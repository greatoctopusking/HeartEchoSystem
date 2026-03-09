import sys
import requests
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
BASE_URL = "http://127.0.0.1:5000"

class HeartApp(QWidget):
    def __init__(self):
        super().__init__()
        self.file_2ch_path = None
        self.file_4ch_path = None
        self.token = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("心脏超声智能分析系统（双平面 Simpson）")
        self.setGeometry(200, 100, 1000, 800)  # ✅ 窗口变大

        layout = QVBoxLayout()

        # ================= 登录区 =================
        login_group = QGroupBox("登录")
        login_layout = QHBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("用户名")

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("密码")
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.do_login)

        login_layout.addWidget(self.username_input)
        login_layout.addWidget(self.password_input)
        login_layout.addWidget(self.login_btn)
        login_group.setLayout(login_layout)

        # ================= 患者信息 =================
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("患者姓名")

        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("年龄")

        self.gender_input = QLineEdit()
        self.gender_input.setPlaceholderText("性别")

        # ================= 文件选择 =================
        self.file2ch_btn = QPushButton("选择 2CH NIfTI 文件")
        self.file2ch_btn.clicked.connect(self.choose_2ch)

        self.file4ch_btn = QPushButton("选择 4CH NIfTI 文件")
        self.file4ch_btn.clicked.connect(self.choose_4ch)

        # ================= 上传按钮 =================
        self.upload_btn = QPushButton("开始分析")
        self.upload_btn.clicked.connect(self.upload_file)

        # 在 upload_btn 下方添加
        self.view3d_btn = QPushButton("查看 3D 旋转模型 (非对称重建)")
        self.view3d_btn.setEnabled(False)  # 默认禁用，分析完后再开启
        self.view3d_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.view3d_btn.clicked.connect(self.open_3d_viewer)

        # ================= 结果显示 =================
        self.result_label = QLabel("分析结果显示在这里")
        self.result_label.setAlignment(Qt.AlignCenter)

        # ✅ 图片区域变大
        self.overlay_label = QLabel("Overlay 图片显示在这里")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setMinimumHeight(500)

        layout.addWidget(login_group)
        layout.addWidget(self.name_input)
        layout.addWidget(self.age_input)
        layout.addWidget(self.gender_input)
        layout.addWidget(self.file2ch_btn)
        layout.addWidget(self.file4ch_btn)
        layout.addWidget(self.upload_btn)

        # ✅ 只调整位置：放到“开始分析”下面
        layout.addWidget(self.view3d_btn)


        # ================= 3D 序列控制条 =================
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)

        self.frame_label = QLabel("3D Frame: - / -")
        self.frame_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_slider)


        layout.addWidget(self.result_label)
        layout.addWidget(self.overlay_label)

        self.setLayout(layout)

    def do_login(self):
        try:
            r = requests.post(
                f"{BASE_URL}/login",
                json={
                    "username": self.username_input.text().strip(),
                    "password": self.password_input.text().strip()
                },
                timeout=20
            )
            if r.status_code == 200:
                self.token = r.json().get("token")
                QMessageBox.information(self, "成功", "登录成功")
            else:
                QMessageBox.critical(self, "登录失败", r.text)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def choose_2ch(self):
        self.file_2ch_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 2CH NIfTI 文件",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )

    def choose_4ch(self):
        self.file_4ch_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 4CH NIfTI 文件",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )

    def upload_file(self):
        if not self.token:
            QMessageBox.warning(self, "提示", "请先登录")
            return

        if not self.file_2ch_path or not self.file_4ch_path:
            QMessageBox.warning(self, "错误", "请先选择 2CH 和 4CH 文件")
            return

        url = f"{BASE_URL}/analyze"

        try:
            with open(self.file_2ch_path, 'rb') as f2ch, open(self.file_4ch_path, 'rb') as f4ch:
                files = {
                    'file_2ch': (self.file_2ch_path.split("/")[-1], f2ch),
                    'file_4ch': (self.file_4ch_path.split("/")[-1], f4ch)
                }

                data = {
                    'name': self.name_input.text().strip(),
                    'age': self.age_input.text().strip(),
                    'gender': self.gender_input.text().strip()
                }

                headers = {"Authorization": f"Bearer {self.token}"}

                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=600
                )

            if response.status_code == 200:
                result = response.json()

                self.result_label.setText(
                    f"LVEF: {result['LVEF']}%\n"
                    f"EDV: {result['EDV']} ml\n"
                    f"ESV: {result['ESV']} ml"
                )

                """
                # 【添加位置 2】：处理返回的 3D 数据
                if "mesh_3d" in result and result["mesh_3d"] is not None:
                    self.mesh_data = result["mesh_3d"] # 把数据存起来
                    self.view3d_btn.setEnabled(True)   # 激活 3D 按钮
                    self.view3d_btn.setText("查看 3D 旋转模型 (数据就绪)")
                else:
                    self.view3d_btn.setEnabled(False)
                    self.view3d_btn.setText("3D 数据生成失败")
                """
                # 处理返回的 3D 序列数据
                if "mesh_3d_series" in result and result["mesh_3d_series"] is not None:
                    self.mesh_series = result["mesh_3d_series"]
                    n_frames = int(self.mesh_series.get("n_frames", 0))

                    if n_frames > 0:
                        self.view3d_btn.setEnabled(True)
                        self.view3d_btn.setText("查看 3D 旋转模型 (序列数据就绪)")

                        self.frame_slider.setEnabled(True)
                        self.frame_slider.setMinimum(0)
                        self.frame_slider.setMaximum(n_frames - 1)
                        self.frame_slider.setValue(0)
                        self.frame_label.setText(f"3D Frame: 0 / {n_frames-1}")
                    else:
                        self.view3d_btn.setEnabled(False)
                        self.view3d_btn.setText("3D 序列为空")
                        self.frame_slider.setEnabled(False)
                        self.frame_label.setText("3D Frame: - / -")
                else:
                    self.view3d_btn.setEnabled(False)
                    self.view3d_btn.setText("3D 数据生成失败")
                    self.frame_slider.setEnabled(False)
                    self.frame_label.setText("3D Frame: - / -")
                
                
                overlay_url = result.get("overlay_url")
                if overlay_url:
                    img = requests.get(overlay_url, timeout=60).content
                    with open("tmp_overlay.png", "wb") as fp:
                        fp.write(img)

                    pix = QPixmap("tmp_overlay.png")
                    self.overlay_label.setPixmap(
                        pix.scaled(
                            self.overlay_label.width(),
                            self.overlay_label.height(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    )

            else:
                QMessageBox.critical(self, "错误", response.text)



        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

# --- main.py ---
# 在 HeartApp 类的末尾（if __name__ == "__main__": 之前）添加：
    def open_3d_viewer(self):
        if not hasattr(self, 'mesh_series') or self.mesh_series is None:
            QMessageBox.warning(self, "提示", "暂无 3D 模型数据")
            return

        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
            import numpy as np

            # 1. 取第0帧数据（全局坐标）
            verts0 = np.array(self.mesh_series["vertices_series"][0], dtype=float)
            faces = np.array(self.mesh_series["faces"], dtype=int)
            self.surf = pv.PolyData(verts0, faces)

            # 2. 创建Plotter，固定全局参考系
            self.plotter = BackgroundPlotter(title="左心室收缩舒张 3D 视图（原始坐标参考系）")
            """
            # ========== 关键：固定参考系，关闭自动旋转/缩放 ==========
            self.plotter.enable_trackball_style()  # 手动操作模式
            self.plotter.camera_position = 'iso'   # 固定初始视角
            self.plotter.camera.focal_point = np.mean(verts0, axis=0)  # 聚焦心室中心
            self.plotter.camera.view_up = [0, 1, 0]  # 固定Y轴向上（原始数据坐标）
            """
            self.plotter.enable_trackball_style()
            self.plotter.camera.focal_point = np.mean(verts0, axis=0)
            self.plotter.view_isometric()
            self.plotter.camera.up = (0, 1, 0)
            self.plotter.reset_camera()
            
            # 3. 添加心室网格（全局坐标）
            self.plotter.add_mesh(
                self.surf, 
                color="salmon", 
                smooth_shading=True,
                specular=0.5, 
                opacity=0.9, 
                show_edges=False,
                name="ventricle"  # 命名网格，便于后续更新
            )

            # 4. 添加全局坐标系参考（可选，便于验证）
            self.plotter.add_axes()  # 显示XYZ轴（原始数据坐标）
            #self.plotter.add_grid()  # 显示网格参考系

            # 5. 固定背景和视角
            self.plotter.set_background("black")
            self.plotter.reset_camera()
            self.plotter.show()

        except ImportError:
            QMessageBox.critical(self, "错误", "请先安装 3D 渲染组件：\\npip install pyvista pyvistaqt")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"3D 渲染失败: {str(e)}")

    # 修改帧更新逻辑：仅更新顶点坐标，不改变参考系
    def on_frame_changed(self, idx: int):
        if not hasattr(self, "mesh_series"):
            return
        n_frames = int(self.mesh_series.get("n_frames", 0))
        self.frame_label.setText(f"3D Frame: {idx} / {max(0, n_frames-1)}")

        if not hasattr(self, "surf") or not hasattr(self, "plotter"):
            return

        try:
            # 获取当前帧的全局坐标顶点
            verts = np.array(self.mesh_series["vertices_series"][int(idx)], dtype=float)
            # 仅更新顶点坐标，保留面片和参考系不变
            self.surf.points = verts
            # 强制刷新渲染（不改变相机视角）
            #self.plotter.update_scalar_bar_range()
            self.plotter.render()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新 3D 帧失败: {str(e)}")


    def open_3d_viewer1(self):
        """弹出 3D 交互窗口"""
        if not hasattr(self, 'mesh_series') or self.mesh_series is None:
            QMessageBox.warning(self, "提示", "暂无 3D 模型数据")
            return

        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
            import numpy as np

            # 1. 准备数据：取第 0 帧 vertices + 固定 faces
            verts0 = np.array(self.mesh_series["vertices_series"][0], dtype=float)
            faces = np.array(self.mesh_series["faces"], dtype=int)
            # 2. 创建 PyVista 表面网格（保存到 self.surf，便于滑条更新）
            self.surf = pv.PolyData(verts0, faces)

            # 3. 创建独立窗口 (BackgroundPlotter 不会阻塞 PyQt 主线程)
            # 我们把 plotter 挂在 self 上防止它被垃圾回收导致窗口闪退
            self.plotter = BackgroundPlotter(title="左心室非对称 3D 重建 (Simpson Biplane)")

            # 4. 添加网格并修饰外观
            # 使用 salmon 颜色模拟心肌，加点光泽感 (specular)
            self.plotter.add_mesh(self.surf, color="salmon", smooth_shading=True,
                                 specular=0.5, opacity=0.9, show_edges=False)

            # 绘制一个中心轴线参考（黄色虚线感）
            max_z = verts0[:, 2].max() if len(verts0) > 0 else 100
            self.plotter.add_lines(np.array([[0,0,0], [0,0,max_z]]),
                                  color="yellow", width=2, label="Central Axis")

            # 设置默认视角和背景
            self.plotter.set_background("black") # 黑色背景更有科技感
            self.plotter.view_isometric()
            self.plotter.add_legend()
            self.plotter.reset_camera()
            self.plotter.show()

        except ImportError:
            QMessageBox.critical(self, "错误", "请先安装 3D 渲染组件：\\npip install pyvista pyvistaqt")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"3D 渲染失败: {str(e)}")

  
    def on_frame_changed1(self, idx: int):
        if not hasattr(self, "mesh_series"):
            return
        n_frames = int(self.mesh_series.get("n_frames", 0))
        self.frame_label.setText(f"3D Frame: {idx} / {max(0, n_frames-1)}")

        # 如果 3D 窗口没开，就只更新 label
        if not hasattr(self, "surf"):
            return

        try:
            verts = np.array(self.mesh_series["vertices_series"][int(idx)], dtype=float)
            # 更新点坐标（faces 不变）
            self.surf.points = verts
            # 触发渲染
            if hasattr(self, "plotter"):
                self.plotter.render()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新 3D 帧失败: {str(e)}")



# 这里是类的结尾

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartApp()
    window.show()
    sys.exit(app.exec_())