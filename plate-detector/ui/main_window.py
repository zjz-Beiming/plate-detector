import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QInputDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from ui.image_panel import ImagePanel
from ui.plate_list import PlateListPanel
from ui.control_bar import ControlBar
from ui.settings_dialog import SettingsDialog
from core.detector import DetectionEngine
from core.detection_thread import DetectionThread
from utils.image_utils import cv2_to_qpixmap
from utils.save_utils import save_annotated_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车牌检测器 - YOLOv8")
        self.setMinimumSize(1000, 650)

        self._engine = DetectionEngine()
        self._thread = None

        self._current_annotated = None
        self._source_name = "image"

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        toolbar = QHBoxLayout()
        self._btn_image = QPushButton("📁 打开图片")
        self._btn_image.clicked.connect(self._open_image)
        toolbar.addWidget(self._btn_image)

        self._btn_video = QPushButton("🎬 打开视频")
        self._btn_video.clicked.connect(self._open_video)
        toolbar.addWidget(self._btn_video)

        self._btn_camera = QPushButton("📷 打开摄像头")
        self._btn_camera.clicked.connect(self._open_camera)
        toolbar.addWidget(self._btn_camera)

        self._btn_stream = QPushButton("🌐 网络流")
        self._btn_stream.clicked.connect(self._open_stream)
        toolbar.addWidget(self._btn_stream)

        toolbar.addStretch()

        self._model_label = QLabel("模型: 未加载")
        self._model_label.setStyleSheet("color: #6b7280; font-size: 10px;")
        toolbar.addWidget(self._model_label)

        self._btn_settings = QPushButton("⚙ 设置")
        self._btn_settings.clicked.connect(self._open_settings)
        toolbar.addWidget(self._btn_settings)

        main_layout.addLayout(toolbar)

        content = QHBoxLayout()
        self._image_panel = ImagePanel()
        content.addWidget(self._image_panel, 2)
        self._plate_list = PlateListPanel()
        self._plate_list.setMinimumWidth(220)
        self._plate_list.setMaximumWidth(320)
        content.addWidget(self._plate_list, 1)
        main_layout.addLayout(content, 1)

        self._control_bar = ControlBar()
        self._control_bar.play_clicked.connect(self._play)
        self._control_bar.pause_clicked.connect(self._pause)
        self._control_bar.stop_clicked.connect(self._stop)
        self._control_bar.save_clicked.connect(self._save)
        self._control_bar.position_changed.connect(self._on_slider_changed)
        main_layout.addWidget(self._control_bar)

        self._apply_dark_theme()
        self._load_default_model()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #111827; }
            QPushButton {
                background-color: #374151;
                color: #e5e7eb;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #4b5563; }
            QPushButton:pressed { background-color: #1f2937; }
            QLabel { color: #d1d5db; }
            QSlider::groove:horizontal {
                background: #3b3b55;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                width: 12px;
                height: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
            QStatusBar { color: #9ca3af; }
        """)

    def _load_default_model(self):
        try:
            self._engine.load_model()
            name = os.path.basename(self._engine.model_path)
            self._model_label.setText(f"模型: {name}")
        except Exception as e:
            QMessageBox.warning(self, "模型加载失败", f"无法加载默认模型:\n{e}\n\n请在设置中手动选择模型文件。")

    def _stop_thread(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(3000)
        self._thread = None

    def _create_thread(self) -> DetectionThread:
        thread = DetectionThread(self)
        thread.detection_finished.connect(self._on_detection_finished)
        thread.plate_cropped.connect(self._on_plate_cropped)
        thread.fps_updated.connect(self._on_fps_updated)
        thread.error_occurred.connect(self._on_error)
        thread.position_updated.connect(self._on_position_updated)
        thread.video_finished.connect(self._on_video_finished)
        return thread

    def _open_image(self):
        self._stop_thread()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)",
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "无法读取图片文件")
            return
        self._source_name = os.path.splitext(os.path.basename(path))[0]
        self._control_bar.set_image_mode()
        self._thread = self._create_thread()
        self._thread.set_image_source(frame)
        self._thread.start()

    def _open_video(self):
        self._stop_thread()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频",
            "",
            "视频 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)",
        )
        if not path:
            return
        self._source_name = os.path.splitext(os.path.basename(path))[0]
        self._control_bar.set_video_mode()
        self._thread = self._create_thread()
        self._thread.set_video_source("file", path)
        self._thread.start()

    def _open_camera(self):
        self._stop_thread()
        index, ok = QInputDialog.getInt(self, "摄像头", "设备索引:", 0, 0, 10)
        if not ok:
            return
        self._source_name = "camera"
        self._control_bar.set_camera_mode()
        self._thread = self._create_thread()
        self._thread.set_video_source("camera", str(index))
        self._thread.start()

    def _open_stream(self):
        self._stop_thread()
        url, ok = QInputDialog.getText(self, "网络流", "RTSP/HTTP URL:")
        if not ok or not url:
            return
        self._source_name = "stream"
        self._control_bar.set_stream_mode()
        self._control_bar.set_stream_status(False)
        self._thread = self._create_thread()
        self._thread.set_video_source("stream", url)
        self._thread.start()

    def _play(self):
        if self._thread:
            self._thread.resume()

    def _pause(self):
        if self._thread:
            self._thread.pause()

    def _stop(self):
        self._stop_thread()
        self._image_panel.clear()
        self._plate_list.clear()
        self._current_annotated = None

    def _save(self):
        if self._current_annotated is None:
            QMessageBox.information(self, "提示", "没有可保存的结果")
            return
        success = save_annotated_image(self._current_annotated, self, self._source_name)
        if success:
            self.statusBar().showMessage("标注图已保存", 3000)
        else:
            QMessageBox.warning(self, "保存失败", "保存失败：权限不足或路径无效，请选择其他目录。")

    def _open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_path = dialog.get_model_path()
            new_conf = dialog.get_confidence()
            self._engine.set_confidence(new_conf)
            if new_path != self._engine.model_path:
                try:
                    self._engine.load_model(new_path)
                    name = os.path.basename(new_path)
                    self._model_label.setText(f"模型: {name}")
                except Exception as e:
                    QMessageBox.warning(self, "模型加载失败", str(e))

    def _on_slider_changed(self, value: int):
        if self._thread and self._thread.isRunning():
            self._thread.seek_position(value)

    def _on_detection_finished(self, annotated: np.ndarray, detections: list):
        self._current_annotated = annotated.copy()
        self._image_panel.display_frame(annotated)
        self._control_bar.update_detection_count(len(detections))

    def _on_plate_cropped(self, crops: list):
        self._plate_list.update_plates(crops)

    def _on_fps_updated(self, fps: float):
        self._control_bar.update_fps(fps)

    def _on_position_updated(self, pos: int, total: int):
        self._control_bar.set_slider_range(max(total - 1, 0))
        self._control_bar.set_slider_value(pos)

    def _on_video_finished(self):
        self.statusBar().showMessage("视频播放完毕", 3000)

    def _on_error(self, message: str):
        self.statusBar().showMessage(message, 5000)

    def closeEvent(self, event):
        self._stop_thread()
        super().closeEvent(event)
