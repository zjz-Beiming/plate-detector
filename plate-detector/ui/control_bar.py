from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, pyqtSignal


class ControlBar(QWidget):
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    position_changed = pyqtSignal(int)

    MODE_IMAGE = "image"
    MODE_VIDEO = "video"
    MODE_CAMERA = "camera"
    MODE_STREAM = "stream"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = self.MODE_IMAGE
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self.setStyleSheet("background-color: #2d2d44; border-radius: 6px;")

        self._play_btn = QPushButton("▶ 播放")
        self._play_btn.clicked.connect(self.play_clicked.emit)
        self._layout.addWidget(self._play_btn)

        self._pause_btn = QPushButton("⏸ 暂停")
        self._pause_btn.clicked.connect(self.pause_clicked.emit)
        self._layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._layout.addWidget(self._stop_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self.position_changed.emit)
        self._layout.addWidget(self._slider, 1)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setStyleSheet("color: #6b7280; font-size: 10px;")
        self._layout.addWidget(self._time_label)

        sep = QLabel("|")
        sep.setStyleSheet("color: #444;")
        self._layout.addWidget(sep)

        self._save_btn = QPushButton("💾 保存标注图")
        self._save_btn.clicked.connect(self.save_clicked.emit)
        self._layout.addWidget(self._save_btn)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444;")
        self._layout.addWidget(sep2)

        self._status_label = QLabel("FPS: -- | 检测数: 0")
        self._status_label.setStyleSheet("color: #6b7280; font-size: 10px;")
        self._layout.addWidget(self._status_label)

        self._stream_status = QLabel("")
        self._stream_status.setStyleSheet("color: #ef4444; font-size: 10px;")
        self._layout.addWidget(self._stream_status)

        self._set_mode(self.MODE_IMAGE)

    def _set_mode(self, mode: str):
        self._mode = mode
        is_image = mode == self.MODE_IMAGE
        self._play_btn.setVisible(not is_image)
        self._pause_btn.setVisible(not is_image)
        self._stop_btn.setVisible(not is_image)
        self._slider.setVisible(mode == self.MODE_VIDEO)
        self._time_label.setVisible(mode == self.MODE_VIDEO)
        self._stream_status.setVisible(mode == self.MODE_STREAM)

    def set_image_mode(self):
        self._set_mode(self.MODE_IMAGE)

    def set_video_mode(self):
        self._set_mode(self.MODE_VIDEO)

    def set_camera_mode(self):
        self._set_mode(self.MODE_CAMERA)

    def set_stream_mode(self):
        self._set_mode(self.MODE_STREAM)

    def update_fps(self, fps: float):
        current = self._status_label.text()
        prefix = current.rsplit("|", 1)[0] if "|" in current else ""
        self._status_label.setText(f"{prefix}| FPS: {fps:.1f} | 检测数: ")

    def update_detection_count(self, count: int):
        current = self._status_label.text()
        parts = current.rsplit("检测数:", 1)
        if len(parts) == 2:
            self._status_label.setText(f"{parts[0]}检测数: {count}")
        else:
            self._status_label.setText(f"FPS: -- | 检测数: {count}")

    def update_time(self, current_sec: float, total_sec: float):
        self._time_label.setText(f"{self._fmt(current_sec)} / {self._fmt(total_sec)}")

    def set_slider_range(self, maximum: int):
        self._slider.setRange(0, maximum)

    def set_slider_value(self, value: int):
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)

    def set_stream_status(self, connected: bool):
        if connected:
            self._stream_status.setText("● 已连接")
            self._stream_status.setStyleSheet("color: #22c55e; font-size: 10px;")
        else:
            self._stream_status.setText("● 未连接")
            self._stream_status.setStyleSheet("color: #ef4444; font-size: 10px;")

    @staticmethod
    def _fmt(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"
