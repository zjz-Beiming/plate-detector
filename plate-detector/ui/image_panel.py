import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from utils.image_utils import cv2_to_qpixmap


class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title_label = QLabel("📷 原始图像 + 检测标注")
        self._title_label.setStyleSheet("color: #9ca3af; font-size: 12px; padding: 4px;")
        layout.addWidget(self._title_label)

        self._image_label = QLabel("等待输入...")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #252540; border-radius: 4px; color: #888; font-size: 14px; min-height: 300px;"
        )
        self._image_label.setMinimumSize(400, 300)
        layout.addWidget(self._image_label, 1)

    def display_frame(self, frame: np.ndarray):
        pixmap = cv2_to_qpixmap(frame)
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def clear(self):
        self._image_label.clear()
        self._image_label.setText("等待输入...")

    def show_no_detection(self, frame: np.ndarray):
        self.display_frame(frame)
