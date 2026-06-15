import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from utils.image_utils import cv2_to_qpixmap
from core.plate_classifier import get_color_display, get_color_bgr


class PlateItemWidget(QFrame):
    def __init__(self, crop_data: dict, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #252540; border-radius: 4px; padding: 4px; margin: 2px 0; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        color_bgr = get_color_bgr(crop_data["color_label"])
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        self._color_tag = QLabel(crop_data["display_name"])
        self._color_tag.setFixedWidth(45)
        self._color_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_tag.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: white; border-radius: 2px; font-size: 9px; padding: 2px;"
        )
        if crop_data["color_label"] == "yellow":
            self._color_tag.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: black; border-radius: 2px; font-size: 9px; padding: 2px;"
            )
        layout.addWidget(self._color_tag)

        thumb = cv2_to_qpixmap(crop_data["image"])
        scaled = thumb.scaled(80, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._thumb_label = QLabel()
        self._thumb_label.setPixmap(scaled)
        self._thumb_label.setFixedSize(85, 35)
        layout.addWidget(self._thumb_label)

        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(4, 0, 4, 0)
        name_label = QLabel(f"车牌 #{crop_data.get('index', '?')}")
        name_label.setStyleSheet("color: #e5e7eb; font-size: 11px;")
        info_layout.addWidget(name_label)

        conf_val = crop_data.get("confidence", 0.0)
        conf_text = f"置信度: {conf_val:.0%} | {crop_data['display_name']}"
        conf_label = QLabel(conf_text)
        conf_label.setStyleSheet("color: #6b7280; font-size: 9px;")
        info_layout.addWidget(conf_label)
        layout.addLayout(info_layout, 1)


class PlateListPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._title_label = QLabel("🔍 检测到的车牌")
        self._title_label.setStyleSheet("color: #9ca3af; font-size: 12px; padding: 4px;")
        self._layout.addWidget(self._title_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(2)
        self._scroll.setWidget(self._container)
        self._layout.addWidget(self._scroll, 1)

        self._empty_label = QLabel("未检测到车牌")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6b7280; font-size: 13px; padding: 40px 0;")
        self._container_layout.addWidget(self._empty_label)

    def update_plates(self, crops: list):
        self._clear_items()
        if not crops:
            self._empty_label = QLabel("未检测到车牌")
            self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._empty_label.setStyleSheet("color: #6b7280; font-size: 13px; padding: 40px 0;")
            self._container_layout.addWidget(self._empty_label)
            return
        for i, crop_data in enumerate(crops):
            crop_data["index"] = i + 1
            item = PlateItemWidget(crop_data)
            self._container_layout.addWidget(item)

    def _clear_items(self):
        while self._container_layout.count():
            child = self._container_layout.takeAt(0)
            widget = child.widget()
            if widget:
                widget.deleteLater()

    def clear(self):
        self._clear_items()
        self._empty_label = QLabel("等待输入...")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6b7280; font-size: 13px; padding: 40px 0;")
        self._container_layout.addWidget(self._empty_label)
