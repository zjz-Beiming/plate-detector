import cv2
import numpy as np


PLATE_COLORS = {
    "blue": ((100, 150, 50), (130, 255, 255)),
    "yellow": ((20, 150, 50), (40, 255, 255)),
    "green": ((35, 150, 50), (80, 255, 255)),
}

COLOR_LABELS = {"blue": "蓝牌", "yellow": "黄牌", "green": "绿牌"}


def classify_plate_color(plate_crop: np.ndarray) -> str:
    if plate_crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    max_ratio = 0.0
    best_label = "unknown"
    for name, (lower, upper) in PLATE_COLORS.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = cv2.countNonZero(mask) / mask.size
        if ratio > max_ratio:
            max_ratio = ratio
            best_label = name
    if max_ratio < 0.1:
        return "unknown"
    return best_label


def get_color_bgr(label: str) -> tuple:
    mapping = {
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "unknown": (200, 200, 200),
    }
    return mapping.get(label, (200, 200, 200))


def get_color_display(label: str) -> str:
    return COLOR_LABELS.get(label, "未知")
