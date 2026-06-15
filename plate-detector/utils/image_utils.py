import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame.ndim == 2:
        h, w = frame.shape
        bytes_per_line = w
        q_img = QImage(frame.copy().data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    elif frame.shape[2] == 4:
        h, w, _ = frame.shape
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * w
        q_img = QImage(bgra.copy().data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
    else:
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(rgb.copy().data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_img)


def draw_detections(frame: np.ndarray, results: list) -> np.ndarray:
    annotated = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        conf = r["confidence"]
        label = r.get("label", "plate")
        color = r.get("color", (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return annotated


def crop_plate(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]
