import os
import cv2
from datetime import datetime
from PyQt6.QtWidgets import QFileDialog


_last_save_dir = ""


def save_annotated_image(frame, parent_widget=None, source_name="image"):
    global _last_save_dir
    save_dir = _last_save_dir or os.path.expanduser("~")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{source_name}_annotated_{timestamp}.jpg"

    file_path, _ = QFileDialog.getSaveFileName(
        parent_widget,
        "保存标注图",
        os.path.join(save_dir, default_name),
        "JPEG (*.jpg);;PNG (*.png)",
    )

    if not file_path:
        return False

    try:
        cv2.imwrite(file_path, frame)
        _last_save_dir = os.path.dirname(file_path)
        return True
    except cv2.error:
        return False
