import time
import threading
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from core.detector import DetectionEngine
from core.plate_classifier import classify_plate_color, get_color_bgr, get_color_display
from core.video_capture import VideoCaptureWrapper
from utils.image_utils import draw_detections, crop_plate


class DetectionThread(QThread):
    detection_finished = pyqtSignal(np.ndarray, list)
    plate_cropped = pyqtSignal(list)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    position_updated = pyqtSignal(int, int)
    video_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._paused = threading.Event()
        self._paused.set()
        self._source_type = None
        self._source_path = None
        self._frame = None
        self._engine = DetectionEngine()
        self._cap = None

    def set_image_source(self, frame: np.ndarray):
        self._source_type = "image"
        self._frame = frame.copy()

    def set_video_source(self, source_type: str, source_path: str):
        self._source_type = source_type
        self._source_path = source_path

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def stop(self):
        self._running = False
        self._paused.set()

    def seek_position(self, position: int):
        if self._cap and self._cap.is_opened and self._cap.source_type == "file":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    def run(self):
        if self._source_type == "image":
            self._run_image()
        elif self._source_type in ("file", "camera", "stream"):
            self._run_video()

    def _run_image(self):
        if self._frame is None:
            return
        self._running = True
        detections = self._engine.detect(self._frame)
        enriched = self._enrich_detections(self._frame, detections)
        annotated = draw_detections(self._frame, enriched)
        crops = self._crop_plates(self._frame, enriched)
        self.detection_finished.emit(annotated, enriched)
        self.plate_cropped.emit(crops)

    def _run_video(self):
        self._cap = VideoCaptureWrapper()
        opener = {
            "file": lambda: self._cap.open_file(self._source_path),
            "camera": lambda: self._cap.open_camera(int(self._source_path)),
            "stream": lambda: self._cap.open_stream(self._source_path),
        }
        open_fn = opener.get(self._source_type)
        if open_fn is None or not open_fn():
            self.error_occurred.emit(f"无法打开输入源: {self._source_path}")
            self._cap = None
            return

        if self._source_type == "file":
            total = self._cap.frame_count
            self.position_updated.emit(0, total)

        self._running = True
        prev_time = time.time()
        frame_count = 0

        while self._running:
            self._paused.wait()
            if not self._running:
                break

            ret, frame = self._cap.read()
            if not ret:
                if self._source_type == "stream":
                    retries = 0
                    while retries < 3 and self._running:
                        time.sleep(1)
                        if self._cap.open_stream(self._source_path):
                            break
                        retries += 1
                    if retries >= 3:
                        self.error_occurred.emit("网络流断开，重连失败")
                        break
                    continue
                if self._source_type == "file":
                    self.video_finished.emit()
                break

            detections = self._engine.detect(frame)
            enriched = self._enrich_detections(frame, detections)
            annotated = draw_detections(frame, enriched)
            crops = self._crop_plates(frame, enriched)

            self.detection_finished.emit(annotated, enriched)
            self.plate_cropped.emit(crops)

            if self._source_type == "file":
                pos = self._cap.current_pos
                total = self._cap.frame_count
                self.position_updated.emit(pos, total)

            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                self.fps_updated.emit(frame_count / elapsed)
                frame_count = 0
                prev_time = now

        if self._cap:
            self._cap.close()
            self._cap = None
