import os
import urllib.request
from ultralytics import YOLO

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_URL = "https://github.com/ShawnHymel/yolov8-plate-detection/releases/download/v1.0/plate_yolov8n.pt"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "plate_yolov8n.pt")


class DetectionEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._confidence = 0.5
            cls._instance._model_path = None
        return cls._instance

    def load_model(self, model_path: str = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if not os.path.isfile(model_path):
            self._download_model(model_path)
        self._model = YOLO(model_path)
        self._model_path = model_path

    def _download_model(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, save_path)

    def detect(self, frame):
        if self._model is None:
            self.load_model()
        results = self._model(frame, conf=self._confidence, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self._model.names.get(cls_id, "plate")
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_name": class_name,
                })
        return detections

    def set_confidence(self, threshold: float):
        self._confidence = max(0.0, min(1.0, threshold))

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def model_path(self) -> str:
        return self._model_path or DEFAULT_MODEL_PATH

    @property
    def model_loaded(self) -> bool:
        return self._model is not None
