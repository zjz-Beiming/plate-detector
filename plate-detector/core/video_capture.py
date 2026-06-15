import cv2


class VideoCaptureWrapper:
    def __init__(self):
        self._cap = None
        self._source_type = None
        self._source_path = None

    def open_file(self, path: str) -> bool:
        self.close()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._source_type = "file"
        self._source_path = path
        return True

    def open_camera(self, index: int = 0) -> bool:
        self.close()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._source_type = "camera"
        self._source_path = str(index)
        return True

    def open_stream(self, url: str) -> bool:
        self.close()
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._source_type = "stream"
        self._source_path = url
        return True

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def get(self, prop: int):
        if self._cap is None:
            return 0.0
        return self._cap.get(prop)

    def set(self, prop: int, value):
        if self._cap is not None:
            self._cap.set(prop, value)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._source_type = None
        self._source_path = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def frame_count(self) -> int:
        if self._source_type == "file" and self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def fps(self) -> float:
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    @property
    def current_pos(self) -> int:
        if self._source_type == "file" and self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return -1
