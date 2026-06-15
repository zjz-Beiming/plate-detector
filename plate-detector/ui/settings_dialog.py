import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QDoubleSpinBox,
    QPushButton, QFileDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt
from core.detector import DetectionEngine


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(400)
        self._engine = DetectionEngine()

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._model_path_edit = QLineEdit(self._engine.model_path)
        self._model_path_edit.setReadOnly(True)
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path_edit, 1)
        model_row.addWidget(model_btn)
        form.addRow("模型权重路径:", model_row)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(self._engine.confidence)
        self._conf_spin.setDecimals(2)
        form.addRow("置信度阈值:", self._conf_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", os.path.dirname(self._model_path_edit.text()),
            "PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)",
        )
        if path:
            self._model_path_edit.setText(path)

    def get_model_path(self) -> str:
        return self._model_path_edit.text()

    def get_confidence(self) -> float:
        return self._conf_spin.value()
