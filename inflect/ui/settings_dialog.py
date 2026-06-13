"""Preferences dialog: model/ffmpeg paths, output device, audio + engine knobs."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..logging_setup import get_logger

log = get_logger("settings_dialog")


def _output_devices() -> list[tuple[str, object]]:
    """List output-capable devices as ``(label, device_id_or_None)``."""
    devices: list[tuple[str, object]] = [("System default", None)]
    try:
        import sounddevice as sd

        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                devices.append((f"{dev['name']}", dev["name"]))
    except Exception as exc:
        log.debug("could not enumerate audio devices: %s", exc)
    return devices


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)
        self._cfg = config
        s = config.settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # ffmpeg path
        self._ffmpeg = QLineEdit(s.ffmpeg_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_ffmpeg)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self._ffmpeg, 1)
        ffmpeg_row.addWidget(browse)
        ffmpeg_w = QWidget()
        ffmpeg_w.setLayout(ffmpeg_row)
        form.addRow("ffmpeg path:", ffmpeg_w)

        # output device
        self._device = QComboBox()
        self._devices = _output_devices()
        for label, _ in self._devices:
            self._device.addItem(label)
        current = next((i for i, (_, dev) in enumerate(self._devices)
                        if dev == s.output_device), 0)
        self._device.setCurrentIndex(current)
        form.addRow("Output device:", self._device)

        # default engine
        self._engine = QComboBox()
        self._engine.addItems(["chatterbox", "indextts2"])
        self._engine.setCurrentText(s.default_engine)
        form.addRow("Default engine:", self._engine)

        # toggles
        self._fp16 = QCheckBox("Use fp16 (recommended on 12 GB GPUs)")
        self._fp16.setChecked(s.use_fp16)
        self._kernel = QCheckBox("Use CUDA kernel when available")
        self._kernel.setChecked(s.use_cuda_kernel)
        self._demucs = QCheckBox("Enable Demucs vocal isolation by default")
        self._demucs.setChecked(s.enable_demucs)
        form.addRow(self._fp16)
        form.addRow(self._kernel)
        form.addRow(self._demucs)

        # audio
        self._lufs = QDoubleSpinBox()
        self._lufs.setRange(-30.0, -6.0)
        self._lufs.setValue(s.target_lufs)
        self._lufs.setSuffix(" LUFS")
        form.addRow("Target loudness:", self._lufs)

        self._crossfade = QSpinBox()
        self._crossfade.setRange(0, 100)
        self._crossfade.setValue(s.crossfade_ms)
        self._crossfade.setSuffix(" ms")
        form.addRow("Crossfade:", self._crossfade)

        # HF endpoint
        self._hf = QLineEdit(s.hf_endpoint or "")
        self._hf.setPlaceholderText("optional Hugging Face mirror, e.g. https://hf-mirror.com")
        form.addRow("HF endpoint:", self._hf)

        # Fish license
        self._fish_license = QCheckBox(
            "I accept the Fish Speech research/personal-use license (Phase 6)")
        self._fish_license.setChecked(s.accept_fish_license)
        form.addRow(self._fish_license)

        layout.addLayout(form)
        note = QLabel("Model weights download to the app data folder on first use.")
        note.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate ffmpeg")
        if path:
            self._ffmpeg.setText(path)

    def _accept(self) -> None:
        s = self._cfg.settings
        s.ffmpeg_path = self._ffmpeg.text().strip() or "ffmpeg"
        s.output_device = self._devices[self._device.currentIndex()][1]
        s.default_engine = self._engine.currentText()
        s.use_fp16 = self._fp16.isChecked()
        s.use_cuda_kernel = self._kernel.isChecked()
        s.enable_demucs = self._demucs.isChecked()
        s.target_lufs = self._lufs.value()
        s.crossfade_ms = self._crossfade.value()
        s.hf_endpoint = self._hf.text().strip() or None
        s.accept_fish_license = self._fish_license.isChecked()
        self.accept()
