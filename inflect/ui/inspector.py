"""Right dock: edit the inflection for the current selection.

Reads the selection's current inflection into its controls and emits the edited
inflection on Apply / Set-as-default / Preview. Includes a small live bar
visualization of the 8 emotion dimensions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..document.spans import EMOTIONS, Inflection
from .editor import span_color


class EmotionBars(QWidget):
    """Tiny 8-bar visualization of the emotion vector."""

    def __init__(self) -> None:
        super().__init__()
        self._values = [0.0] * len(EMOTIONS)
        self.setMinimumHeight(56)

    def set_values(self, values: list[float]) -> None:
        self._values = list(values)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt signature
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        n = len(self._values)
        if n == 0:
            return
        w = self.width() / n
        h = self.height()
        for i, v in enumerate(self._values):
            bar_h = max(2.0, v * (h - 4))
            x = i * w + 2
            painter.fillRect(int(x), int(h - bar_h), int(w - 4), int(bar_h),
                             span_color(i))


class InspectorPanel(QWidget):
    apply_to_selection = Signal(object)   # Inflection
    set_as_default = Signal(object)       # Inflection
    preview_requested = Signal(object)    # Inflection

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._scope = QLabel("No selection — editing document default")
        self._scope.setWordWrap(True)
        self._scope.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(self._scope)

        # Emotion sliders
        emo_box = QGroupBox("Emotion")
        emo_form = QFormLayout(emo_box)
        self._emotion_sliders: list[QSlider] = []
        self._emotion_labels: list[QLabel] = []
        for name in EMOTIONS:
            row = QHBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.valueChanged.connect(self._on_changed)
            value_lbl = QLabel("0.00")
            value_lbl.setFixedWidth(34)
            row.addWidget(slider, 1)
            row.addWidget(value_lbl)
            container = QWidget()
            container.setLayout(row)
            emo_form.addRow(name.capitalize(), container)
            self._emotion_sliders.append(slider)
            self._emotion_labels.append(value_lbl)
        self._bars = EmotionBars()
        emo_form.addRow(self._bars)
        layout.addWidget(emo_box)

        # Delivery description + alpha
        deliver_box = QGroupBox("Delivery")
        deliver_form = QFormLayout(deliver_box)
        self._emo_text = QLineEdit()
        self._emo_text.setPlaceholderText('e.g. "whispering, almost crying"')
        self._emo_text.textChanged.connect(self._on_changed)
        deliver_form.addRow("Describe:", self._emo_text)

        self._alpha = QSlider(Qt.Horizontal)
        self._alpha.setRange(0, 100)
        self._alpha.setValue(80)
        self._alpha.valueChanged.connect(self._on_changed)
        self._alpha_lbl = QLabel("0.80")
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self._alpha, 1)
        alpha_row.addWidget(self._alpha_lbl)
        alpha_w = QWidget()
        alpha_w.setLayout(alpha_row)
        deliver_form.addRow("Strength:", alpha_w)
        layout.addWidget(deliver_box)

        # Timing
        timing_box = QGroupBox("Timing")
        timing_form = QFormLayout(timing_box)
        self._speed = QSlider(Qt.Horizontal)
        self._speed.setRange(50, 150)
        self._speed.setValue(100)
        self._speed.valueChanged.connect(self._on_changed)
        self._speed_lbl = QLabel("1.00×")
        speed_row = QHBoxLayout()
        speed_row.addWidget(self._speed, 1)
        speed_row.addWidget(self._speed_lbl)
        speed_w = QWidget()
        speed_w.setLayout(speed_row)
        timing_form.addRow("Speed:", speed_w)

        self._pause = QSpinBox()
        self._pause.setRange(0, 5000)
        self._pause.setSingleStep(50)
        self._pause.setSuffix(" ms")
        self._pause.valueChanged.connect(self._on_changed)
        timing_form.addRow("Pause after:", self._pause)
        layout.addWidget(timing_box)

        # Actions
        self._apply_btn = QPushButton("Apply to selection")
        self._default_btn = QPushButton("Set as document default")
        self._preview_btn = QPushButton("Preview this segment")
        self._apply_btn.clicked.connect(lambda: self.apply_to_selection.emit(self.build_inflection()))
        self._default_btn.clicked.connect(lambda: self.set_as_default.emit(self.build_inflection()))
        self._preview_btn.clicked.connect(
            lambda: self.preview_requested.emit(self.build_inflection()))
        layout.addWidget(self._apply_btn)
        layout.addWidget(self._default_btn)
        layout.addWidget(self._preview_btn)
        layout.addStretch(1)

    # ----- state <-> controls ----------------------------------------------
    def build_inflection(self) -> Inflection:
        vec = [s.value() / 100.0 for s in self._emotion_sliders]
        has_emotion = any(v > 0 for v in vec)
        return Inflection(
            emotion_vector=vec if has_emotion else None,
            emo_text=self._emo_text.text().strip() or None,
            emo_alpha=self._alpha.value() / 100.0,
            speed=self._speed.value() / 100.0,
            pause_after_ms=self._pause.value(),
        )

    def set_inflection(self, infl: Inflection, has_selection: bool) -> None:
        self._updating = True
        vec = infl.emotion_vector or [0.0] * len(EMOTIONS)
        for slider, lbl, v in zip(self._emotion_sliders, self._emotion_labels, vec):
            slider.setValue(int(round(v * 100)))
            lbl.setText(f"{v:.2f}")
        self._emo_text.setText(infl.emo_text or "")
        self._alpha.setValue(int(round(infl.emo_alpha * 100)))
        self._alpha_lbl.setText(f"{infl.emo_alpha:.2f}")
        self._speed.setValue(int(round(infl.speed * 100)))
        self._speed_lbl.setText(f"{infl.speed:.2f}×")
        self._pause.setValue(infl.pause_after_ms)
        self._bars.set_values(vec)
        self._scope.setText(
            "Editing selection" if has_selection else "No selection — editing document default"
        )
        self._apply_btn.setEnabled(has_selection)
        self._updating = False

    def _on_changed(self) -> None:
        if self._updating:
            return
        for slider, lbl in zip(self._emotion_sliders, self._emotion_labels):
            lbl.setText(f"{slider.value() / 100.0:.2f}")
        self._alpha_lbl.setText(f"{self._alpha.value() / 100.0:.2f}")
        self._speed_lbl.setText(f"{self._speed.value() / 100.0:.2f}×")
        self._bars.set_values([s.value() / 100.0 for s in self._emotion_sliders])
