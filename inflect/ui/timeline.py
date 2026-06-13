"""Bottom dock: waveform of the assembled render with playhead and segments.

Shows a min/max envelope of the mix, colored regions per segment matching the
editor's span colors, click-to-seek and space-to-play. Playback uses
:class:`AudioPlayer`; a timer polls its position to drive the playhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..audio.dsp import peak_envelope
from ..audio.playback import AudioPlayer
from ..config import get_config
from .editor import span_color


@dataclass
class SegmentMarker:
    seg_id: int
    start_s: float
    end_s: float
    color_idx: int
    job_hash: str


class TimelinePanel(QWidget):
    seek_changed = Signal(float)
    rerender_segment = Signal(int)  # seg_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = AudioPlayer(get_config().settings.output_device)
        self._sr = 24000
        self._audio = np.zeros(0, dtype=np.float32)
        self._markers: list[SegmentMarker] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self.toggle_play)
        self._stop_btn = QPushButton("⏹")
        self._stop_btn.clicked.connect(self.stop)
        self._time_lbl = QLabel("0.00 / 0.00 s")
        controls.addWidget(self._play_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._time_lbl)
        controls.addStretch(1)
        layout.addLayout(controls)

        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1f22")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getAxis("left").setStyle(showValues=False)
        self._plot.setYRange(-1.0, 1.0)
        self._curve = self._plot.plot(pen=pg.mkPen("#61afef", width=1))
        self._playhead = pg.InfiniteLine(angle=90, movable=False,
                                         pen=pg.mkPen("#e06c75", width=2))
        self._plot.addItem(self._playhead)
        self._plot.scene().sigMouseClicked.connect(self._on_click)
        layout.addWidget(self._plot, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._update_playhead)

    # ----- data -------------------------------------------------------------
    def set_audio(self, audio: np.ndarray, sr: int) -> None:
        self._audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._sr = int(sr)
        mins, maxs = peak_envelope(self._audio, 3000)
        n = len(maxs)
        if n == 0:
            self._curve.setData([], [])
            self._time_lbl.setText("0.00 / 0.00 s")
            return
        dur = self._audio.size / self._sr
        # Build a single polyline that traces max then min for a filled look.
        xs = np.linspace(0, dur, n)
        x = np.concatenate([xs, xs[::-1]])
        y = np.concatenate([maxs, mins[::-1]])
        self._curve.setData(x, y)
        self._plot.setXRange(0, dur)
        self._time_lbl.setText(f"0.00 / {dur:.2f} s")
        self._playhead.setValue(0)

    def set_segments(self, markers: list[SegmentMarker]) -> None:
        """Draw colored regions behind the waveform for each segment."""
        for item in list(self._plot.items()):
            if isinstance(item, pg.LinearRegionItem):
                self._plot.removeItem(item)
        self._markers = markers
        for m in markers:
            color = span_color(m.color_idx)
            color.setAlpha(40)
            region = pg.LinearRegionItem(
                values=(m.start_s, m.end_s), movable=False, brush=color
            )
            region.setZValue(-10)
            self._plot.addItem(region)

    # ----- playback ---------------------------------------------------------
    def toggle_play(self) -> None:
        if self._player.is_playing:
            self._player.stop()
            self._timer.stop()
            self._play_btn.setText("▶ Play")
            return
        if self._audio.size == 0:
            return
        start = int(self._playhead.value() * self._sr)
        if start >= self._audio.size - 1:
            start = 0
        self._play_btn.setText("⏸ Pause")
        self._timer.start()
        self._player.play(self._audio, self._sr, start_frame=start,
                          on_finished=self._on_finished)

    def stop(self) -> None:
        self._player.stop()
        self._timer.stop()
        self._playhead.setValue(0)
        self._play_btn.setText("▶ Play")
        self._update_time_label(0.0)

    def _on_finished(self) -> None:
        self._timer.stop()
        self._play_btn.setText("▶ Play")

    def _update_playhead(self) -> None:
        t = self._player.current_time
        self._playhead.setValue(t)
        self._update_time_label(t)

    def _update_time_label(self, t: float) -> None:
        dur = self._audio.size / self._sr if self._sr else 0.0
        self._time_lbl.setText(f"{t:.2f} / {dur:.2f} s")

    def _on_click(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        vb = self._plot.getPlotItem().vb
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        x = vb.mapSceneToView(pos).x()
        x = max(0.0, min(x, self._audio.size / self._sr if self._sr else 0.0))
        self._playhead.setValue(x)
        self._player.seek_time(x)
        self.seek_changed.emit(x)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt signature
        if event.key() == Qt.Key_Space:
            self.toggle_play()
            event.accept()
            return
        super().keyPressEvent(event)
