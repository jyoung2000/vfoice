"""MP4/audio import wizard: extract → (isolate) → analyze → audition → save.

The heavy work runs on a worker thread so the dialog never blocks. The user
picks one of the auto-selected clean clips, fine-trims it, names the profile and
ticks the consent box before saving into the :class:`VoiceLibrary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..audio.dsp import resample, to_mono_float32
from ..audio.playback import AudioPlayer
from ..config import get_config
from ..logging_setup import get_logger
from . import vad
from .extract import extract_for_ingest
from .isolate import DemucsIsolator
from .profile import VoiceLibrary, VoiceProfile

log = get_logger("ingest_dialog")

MEDIA_FILTER = "Media (*.mp4 *.mkv *.mov *.mp3 *.wav *.m4a *.flac *.aac);;All files (*)"


@dataclass
class IngestAnalysis:
    audio_clone: np.ndarray   # 24 kHz mono, used for the saved reference
    sr_clone: int
    audio_audition: np.ndarray  # 44.1 kHz mono, used for previewing
    sr_audition: int
    result: vad.AnalysisResult
    isolated: bool


def analyze_source(
    src: str | Path,
    work_dir: Path,
    ffmpeg_path: str,
    do_isolate: bool,
    progress,
) -> IngestAnalysis:
    """Extract, optionally isolate vocals, and run VAD analysis."""
    import soundfile as sf

    progress("Extracting audio…", 10)
    clone_path, audition_path = extract_for_ingest(src, work_dir, ffmpeg_path)
    clone_audio, clone_sr = sf.read(str(clone_path), dtype="float32", always_2d=False)
    clone_audio = to_mono_float32(clone_audio)
    audition_audio, audition_sr = sf.read(str(audition_path), dtype="float32", always_2d=False)
    audition_audio = to_mono_float32(audition_audio)

    isolated = False
    if do_isolate:
        progress("Isolating vocals (Demucs)… this can take a while", 40)
        vocals, vsr = DemucsIsolator(get_config().device()).isolate(clone_audio, clone_sr)
        clone_audio = resample(vocals, vsr, clone_sr)
        audition_audio = resample(vocals, vsr, audition_sr)
        isolated = True

    progress("Analyzing speech…", 75)
    result = vad.analyze(clone_audio, clone_sr)
    progress("Done", 100)
    return IngestAnalysis(
        audio_clone=clone_audio,
        sr_clone=clone_sr,
        audio_audition=audition_audio,
        sr_audition=audition_sr,
        result=result,
        isolated=isolated,
    )


class _AnalyzeWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, src: Path, work_dir: Path, ffmpeg_path: str, do_isolate: bool) -> None:
        super().__init__()
        self._src = src
        self._work_dir = work_dir
        self._ffmpeg = ffmpeg_path
        self._do_isolate = do_isolate

    def run(self) -> None:
        try:
            analysis = analyze_source(
                self._src, self._work_dir, self._ffmpeg, self._do_isolate,
                lambda msg, pct: self.progress.emit(msg, pct),
            )
            self.finished.emit(analysis)
        except Exception as exc:  # surface to the dialog
            log.exception("ingest analysis failed")
            self.failed.emit(str(exc))


class IngestDialog(QDialog):
    """Wizard that produces a :class:`VoiceProfile` in ``library``."""

    def __init__(self, library: VoiceLibrary, parent: QWidget | None = None,
                 source: str | Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import voice from video / audio")
        self.setMinimumWidth(560)
        self._library = library
        self._cfg = get_config()
        self._player = AudioPlayer(self._cfg.settings.output_device)
        self._analysis: IngestAnalysis | None = None
        self._thread: QThread | None = None
        self._worker: _AnalyzeWorker | None = None
        self.created_profile: VoiceProfile | None = None

        self._source = Path(source) if source else None
        self._build_ui()
        self._refresh_enabled()
        if self._source:
            self._start_analysis()

    # ----- UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Source row
        src_row = QHBoxLayout()
        self._src_label = QLabel(str(self._source) if self._source else "No file chosen")
        self._src_label.setWordWrap(True)
        choose = QPushButton("Choose file…")
        choose.clicked.connect(self._choose_file)
        src_row.addWidget(QLabel("Source:"))
        src_row.addWidget(self._src_label, 1)
        src_row.addWidget(choose)
        layout.addLayout(src_row)

        # Options
        self._isolate_cb = QCheckBox("Isolate vocals with Demucs (for music/noisy sources)")
        self._isolate_cb.setChecked(self._cfg.settings.enable_demucs)
        layout.addWidget(self._isolate_cb)

        analyze_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.clicked.connect(self._start_analysis)
        analyze_row.addWidget(self._analyze_btn)
        analyze_row.addStretch(1)
        layout.addLayout(analyze_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        layout.addWidget(self._progress)
        layout.addWidget(self._status)

        # Analysis results + candidate picker
        self._results_box = QGroupBox("Clean speech clips")
        self._results_box.setVisible(False)
        rb = QVBoxLayout(self._results_box)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        rb.addWidget(self._summary)

        cand_row = QHBoxLayout()
        cand_row.addWidget(QLabel("Candidate:"))
        self._cand_combo = QComboBox()
        self._cand_combo.currentIndexChanged.connect(self._on_candidate_changed)
        cand_row.addWidget(self._cand_combo, 1)
        rb.addLayout(cand_row)

        trim_row = QHBoxLayout()
        self._start_spin = QDoubleSpinBox()
        self._end_spin = QDoubleSpinBox()
        for spin in (self._start_spin, self._end_spin):
            spin.setSuffix(" s")
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
        self._start_spin.valueChanged.connect(self._clamp_trim)
        self._end_spin.valueChanged.connect(self._clamp_trim)
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_play)
        trim_row.addWidget(QLabel("Trim:"))
        trim_row.addWidget(self._start_spin)
        trim_row.addWidget(QLabel("→"))
        trim_row.addWidget(self._end_spin)
        trim_row.addWidget(self._play_btn)
        rb.addLayout(trim_row)
        layout.addWidget(self._results_box)

        # Profile metadata
        meta = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Profile name")
        self._name_edit.textChanged.connect(self._refresh_enabled)
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setFixedHeight(54)
        self._notes_edit.setPlaceholderText("Notes (optional)")
        meta.addRow("Name:", self._name_edit)
        meta.addRow("Notes:", self._notes_edit)
        layout.addLayout(meta)

        self._consent_cb = QCheckBox("I have permission to clone this voice.")
        self._consent_cb.stateChanged.connect(self._refresh_enabled)
        layout.addWidget(self._consent_cb)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ----- actions ----------------------------------------------------------
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose media file", "", MEDIA_FILTER)
        if path:
            self._source = Path(path)
            self._src_label.setText(path)
            self._refresh_enabled()

    def _start_analysis(self) -> None:
        if not self._source:
            self._choose_file()
            if not self._source:
                return
        self._set_busy(True)
        work_dir = self._cfg.paths.tmp / "ingest"
        self._thread = QThread(self)
        self._worker = _AnalyzeWorker(
            self._source, work_dir, self._cfg.settings.ffmpeg_path,
            self._isolate_cb.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.failed.connect(self._on_analysis_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, msg: str, pct: int) -> None:
        self._status.setText(msg)
        self._progress.setValue(pct)

    def _on_analysis_done(self, analysis: IngestAnalysis) -> None:
        self._analysis = analysis
        self._set_busy(False)
        r = analysis.result
        rec = ""
        if r.recommend_isolation and not analysis.isolated:
            rec = ("  ⚠ Speech looks sparse/musical — consider enabling vocal "
                   "isolation and re-analyzing.")
        self._summary.setText(
            f"Speech: {r.speech_ratio * 100:.0f}% · flatness {r.flatness:.2f}"
            f"{' · isolated' if analysis.isolated else ''}.{rec}"
        )
        self._cand_combo.clear()
        for i, c in enumerate(r.candidates):
            self._cand_combo.addItem(
                f"#{i + 1}  {c.start_s:.1f}–{c.end_s:.1f}s  (score {c.score:.2f})"
            )
        self._results_box.setVisible(bool(r.candidates))
        if r.candidates:
            self._cand_combo.setCurrentIndex(0)
            self._on_candidate_changed(0)
        self._refresh_enabled()

    def _on_analysis_failed(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Ingest failed", message)

    def _on_candidate_changed(self, index: int) -> None:
        if not self._analysis or index < 0 or index >= len(self._analysis.result.candidates):
            return
        c = self._analysis.result.candidates[index]
        for spin in (self._start_spin, self._end_spin):
            spin.blockSignals(True)
            spin.setRange(c.start_s, c.end_s)
        self._start_spin.setValue(c.start_s)
        self._end_spin.setValue(c.end_s)
        for spin in (self._start_spin, self._end_spin):
            spin.blockSignals(False)

    def _clamp_trim(self) -> None:
        if self._start_spin.value() > self._end_spin.value() - 0.5:
            self._end_spin.setValue(min(self._end_spin.maximum(),
                                        self._start_spin.value() + 0.5))

    def _toggle_play(self) -> None:
        if self._player.is_playing:
            self._player.stop()
            self._play_btn.setText("▶ Play")
            return
        if not self._analysis:
            return
        a = self._analysis
        i0 = int(self._start_spin.value() * a.sr_audition)
        i1 = int(self._end_spin.value() * a.sr_audition)
        clip = a.audio_audition[i0:i1]
        self._play_btn.setText("⏹ Stop")
        self._player.play(clip, a.sr_audition,
                          on_finished=lambda: self._play_btn.setText("▶ Play"))

    def _save(self) -> None:
        if not self._analysis:
            return
        a = self._analysis
        i0 = int(self._start_spin.value() * a.sr_clone)
        i1 = int(self._end_spin.value() * a.sr_clone)
        clip = a.audio_clone[i0:i1]
        if clip.size < a.sr_clone:  # < 1 s is too short to clone well
            QMessageBox.warning(self, "Clip too short",
                                "Select at least ~1 second of speech.")
            return
        try:
            self.created_profile = self._library.add_profile(
                name=self._name_edit.text(),
                audio=clip,
                sample_rate=a.sr_clone,
                source_filename=self._source.name if self._source else "",
                consent=self._consent_cb.isChecked(),
                notes=self._notes_edit.toPlainText().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.accept()

    # ----- helpers ----------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        if busy:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
        self._analyze_btn.setEnabled(not busy)
        self._isolate_cb.setEnabled(not busy)
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        ready = (
            self._analysis is not None
            and bool(self._name_edit.text().strip())
            and self._consent_cb.isChecked()
        )
        self._buttons.button(QDialogButtonBox.Save).setEnabled(ready)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._player.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        super().closeEvent(event)
