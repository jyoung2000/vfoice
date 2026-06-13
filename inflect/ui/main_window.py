"""The main window: editor + docks + toolbar, wired to the synth pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QWidget,
)

from ..config import get_config
from ..document.project_io import Project, load_project, save_project
from ..document.segmenter import SegmentJob, segment_document
from ..logging_setup import get_logger
from ..models.model_manager import ModelManager
from ..synth.pipeline import SynthesisPipeline
from ..synth.worker import PreviewWorker, SynthWorker
from .editor import InflectionTextEdit
from .inspector import InspectorPanel
from .timeline import SegmentMarker, TimelinePanel
from .voice_library import VoiceLibraryPanel

log = get_logger("main_window")

ENGINE_MODES = [("Draft (Chatterbox)", "chatterbox"), ("Final (IndexTTS-2)", "indextts2")]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Inflect Studio")
        self.resize(1400, 900)
        self._cfg = get_config()
        self._manager = ModelManager(self._cfg.device())
        self._pipeline = SynthesisPipeline(self._manager, self._cfg.paths.cache, self._cfg.settings)
        self._project = Project(engine=self._cfg.settings.default_engine)
        self._project_path: Path | None = None
        self._mix = np.zeros(0, dtype=np.float32)
        self._mix_sr = 24000
        self._syncing_profile = False

        # worker handles (kept alive while running)
        self._synth_thread: QThread | None = None
        self._synth_worker: SynthWorker | None = None
        self._preview_thread: QThread | None = None
        self._preview_worker: PreviewWorker | None = None
        self._preview_player = None

        self._build_central()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._wire()
        self._refresh_profiles()
        self._update_inspector()

    # ----- construction -----------------------------------------------------
    def _build_central(self) -> None:
        self.editor = InflectionTextEdit()
        self.editor.set_model(self._project.document)
        self.setCentralWidget(self.editor)

    def _build_docks(self) -> None:
        self.library = VoiceLibraryPanel()
        left = QDockWidget("Voice Library", self)
        left.setWidget(self.library)
        self.addDockWidget(Qt.LeftDockWidgetArea, left)

        self.inspector = InspectorPanel()
        right = QDockWidget("Inspector", self)
        right.setWidget(self.inspector)
        self.addDockWidget(Qt.RightDockWidgetArea, right)

        self.timeline = TimelinePanel()
        bottom = QDockWidget("Timeline", self)
        bottom.setWidget(self.timeline)
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Voice: "))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(160)
        tb.addWidget(self.profile_combo)

        tb.addWidget(QLabel("  Engine: "))
        self.engine_combo = QComboBox()
        for label, _ in ENGINE_MODES:
            self.engine_combo.addItem(label)
        tb.addWidget(self.engine_combo)
        tb.addSeparator()

        self.synth_action = QAction("▶ Synthesize All", self)
        self.cancel_action = QAction("⏹ Cancel", self)
        self.cancel_action.setEnabled(False)
        self.export_action = QAction("Export…", self)
        tb.addAction(self.synth_action)
        tb.addAction(self.cancel_action)
        tb.addAction(self.export_action)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in [
            ("New", self.new_project, QKeySequence.New),
            ("Open…", self.open_project, QKeySequence.Open),
            ("Save", self.save_project_action, QKeySequence.Save),
            ("Save As…", self.save_project_as, QKeySequence.SaveAs),
            ("Export Audio…", self.export_audio, None),
        ]:
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        settings_menu = self.menuBar().addMenu("&Settings")
        prefs = QAction("Preferences…", self)
        prefs.triggered.connect(self.open_settings)
        settings_menu.addAction(prefs)

    def _build_statusbar(self) -> None:
        self.vram_label = QLabel("VRAM: n/a")
        self.vram_bar = QProgressBar()
        self.vram_bar.setMaximumWidth(140)
        self.vram_bar.setRange(0, 100)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.vram_label)
        self.statusBar().addPermanentWidget(self.vram_bar)
        self._vram_timer = QTimer(self)
        self._vram_timer.setInterval(1500)
        self._vram_timer.timeout.connect(self._update_vram)
        self._vram_timer.start()
        self._update_vram()

    def _wire(self) -> None:
        self.editor.selection_changed.connect(self._update_inspector)
        self.editor.apply_requested.connect(self._apply_from_inspector)
        self.editor.clear_requested.connect(self.editor.clear_inflection_on_selection)
        self.editor.pause_requested.connect(self._insert_pause)

        self.inspector.apply_to_selection.connect(self._apply_inflection)
        self.inspector.set_as_default.connect(self._set_default_inflection)
        self.inspector.preview_requested.connect(self._preview_segment)
        self.inspector.audition_performance.connect(self._audition_performance)

        self.library.profile_selected.connect(self._on_library_selected)
        self.profile_combo.currentIndexChanged.connect(self._on_combo_selected)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        self.synth_action.triggered.connect(self.synthesize_all)
        self.cancel_action.triggered.connect(self.cancel_synthesis)
        self.export_action.triggered.connect(self.export_audio)
        self.timeline.rerender_segment.connect(self._rerender_segment)

    # ----- profile syncing --------------------------------------------------
    def _refresh_profiles(self) -> None:
        self._syncing_profile = True
        self.profile_combo.clear()
        self.profile_combo.addItem("— none —", None)
        for p in self.library.library.list():
            self.profile_combo.addItem(p.name, p.id)
        # restore selection
        pid = self._project.document.voice_profile_id
        if pid:
            idx = self.profile_combo.findData(pid)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        self._syncing_profile = False

    def _on_library_selected(self, profile_id) -> None:
        if self._syncing_profile:
            return
        self._project.document.voice_profile_id = profile_id
        idx = self.profile_combo.findData(profile_id)
        if idx >= 0:
            self._syncing_profile = True
            self.profile_combo.setCurrentIndex(idx)
            self._syncing_profile = False

    def _on_combo_selected(self, _index: int) -> None:
        if self._syncing_profile:
            return
        pid = self.profile_combo.currentData()
        self._project.document.voice_profile_id = pid
        if pid:
            self._syncing_profile = True
            self.library.select_profile(pid)
            self._syncing_profile = False

    def _on_engine_changed(self, index: int) -> None:
        self._project.engine = ENGINE_MODES[index][1]

    def _current_profile(self):
        pid = self._project.document.voice_profile_id
        return self.library.library.get(pid) if pid else None

    # ----- inspector wiring -------------------------------------------------
    def _update_inspector(self) -> None:
        self.inspector.set_inflection(self.editor.current_inflection(), self.editor.has_selection())

    def _apply_from_inspector(self) -> None:
        self._apply_inflection(self.inspector.build_inflection())

    def _apply_inflection(self, inflection) -> None:
        self.editor.apply_inflection_to_selection(inflection)

    def _set_default_inflection(self, inflection) -> None:
        self.editor.set_default_inflection(inflection)
        self.statusBar().showMessage("Document default delivery updated.", 3000)

    def _insert_pause(self) -> None:
        ms, ok = QInputDialog.getInt(self, "Insert pause", "Pause (ms):", 300, 0, 5000, 50)
        if not ok:
            return
        infl = self.editor.current_inflection()
        infl.pause_after_ms = ms
        self.editor.apply_inflection_to_selection(infl)

    # ----- synthesis --------------------------------------------------------
    def _build_jobs(self) -> list[SegmentJob]:
        doc = self.editor.model()
        return segment_document(
            doc, voice_profile=self._current_profile(),
            engine=self._project.engine, engine_params={},
        )

    def synthesize_all(self) -> None:
        if self._synth_thread is not None:
            return
        doc = self.editor.model()
        if not doc.text.strip():
            QMessageBox.information(self, "Nothing to synthesize", "Type some text first.")
            return
        if self._current_profile() is None:
            QMessageBox.warning(self, "No voice selected",
                                "Choose or import a voice profile first.")
            return
        jobs = self._build_jobs()
        if not jobs:
            return

        self._set_synth_running(True)
        self._synth_thread = QThread(self)
        self._synth_worker = SynthWorker(self._pipeline, jobs)
        self._synth_worker.moveToThread(self._synth_thread)
        self._synth_thread.started.connect(self._synth_worker.run)
        self._synth_worker.progress.connect(self._on_synth_progress)
        self._synth_worker.finished.connect(self._on_synth_finished)
        self._synth_worker.failed.connect(self._on_synth_failed)
        self._synth_worker.cancelled.connect(self._on_synth_cancelled)
        for sig in (self._synth_worker.finished, self._synth_worker.failed,
                    self._synth_worker.cancelled):
            sig.connect(self._synth_thread.quit)
        self._synth_thread.finished.connect(self._cleanup_synth_thread)
        self._synth_thread.start()

    def cancel_synthesis(self) -> None:
        if self._synth_worker:
            self._synth_worker.cancel()
            self.statusBar().showMessage("Cancelling…", 2000)

    def _on_synth_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.statusBar().showMessage(message)

    def _on_synth_finished(self, mix: np.ndarray, sr: int) -> None:
        self._mix = np.asarray(mix, dtype=np.float32)
        self._mix_sr = sr
        self.timeline.set_audio(self._mix, sr)
        self.timeline.set_segments(self._build_markers())
        self.statusBar().showMessage(f"Rendered {self._mix.size / sr:.1f}s of audio.", 5000)

    def _on_synth_failed(self, message: str) -> None:
        from .error_dialog import show_error

        show_error(self, "Synthesis failed", message)

    def _on_synth_cancelled(self) -> None:
        self.statusBar().showMessage("Synthesis cancelled.", 4000)

    def _cleanup_synth_thread(self) -> None:
        self._set_synth_running(False)
        self._synth_worker = None
        self._synth_thread = None
        self._update_vram()

    def _set_synth_running(self, running: bool) -> None:
        self.synth_action.setEnabled(not running)
        self.cancel_action.setEnabled(running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)

    def _build_markers(self) -> list[SegmentMarker]:
        doc = self.editor.model()
        markers: list[SegmentMarker] = []
        for layout in self._pipeline.last_layout:
            span = doc.span_at(layout.char_start)
            markers.append(SegmentMarker(
                seg_id=layout.seg_id,
                start_s=layout.start_s,
                end_s=layout.end_s,
                color_idx=span.color_idx if span else 0,
                job_hash=layout.job_hash,
            ))
        return markers

    def _rerender_segment(self, seg_id: int) -> None:
        jobs = self._build_jobs()
        job = next((j for j in jobs if j.seg_id == seg_id), None)
        if job:
            self._pipeline.invalidate(job.hash)
            self.synthesize_all()

    # ----- preview ----------------------------------------------------------
    def _selection_text(self) -> str:
        start, end = self.editor.selected_range()
        doc_text = self.editor.model().text
        text = doc_text[start:end] if end > start else doc_text
        return text[:600]

    def _preview_segment(self, inflection) -> None:
        if self._current_profile() is None:
            QMessageBox.warning(self, "No voice selected", "Choose a voice profile first.")
            return
        text = self._selection_text()
        if not text.strip():
            return
        engine = inflection.engine or self._project.engine
        if engine == "hybrid":
            # Preview of a hybrid span = the finished transfer; render via pipeline.
            self._run_preview_job(SegmentJob(
                seg_id=-1, text=text, inflection=inflection.normalized(),
                voice_profile=self._current_profile(), engine="hybrid"))
            return
        self._run_preview_job(SegmentJob(
            seg_id=-1, text=text, inflection=inflection.normalized(),
            voice_profile=self._current_profile(), engine=engine))

    def _audition_performance(self, inflection) -> None:
        """Render just the Fish stage-1 performance for a hybrid span."""
        text = self._selection_text()
        if not text.strip():
            return
        hybrid_job = SegmentJob(
            seg_id=-1, text=text, inflection=inflection.normalized(),
            voice_profile=self._current_profile(), engine="hybrid")
        stage1, _, _ = self._pipeline._make_hybrid_stages(hybrid_job)
        self.statusBar().showMessage("Rendering performance (Fish)…")
        self._run_preview_job(stage1)

    def _run_preview_job(self, job: SegmentJob) -> None:
        if self._preview_thread is not None:
            return
        self.statusBar().showMessage("Rendering preview…")
        self._preview_thread = QThread(self)
        self._preview_worker = PreviewWorker(self._pipeline, job)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.failed.connect(self._on_preview_failed)
        for sig in (self._preview_worker.finished, self._preview_worker.failed):
            sig.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._cleanup_preview_thread)
        self._preview_thread.start()

    def _on_preview_done(self, audio: np.ndarray, sr: int) -> None:
        from ..audio.playback import AudioPlayer

        self.statusBar().showMessage("Preview ready.", 3000)
        self._preview_player = AudioPlayer(self._cfg.settings.output_device)
        self._preview_player.play(np.asarray(audio, dtype=np.float32), sr)

    def _on_preview_failed(self, message: str) -> None:
        from .error_dialog import show_error

        show_error(self, "Preview failed", message)

    def _cleanup_preview_thread(self) -> None:
        self._preview_worker = None
        self._preview_thread = None

    # ----- project I/O ------------------------------------------------------
    def new_project(self) -> None:
        self._project = Project(engine=self._cfg.settings.default_engine)
        self._project_path = None
        self.editor.set_model(self._project.document)
        self._refresh_profiles()
        self.engine_combo.setCurrentIndex(self._engine_index(self._project.engine))
        self.setWindowTitle("Inflect Studio — Untitled")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Inflect (*.inflect)")
        if not path:
            return
        try:
            self._project = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._project_path = Path(path)
        self.editor.set_model(self._project.document)
        self.engine_combo.setCurrentIndex(self._engine_index(self._project.engine))
        self._refresh_profiles()
        self.setWindowTitle(f"Inflect Studio — {self._project.name}")

    def save_project_action(self) -> None:
        if self._project_path is None:
            self.save_project_as()
        else:
            self._save_to(self._project_path)

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "", "Inflect (*.inflect)")
        if path:
            self._save_to(Path(path))

    def _save_to(self, path: Path) -> None:
        self._project.document = self.editor.model()
        self._project.name = path.stem
        try:
            saved = save_project(self._project, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._project_path = saved
        self.setWindowTitle(f"Inflect Studio — {self._project.name}")
        self.statusBar().showMessage(f"Saved {saved.name}", 3000)

    def export_audio(self) -> None:
        if self._mix.size == 0:
            QMessageBox.information(self, "Nothing to export", "Synthesize the audio first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", "", "WAV (*.wav);;MP3 (*.mp3)")
        if not path:
            return
        try:
            self._export_to(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(path).name}", 4000)

    def _export_to(self, path: Path) -> None:
        import soundfile as sf

        if path.suffix.lower() == ".mp3":
            from ..ingest.extract import encode_audio

            tmp = self._cfg.paths.tmp / "export.wav"
            sf.write(str(tmp), self._mix, self._mix_sr)
            encode_audio(tmp, path, ffmpeg_path=self._cfg.settings.ffmpeg_path)
        else:
            sf.write(str(path), self._mix, self._mix_sr)

    def open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        dialog = SettingsDialog(self._cfg, self)
        if dialog.exec():
            self._cfg.save_settings()
            self.statusBar().showMessage("Settings saved.", 3000)

    # ----- misc -------------------------------------------------------------
    def _engine_index(self, engine: str) -> int:
        for i, (_, name) in enumerate(ENGINE_MODES):
            if name == engine:
                return i
        return 0

    def _update_vram(self) -> None:
        total = ModelManager.vram_total_mb()
        used = ModelManager.vram_allocated_mb()
        if total <= 0:
            self.vram_label.setText("VRAM: n/a")
            self.vram_bar.setValue(0)
            return
        self.vram_label.setText(f"VRAM: {used:.0f}/{total:.0f} MB")
        self.vram_bar.setValue(int(100 * used / total))

    def warn_missing_ffmpeg(self) -> None:
        QMessageBox.warning(
            self, "ffmpeg not found",
            "ffmpeg was not found on your PATH. Importing video/audio and MP3 "
            "export will not work until ffmpeg is installed or its path is set "
            "in Settings.")

    def warn_no_cuda(self) -> None:
        self.statusBar().showMessage(
            "No CUDA GPU detected — IndexTTS-2 will be very slow; Chatterbox runs on CPU.",
            8000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        if self._synth_worker:
            self._synth_worker.cancel()
        if self._synth_thread and self._synth_thread.isRunning():
            self._synth_thread.quit()
            self._synth_thread.wait(3000)
        self._manager.unload_all()
        super().closeEvent(event)
