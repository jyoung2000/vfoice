"""Offscreen GUI smoke tests — construct widgets and exercise wiring.

These don't load any TTS model; they verify the widgets build and the
editor/inspector/timeline/main-window plumbing is internally consistent. The
whole module skips if PySide6 (or its system libs) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")


def _select(editor, start: int, end: int) -> None:
    from PySide6.QtGui import QTextCursor

    cursor = editor.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.KeepAnchor)
    editor.setTextCursor(cursor)


def test_editor_apply_and_remap(qapp):
    from inflect.document.spans import Inflection
    from inflect.ui.editor import InflectionTextEdit

    editor = InflectionTextEdit()
    editor.setPlainText("Hello world goodbye")
    _select(editor, 6, 11)  # "world"
    editor.apply_inflection_to_selection(Inflection(emo_text="angry"))
    spans = editor.model().sorted_spans()
    assert [(s.start, s.end) for s in spans] == [(6, 11)]
    assert len(editor.extraSelections()) == 1

    # type at the front; the span should shift right and stay attached to "world"
    cursor = editor.textCursor()
    cursor.setPosition(0)
    editor.setTextCursor(cursor)
    editor.insertPlainText("XX")
    spans = editor.model().sorted_spans()
    assert editor.model().text[spans[0].start:spans[0].end] == "world"

    _select(editor, spans[0].start, spans[0].end)
    editor.clear_inflection_on_selection()
    assert editor.model().spans == []


def test_inspector_roundtrip(qapp):
    from inflect.document.spans import EMOTIONS, Inflection
    from inflect.ui.inspector import InspectorPanel

    panel = InspectorPanel()
    vec = [0.0] * len(EMOTIONS)
    vec[EMOTIONS.index("angry")] = 0.7
    infl = Inflection(emotion_vector=vec, emo_text="tense", emo_alpha=0.6,
                      speed=1.2, pause_after_ms=250)
    panel.set_inflection(infl, has_selection=True)
    built = panel.build_inflection()
    assert built.emo_text == "tense"
    assert abs(built.emo_alpha - 0.6) < 0.02
    assert abs(built.speed - 1.2) < 0.02
    assert built.pause_after_ms == 250
    assert built.emotion_vector[EMOTIONS.index("angry")] == pytest.approx(0.7, abs=0.02)


def test_timeline_set_audio(qapp):
    from inflect.ui.timeline import SegmentMarker, TimelinePanel

    tl = TimelinePanel()
    sr = 24000
    audio = (0.3 * np.sin(np.linspace(0, 200, sr * 2))).astype(np.float32)
    tl.set_audio(audio, sr)
    tl.set_segments([
        SegmentMarker(0, 0.0, 1.0, 0, "h0"),
        SegmentMarker(1, 1.0, 2.0, 1, "h1"),
    ])
    assert tl._audio.size == audio.size  # stored for playback


def test_main_window_builds_and_segments(qapp, tmp_path, monkeypatch):
    # isolate app data so we don't touch the real library
    import inflect.config as config

    config.get_config.cache_clear()
    monkeypatch.setenv("INFLECT_HOME", str(tmp_path / "home"))

    from inflect.document.spans import Inflection
    from inflect.ui.main_window import MainWindow

    win = MainWindow()
    win.editor.setPlainText("One sentence here. Another sentence there.")
    _select(win.editor, 0, 3)
    win.editor.apply_inflection_to_selection(Inflection(emo_text="happy"))

    # a profile-less build still segments the text
    jobs = win._build_jobs()
    assert len(jobs) >= 2
    assert any(j.text.startswith("One") for j in jobs)

    # fake a layout and confirm markers map to spans
    from inflect.synth.pipeline import SegmentLayout

    win._pipeline.last_layout = [SegmentLayout(0, 0.0, 1.0, 0, 3, "h")]
    markers = win._build_markers()
    assert len(markers) == 1
    win.close()
    config.get_config.cache_clear()


def test_settings_dialog_writes_back(qapp, tmp_path, monkeypatch):
    import inflect.config as config

    config.get_config.cache_clear()
    monkeypatch.setenv("INFLECT_HOME", str(tmp_path / "home2"))
    cfg = config.get_config()

    from inflect.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(cfg)
    dialog._lufs.setValue(-14.0)
    dialog._crossfade.setValue(25)
    dialog._accept()
    assert cfg.settings.target_lufs == -14.0
    assert cfg.settings.crossfade_ms == 25
    config.get_config.cache_clear()
