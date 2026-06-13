"""Left dock: the voice library — profile cards, import, preview, manage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..audio.playback import AudioPlayer
from ..config import get_config
from ..ingest.ingest_dialog import IngestDialog
from ..ingest.profile import VoiceLibrary, VoiceProfile
from ..logging_setup import get_logger

log = get_logger("voice_library")


class _ProfileCard(QWidget):
    """One row: name + duration on the left, a play-preview button on the right."""

    preview_requested = Signal(str)

    def __init__(self, profile: VoiceProfile) -> None:
        super().__init__()
        self.profile_id = profile.id
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        text = QVBoxLayout()
        name = QLabel(profile.name)
        name.setStyleSheet("font-weight: 600;")
        meta = QLabel(f"{profile.duration:.1f}s · {profile.source_filename or 'imported'}")
        meta.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        text.addWidget(name)
        text.addWidget(meta)
        row.addLayout(text, 1)
        play = QPushButton("▶")
        play.setFixedWidth(34)
        play.setToolTip("Preview reference clip")
        play.clicked.connect(lambda: self.preview_requested.emit(self.profile_id))
        row.addWidget(play)


class VoiceLibraryPanel(QWidget):
    """Self-contained panel; the main window wraps it in a QDockWidget."""

    profile_selected = Signal(object)  # str | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        cfg = get_config()
        self._library = VoiceLibrary.open(cfg.paths.voices)
        self._player = AudioPlayer(cfg.settings.output_device)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        import_btn = QPushButton("＋ Import from video / audio…")
        import_btn.clicked.connect(self.import_profile)
        layout.addWidget(import_btn)

        self._list = QListWidget()
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, 1)

        self.refresh()

    # ----- public API -------------------------------------------------------
    @property
    def library(self) -> VoiceLibrary:
        return self._library

    def current_profile_id(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def current_profile(self) -> VoiceProfile | None:
        pid = self.current_profile_id()
        return self._library.get(pid) if pid else None

    def select_profile(self, profile_id: str) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.UserRole) == profile_id:
                self._list.setCurrentRow(i)
                return

    def refresh(self) -> None:
        current = self.current_profile_id()
        self._list.clear()
        for profile in self._library.list():
            item = QListWidgetItem(self._list)
            item.setData(Qt.UserRole, profile.id)
            card = _ProfileCard(profile)
            card.preview_requested.connect(self._preview)
            item.setSizeHint(card.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, card)
        if current:
            self.select_profile(current)

    # ----- actions ----------------------------------------------------------
    def import_profile(self, source: str | Path | None = None) -> None:
        dialog = IngestDialog(self._library, parent=self, source=source)
        if dialog.exec() and dialog.created_profile:
            self.refresh()
            self.select_profile(dialog.created_profile.id)

    def _preview(self, profile_id: str) -> None:
        import soundfile as sf

        profile = self._library.get(profile_id)
        if not profile or not profile.exists:
            QMessageBox.warning(self, "Missing audio", "Reference clip not found.")
            return
        if self._player.is_playing:
            self._player.stop()
            return
        audio, sr = sf.read(profile.reference_path, dtype="float32", always_2d=False)
        self._player.play(np.asarray(audio, dtype=np.float32).reshape(-1), sr)

    def _on_selection_changed(self) -> None:
        self.profile_selected.emit(self.current_profile_id())

    def _show_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if not item:
            return
        profile_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        rename_act = menu.addAction("Rename…")
        delete_act = menu.addAction("Delete")
        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen == rename_act:
            self._rename(profile_id)
        elif chosen == delete_act:
            self._delete(profile_id)

    def _rename(self, profile_id: str) -> None:
        profile = self._library.get(profile_id)
        if not profile:
            return
        name, ok = QInputDialog.getText(self, "Rename voice", "Name:", text=profile.name)
        if ok and name.strip():
            self._library.rename(profile_id, name.strip())
            self.refresh()

    def _delete(self, profile_id: str) -> None:
        profile = self._library.get(profile_id)
        if not profile:
            return
        if QMessageBox.question(
            self, "Delete voice", f"Delete “{profile.name}”? This cannot be undone."
        ) == QMessageBox.Yes:
            self._library.delete(profile_id)
            self.refresh()
