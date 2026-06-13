"""The center Inflection editor: a QTextEdit driven by the Document model.

Span underlines are *derived* from the model and painted via extra selections —
the document text never carries baked-in formatting. Text edits remap span
offsets through ``Document.remap_on_edit`` so highlights track their words.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextEdit

from ..document.spans import Document, Inflection, InflectionSpan, NUM_SPAN_COLORS
from ..logging_setup import get_logger

log = get_logger("editor")

# Distinct, readable-on-dark underline colors, one per color_idx.
SPAN_COLORS = [
    "#e06c75", "#98c379", "#61afef", "#e5c07b",
    "#c678dd", "#56b6c2", "#d19a66", "#7f9f7f",
]


def span_color(idx: int) -> QColor:
    return QColor(SPAN_COLORS[idx % NUM_SPAN_COLORS])


class InflectionTextEdit(QTextEdit):
    """Editor whose spans mirror an :class:`inflect.document.spans.Document`."""

    selection_changed = Signal()      # caret/selection moved -> refresh inspector
    document_edited = Signal()        # text content changed
    apply_requested = Signal()        # context-menu "Apply inflection"
    clear_requested = Signal()        # context-menu "Clear inflection"
    pause_requested = Signal()        # context-menu "Insert pause…"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc = Document()
        self._color_counter = 0
        self._syncing = False
        self.setMouseTracking(True)
        self.setAcceptRichText(False)
        self.setPlaceholderText("Type or paste your script here, then highlight a "
                                "phrase and shape how it's spoken…")
        self.document().contentsChange.connect(self._on_contents_change)
        self.cursorPositionChanged.connect(self.selection_changed)
        self.selectionChanged.connect(self.selection_changed)

    # ----- model sync -------------------------------------------------------
    def model(self) -> Document:
        self._doc.text = self.toPlainText()
        return self._doc

    def set_model(self, doc: Document) -> None:
        self._doc = doc
        self._syncing = True
        self.setPlainText(doc.text)
        self._syncing = False
        self._refresh_underlines()

    def _on_contents_change(self, position: int, removed: int, added: int) -> None:
        if self._syncing:
            return
        self._doc.text = self.toPlainText()
        self._doc.remap_on_edit(position, removed, added)
        self._refresh_underlines()
        self.document_edited.emit()

    # ----- selection / inflection access -----------------------------------
    def selected_range(self) -> tuple[int, int]:
        cursor = self.textCursor()
        return cursor.selectionStart(), cursor.selectionEnd()

    def has_selection(self) -> bool:
        return self.textCursor().hasSelection()

    def current_inflection(self) -> Inflection:
        start, end = self.selected_range()
        pos = start if start != end else self.textCursor().position()
        return self._doc.inflection_at(min(pos, max(0, len(self._doc.text) - 1)))

    def current_span(self) -> InflectionSpan | None:
        start, _ = self.selected_range()
        return self._doc.span_at(start)

    # ----- mutation ---------------------------------------------------------
    def apply_inflection_to_selection(self, inflection: Inflection) -> None:
        start, end = self.selected_range()
        if start == end:
            return
        self._doc.text = self.toPlainText()
        self._doc.apply_inflection(start, end, inflection, self._next_color())
        self._refresh_underlines()
        self.document_edited.emit()

    def clear_inflection_on_selection(self) -> None:
        start, end = self.selected_range()
        if start == end:
            return
        self._doc.text = self.toPlainText()
        self._doc.clear_inflection(start, end)
        self._refresh_underlines()
        self.document_edited.emit()

    def set_default_inflection(self, inflection: Inflection) -> None:
        self._doc.default_inflection = inflection

    def _next_color(self) -> int:
        self._color_counter = (self._color_counter + 1) % NUM_SPAN_COLORS
        return self._color_counter

    # ----- rendering --------------------------------------------------------
    def _refresh_underlines(self) -> None:
        selections: list[QTextEdit.ExtraSelection] = []
        for span in self._doc.sorted_spans():
            sel = QTextEdit.ExtraSelection()
            fmt = QTextCharFormat()
            fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)
            color = span_color(span.color_idx)
            fmt.setUnderlineColor(color)
            tinted = QColor(color)
            tinted.setAlpha(38)
            fmt.setBackground(tinted)
            sel.format = fmt
            cursor = self.textCursor()
            cursor.setPosition(span.start)
            cursor.setPosition(min(span.end, len(self._doc.text)), QTextCursor.KeepAnchor)
            sel.cursor = cursor
            selections.append(sel)
        self.setExtraSelections(selections)

    # ----- interaction ------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt signature
        cursor = self.cursorForPosition(event.pos())
        span = self._doc.span_at(cursor.position())
        self.setToolTip(span.inflection.summary() if span else "")
        super().mouseMoveEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt signature
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        apply_act = menu.addAction("Apply inflection")
        clear_act = menu.addAction("Clear inflection")
        pause_act = menu.addAction("Insert pause…")
        apply_act.setEnabled(self.has_selection())
        clear_act.setEnabled(self.has_selection())
        chosen = menu.exec(event.globalPos())
        if chosen == apply_act:
            self.apply_requested.emit()
        elif chosen == clear_act:
            self.clear_requested.emit()
        elif chosen == pause_act:
            self.pause_requested.emit()
