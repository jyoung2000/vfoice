"""Error dialog with a 'Copy diagnostics' button (recent log tail)."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from ..logging_setup import read_diagnostics


def show_error(parent: QWidget | None, title: str, message: str) -> None:
    """Show an error with the message plus a button to copy log diagnostics."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    box.setDetailedText(read_diagnostics())
    copy_btn = box.addButton("Copy diagnostics", QMessageBox.ActionRole)
    box.addButton(QMessageBox.Close)
    box.exec()
    if box.clickedButton() is copy_btn:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(f"{title}\n{message}\n\n{read_diagnostics()}")
