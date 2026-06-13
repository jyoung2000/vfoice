"""QApplication bootstrap: dark theme, environment checks, main window.

Heavy imports (PySide6, the main window) are deferred into :func:`main` so the
package can be imported by workers, engines and tests without a display.
"""

from __future__ import annotations

import shutil

from .config import get_config
from .logging_setup import get_logger, setup_logging

log = get_logger("app")


def ffmpeg_available(ffmpeg_path: str | None = None) -> bool:
    """True if the configured ffmpeg binary is resolvable on PATH."""
    cfg = get_config()
    candidate = ffmpeg_path or cfg.settings.ffmpeg_path
    return shutil.which(candidate) is not None


def cuda_available() -> bool:
    """True if a CUDA torch runtime is present."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _apply_dark_theme(app) -> None:
    """Apply a dark Fusion palette."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    p = QPalette()
    base = QColor(30, 31, 34)
    alt = QColor(38, 39, 43)
    text = QColor(220, 221, 222)
    accent = QColor(94, 129, 172)
    p.setColor(QPalette.Window, base)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, alt)
    p.setColor(QPalette.AlternateBase, base)
    p.setColor(QPalette.ToolTipBase, alt)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, alt)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor(255, 90, 90))
    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, QColor(20, 20, 20))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(p)
    _ = Qt  # keep import explicit for readers


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns the Qt exit code."""
    setup_logging()
    get_config()  # ensure data dirs exist

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on install
        print(
            "PySide6 is not installed. Install dependencies with:\n"
            "    pip install -r requirements.txt\n"
            f"(import error: {exc})"
        )
        return 1

    import sys

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Inflect Studio")
    app.setOrganizationName("InflectStudio")
    _apply_dark_theme(app)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    if not ffmpeg_available():
        window.warn_missing_ffmpeg()
    if not cuda_available():
        window.warn_no_cuda()

    log.info("Inflect Studio started (cuda=%s, ffmpeg=%s)",
             cuda_available(), ffmpeg_available())
    return app.exec()
