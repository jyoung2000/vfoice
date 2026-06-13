"""Rotating-file + console logging for Inflect Studio.

Logs per-segment synth timings and VRAM snapshots to ``logs/inflect.log`` so
the error dialog's "copy diagnostics" button has something useful to grab.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import get_config

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once; safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger("inflect")
    if _CONFIGURED:
        return root

    root.setLevel(level)
    cfg = get_config()
    fmt = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        cfg.paths.log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    _CONFIGURED = True
    root.info("Logging initialized -> %s", cfg.paths.log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"inflect.{name}")


def read_diagnostics(max_lines: int = 400) -> str:
    """Return the tail of the log file for the 'copy diagnostics' button."""
    cfg = get_config()
    try:
        lines = cfg.paths.log_file.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return "(no log file available)"
