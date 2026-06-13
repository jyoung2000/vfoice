"""Pytest fixtures. Forces Qt offscreen so GUI smoke tests run headlessly."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole test session (skips if Qt unusable)."""
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # PySide6 missing or system libs absent
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app
