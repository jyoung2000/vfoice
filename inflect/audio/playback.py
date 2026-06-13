"""sounddevice-backed audio playback with a pollable position.

Kept Qt-agnostic: the audio callback updates ``current_frame`` on the audio
thread and the UI polls it (e.g. via a QTimer) to drive a playhead. Optional
``on_finished`` fires when playback reaches the end.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from ..logging_setup import get_logger

log = get_logger("playback")


class AudioPlayer:
    """Single-stream player for one mono float32 buffer at a time."""

    def __init__(self, device: str | int | None = None) -> None:
        self.device = device
        self._stream = None
        self._audio: np.ndarray = np.zeros(0, dtype=np.float32)
        self._sr = 24000
        self._pos = 0
        self._lock = threading.Lock()
        self._on_finished: Callable[[], None] | None = None
        self._finished_fired = False

    # ----- state ------------------------------------------------------------
    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def current_frame(self) -> int:
        with self._lock:
            return self._pos

    @property
    def current_time(self) -> float:
        return self.current_frame / self._sr if self._sr else 0.0

    @property
    def duration(self) -> float:
        return len(self._audio) / self._sr if self._sr else 0.0

    # ----- control ----------------------------------------------------------
    def play(
        self,
        audio: np.ndarray,
        sr: int,
        start_frame: int = 0,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """Begin playing ``audio`` from ``start_frame``."""
        import sounddevice as sd

        self.stop()
        self._audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        self._sr = int(sr)
        self._on_finished = on_finished
        self._finished_fired = False
        with self._lock:
            self._pos = max(0, min(start_frame, len(self._audio)))

        def _callback(outdata, frames, _time, status):  # noqa: ANN001 - sd signature
            if status:
                log.debug("playback status: %s", status)
            with self._lock:
                pos = self._pos
                end = min(pos + frames, len(self._audio))
                chunk = self._audio[pos:end]
                self._pos = end
            outdata[: len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0.0
                raise sd.CallbackStop()

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=_callback,
            finished_callback=self._handle_finished,
        )
        self._stream.start()

    def seek(self, frame: int) -> None:
        with self._lock:
            self._pos = max(0, min(frame, len(self._audio)))

    def seek_time(self, seconds: float) -> None:
        self.seek(int(seconds * self._sr))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _handle_finished(self) -> None:
        if not self._finished_fired and self._on_finished is not None:
            self._finished_fired = True
            self._on_finished()
