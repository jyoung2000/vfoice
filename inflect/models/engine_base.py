"""Abstract base for TTS engines.

Every engine (Chatterbox, IndexTTS-2, Fish) implements this interface. The
synthesis pipeline only ever talks to a ``TTSEngine`` — it never imports a
concrete engine — so engines can be swapped without touching anything above
this layer (a hard requirement from the project spec).
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from ..document.segmenter import SegmentJob


class Quality(str, Enum):
    """User-facing quality tier. Draft = fast preview, Final = best."""

    DRAFT = "Draft"
    FINAL = "Final"


class EngineError(RuntimeError):
    """Engine-level failure surfaced to the UI (e.g. OOM, missing weights)."""


class EngineOOMError(EngineError):
    """Raised on CUDA out-of-memory so the UI can suggest remedies."""


class TTSEngine(ABC):
    """Contract for a zero-shot voice-cloning TTS engine."""

    #: Stable identifier used in segment hashes and engine selection.
    name: str = "base"
    #: Human-facing label.
    display_name: str = "Base Engine"
    #: Draft (fast) or Final (best) tier.
    quality: Quality = Quality.FINAL
    #: Whether the engine needs CUDA to be usable at acceptable speed.
    requires_cuda: bool = False

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._loaded = False

    # ----- lifecycle --------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights onto ``self.device``. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release weights and free GPU memory. Idempotent."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this engine in Hz."""

    @abstractmethod
    def synthesize(self, job: SegmentJob) -> np.ndarray:
        """Render ``job`` to a mono float32 waveform in ``[-1, 1]``.

        Implementations must call :func:`free_cuda` after each call and raise
        :class:`EngineOOMError` (not a bare RuntimeError) on CUDA OOM so the UI
        can offer remedies.
        """

    # ----- context-manager sugar -------------------------------------------
    def __enter__(self) -> "TTSEngine":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()


def free_cuda() -> None:
    """Best-effort GPU memory reclamation. Safe to call without CUDA/torch."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # torch missing or CUDA unavailable — nothing to free
        pass


def is_cuda_oom(exc: BaseException) -> bool:
    """Heuristic: does this exception look like a CUDA out-of-memory error?"""
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda oom" in msg
        or ("alloc" in msg and "cuda" in msg)
    )
