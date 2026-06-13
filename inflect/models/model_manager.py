"""Engine lifecycle manager: exactly one TTS engine resident at a time.

Switching engines unloads the previous one (freeing GPU memory) before loading
the next, which is mandatory on a 12 GB card. Engine classes are imported
lazily so this module — and the synthesis pipeline above it — can be imported
without torch/chatterbox installed.
"""

from __future__ import annotations

from typing import Callable

from ..config import get_config
from ..logging_setup import get_logger
from .engine_base import TTSEngine, free_cuda

log = get_logger("model_manager")

# name -> "module:ClassName" for lazy import
_ENGINE_SPECS: dict[str, tuple[str, str]] = {
    "chatterbox": ("inflect.models.engine_chatterbox", "ChatterboxEngine"),
    "indextts2": ("inflect.models.engine_indextts2", "IndexTTS2Engine"),
    "fish": ("inflect.models.engine_fish", "FishEngine"),
}


def available_engines() -> list[str]:
    return list(_ENGINE_SPECS)


def _construct(name: str, device: str) -> TTSEngine:
    import importlib

    if name not in _ENGINE_SPECS:
        raise KeyError(f"Unknown engine '{name}'")
    module_path, cls_name = _ENGINE_SPECS[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(device=device)


class ModelManager:
    """Owns engine instances and enforces single-residency on the GPU."""

    def __init__(self, device: str | None = None) -> None:
        self._device = device or get_config().device()
        self._engines: dict[str, TTSEngine] = {}
        self._current: str | None = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def current_engine_name(self) -> str | None:
        return self._current

    def get_engine(self, name: str) -> TTSEngine:
        """Return a loaded engine ``name``, unloading any other resident engine."""
        if self._current == name and self._engines.get(name, None) is not None \
                and self._engines[name].is_loaded:
            return self._engines[name]

        # Unload whatever is currently resident first (VRAM discipline).
        if self._current and self._current != name:
            self._unload(self._current)

        engine = self._engines.get(name)
        if engine is None:
            engine = _construct(name, self._device)
            self._engines[name] = engine

        log.info("Loading engine '%s' on %s", name, self._device)
        engine.load()
        self._current = name
        return engine

    def ensure_downloaded(self, name: str, progress_cb: Callable[[str, int], None] | None = None):
        """Download model weights for ``name`` if missing (no GPU residency)."""
        engine = self._engines.get(name) or _construct(name, self._device)
        self._engines[name] = engine
        ensure = getattr(engine, "ensure_models", None)
        if callable(ensure):
            ensure(progress_cb)

    def is_downloaded(self, name: str) -> bool:
        engine = self._engines.get(name) or _construct(name, self._device)
        self._engines[name] = engine
        check = getattr(engine, "models_present", None)
        return bool(check()) if callable(check) else True

    def _unload(self, name: str) -> None:
        engine = self._engines.get(name)
        if engine is not None and engine.is_loaded:
            log.info("Unloading engine '%s'", name)
            engine.unload()
        if self._current == name:
            self._current = None

    def unload_all(self) -> None:
        for name in list(self._engines):
            self._unload(name)
        self._current = None
        free_cuda()

    @staticmethod
    def vram_allocated_mb() -> float:
        """Currently allocated CUDA memory in MB (0.0 if no CUDA)."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def vram_total_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except Exception:
            pass
        return 0.0
