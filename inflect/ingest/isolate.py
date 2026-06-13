"""Optional vocal isolation via Demucs (htdemucs).

Used when the source has music/noise. Demucs and a TTS engine must never be
GPU-resident at the same time, so :meth:`DemucsIsolator.isolate` loads, runs
and unloads within the call.
"""

from __future__ import annotations

import numpy as np

from ..audio.dsp import resample, to_mono_float32
from ..logging_setup import get_logger
from ..models.engine_base import free_cuda

log = get_logger("isolate")

DEMUCS_MODEL = "htdemucs"


class DemucsIsolator:
    """Run Demucs to keep only the ``vocals`` stem."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from demucs.pretrained import get_model

        self._model = get_model(DEMUCS_MODEL)
        self._model.to(self.device)
        self._model.eval()
        log.info("Demucs %s loaded on %s", DEMUCS_MODEL, self.device)

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
        free_cuda()
        log.info("Demucs unloaded")

    @property
    def model_sr(self) -> int:
        return int(self._model.samplerate) if self._model is not None else 44100

    def isolate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Return ``(vocals_mono_float32, model_sr)``. Always unloads afterwards."""
        import torch
        from demucs.apply import apply_model

        try:
            self.load()
            model_sr = self.model_sr
            mono = resample(to_mono_float32(audio), sr, model_sr)
            # Demucs expects (batch, channels, samples); duplicate mono -> stereo.
            wav = torch.from_numpy(mono).unsqueeze(0).repeat(2, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                sources = apply_model(self._model, wav, device=self.device, progress=False)[0]
            idx = self._model.sources.index("vocals")
            vocals = sources[idx].mean(dim=0).cpu().numpy().astype(np.float32)
            return vocals, model_sr
        finally:
            self.unload()
