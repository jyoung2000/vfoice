"""Chatterbox engine — the fast "Draft" preview tier.

Maps our :class:`Inflection` onto Chatterbox's ``exaggeration`` / ``cfg_weight``
controls (it has no numeric emotion vector). Speed is applied as a post
time-stretch since Chatterbox has no duration control.
"""

from __future__ import annotations

import numpy as np

from ..audio.dsp import time_stretch, to_mono_float32
from ..document.segmenter import SegmentJob
from ..document.spans import Inflection
from ..logging_setup import get_logger
from .engine_base import EngineError, EngineOOMError, Quality, TTSEngine, free_cuda, is_cuda_oom

log = get_logger("chatterbox")


def derive_chatterbox_params(inflection: Inflection, params: dict) -> tuple[float, float]:
    """Return ``(exaggeration, cfg_weight)`` from an inflection + overrides."""
    vec = inflection.emotion_vector
    if "exaggeration" in params:
        exaggeration = float(params["exaggeration"])
    elif vec is not None and any(v > 0 for v in vec):
        dominant = max(vec)
        exaggeration = 0.5 + 0.5 * dominant * inflection.emo_alpha
    elif inflection.emo_text:
        exaggeration = 0.6  # some extra expressiveness when a description is given
    else:
        exaggeration = 0.5
    exaggeration = float(np.clip(exaggeration, 0.25, 1.5))
    cfg_weight = float(params.get("cfg_weight", 0.5))
    return exaggeration, cfg_weight


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    display_name = "Chatterbox (Draft)"
    quality = Quality.DRAFT
    requires_cuda = False  # usable (slowly) on CPU

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
        self._model = None
        self._sr = 24000

    def models_present(self) -> bool:
        # Chatterbox pulls its own weights via from_pretrained on first load.
        return True

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            self._sr = int(getattr(self._model, "sr", 24000))
            self._loaded = True
            log.info("Chatterbox loaded (sr=%d, device=%s)", self._sr, self.device)
        except ImportError as exc:
            raise EngineError(
                "chatterbox-tts is not installed. Run: pip install chatterbox-tts"
            ) from exc
        except Exception as exc:
            raise EngineError(f"Failed to load Chatterbox: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        free_cuda()

    @property
    def sample_rate(self) -> int:
        return self._sr

    def synthesize(self, job: SegmentJob) -> np.ndarray:
        if not self._loaded:
            self.load()
        if job.is_silent:
            return np.zeros(0, dtype=np.float32)
        ref = job.voice_profile.reference_path if job.voice_profile else None
        exaggeration, cfg_weight = derive_chatterbox_params(job.inflection, job.engine_params)
        try:
            kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
            if ref:
                kwargs["audio_prompt_path"] = ref
            wav = self._model.generate(job.text, **kwargs)
            audio = to_mono_float32(_to_numpy(wav))
        except Exception as exc:
            if is_cuda_oom(exc):
                raise EngineOOMError(
                    "GPU ran out of memory. Close other GPU apps or shorten the segment."
                ) from exc
            raise EngineError(f"Chatterbox synthesis failed: {exc}") from exc
        finally:
            free_cuda()

        if abs(job.inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, job.inflection.speed)
        return audio


def _to_numpy(wav) -> np.ndarray:
    """Coerce a torch tensor / array / (sr, wav) tuple to a 1-D numpy float array."""
    if isinstance(wav, tuple):  # some versions return (sample_rate, audio)
        wav = wav[-1]
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)
