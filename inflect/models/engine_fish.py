"""Fish Speech / S2 engine (Phase 6, optional) — a "Final"-quality engine.

Fish has no numeric emotion vector; it responds to free-form inline tags. This
module therefore provides :func:`inflection_to_tags` (pure, unit-tested) and a
:class:`FishEngine` adapter built on the real resident-model API discovered in
the fish-speech repo:

    fish_speech.inference_engine.TTSInferenceEngine(
        llama_queue, decoder_model, precision, compile)
    engine.inference(ServeTTSRequest(text=..., references=[ServeReferenceAudio(...)],
                                     temperature=..., top_p=..., ...))
      -> yields InferenceResult chunks with audio.

Because fish-speech's loader-helper names drift between releases, model
construction is isolated in :meth:`_build_engine` and raises a clear
:class:`EngineError` if the installed version's API differs, so the rest of the
app is unaffected. Fish is research/personal-use licensed and VRAM-hungry — it
warns and is gated on the Settings license acknowledgement.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..audio.dsp import time_stretch, to_mono_float32
from ..config import get_config
from ..document.segmenter import SegmentJob
from ..document.spans import EMOTIONS, Inflection
from ..logging_setup import get_logger
from .engine_base import EngineError, EngineOOMError, Quality, TTSEngine, free_cuda, is_cuda_oom

log = get_logger("fish")

# fishaudio weights on Hugging Face; adjust to the current S2 release if needed.
FISH_HF_REPO = "fishaudio/openaudio-s1-mini"
EMOTION_THRESHOLD = 0.35

# Per-emotion adjective tiers: (slight <0.5, moderate 0.5-0.8, intense >=0.8).
_EMOTION_WORDS: dict[str, tuple[str, str, str]] = {
    "happy": ("slightly happy", "happy", "joyful and elated"),
    "angry": ("slightly annoyed", "angry", "furious"),
    "sad": ("a little sad", "sad", "deeply sorrowful"),
    "afraid": ("uneasy", "afraid", "terrified"),
    "disgusted": ("mildly distasteful", "disgusted", "revolted"),
    "melancholic": ("wistful", "melancholic", "despondent"),
    "surprised": ("mildly surprised", "surprised", "astonished"),
    "calm": ("relaxed", "calm", "serene and soothing"),
}


def _emotion_phrase(name: str, value: float) -> str:
    tier = 0 if value < 0.5 else (1 if value < 0.8 else 2)
    return _EMOTION_WORDS[name][tier]


def _speed_word(speed: float) -> str | None:
    if speed >= 1.3:
        return "very fast"
    if speed >= 1.1:
        return "fast"
    if speed <= 0.7:
        return "very slow"
    if speed <= 0.9:
        return "slow"
    return None


def inflection_to_tags(inflection: Inflection) -> str:
    """Translate an :class:`Inflection` into a Fish inline tag prefix.

    Returns something like ``"[furious and fast] "`` or ``""`` when neutral.
    A verbatim ``emo_text`` is preferred over the numeric vector.
    """
    parts: list[str] = []
    if inflection.emo_text:
        parts.append(inflection.emo_text.strip())
    elif inflection.emotion_vector is not None:
        ranked = sorted(
            ((EMOTIONS[i], v) for i, v in enumerate(inflection.emotion_vector)
             if v >= EMOTION_THRESHOLD),
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        parts.extend(_emotion_phrase(name, v) for name, v in ranked)

    speed = _speed_word(inflection.speed)
    if speed:
        parts.append(speed)

    if not parts:
        return ""
    return f"[{' and '.join(parts)}] "


class FishEngine(TTSEngine):
    name = "fish"
    display_name = "Fish S2 (Final)"
    quality = Quality.FINAL
    requires_cuda = True

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
        self._engine = None
        self._sr = 44100
        self._cfg = get_config()

    @property
    def _model_dir(self):
        return self._cfg.paths.fish

    def models_present(self) -> bool:
        return self._model_dir.exists() and any(self._model_dir.iterdir())

    def ensure_models(self, progress_cb: Callable[[str, int], None] | None = None) -> None:
        if self.models_present():
            return
        if progress_cb:
            progress_cb("Downloading Fish Speech weights (large, research-licensed)…", 0)
        try:
            import os

            from huggingface_hub import snapshot_download

            if self._cfg.settings.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self._cfg.settings.hf_endpoint
            snapshot_download(repo_id=FISH_HF_REPO, local_dir=str(self._model_dir))
        except Exception as exc:
            raise EngineError(f"Failed to download Fish weights: {exc}") from exc

    def load(self) -> None:
        if self._loaded:
            return
        if not self._cfg.settings.accept_fish_license:
            raise EngineError(
                "Fish Speech is research/personal-use licensed. Enable it in "
                "Settings (accept the Fish license) before using this engine.")
        if self.device != "cuda":
            log.warning("Fish on CPU is impractically slow; expect very long renders.")
        self.ensure_models()
        self._engine = self._build_engine()
        self._loaded = True
        log.info("Fish engine loaded (sr=%d)", self._sr)

    def _build_engine(self):
        """Construct the resident TTSInferenceEngine.

        Isolated so version drift in fish-speech's loader helpers surfaces as a
        single clear error instead of breaking the whole app.
        """
        try:
            import torch

            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

            precision = torch.float16 if (self._cfg.settings.use_fp16 and self.device == "cuda") \
                else torch.bfloat16
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(self._model_dir),
                device=self.device,
                precision=precision,
                compile=False,
            )
            decoder = load_decoder(
                config_name="modded_dac_vq",
                checkpoint_path=str(self._model_dir / "codec.pth"),
                device=self.device,
            )
            self._sr = int(getattr(decoder, "sample_rate", 44100))
            return TTSInferenceEngine(
                llama_queue=llama_queue, decoder_model=decoder,
                precision=precision, compile=False,
            )
        except ImportError as exc:
            raise EngineError(
                "fish-speech is not installed. Install it (pip install fish-speech) "
                "or follow github.com/fishaudio/fish-speech."
            ) from exc
        except Exception as exc:
            raise EngineError(
                "Could not build the Fish inference engine — your installed "
                f"fish-speech version's API may differ from the adapter: {exc}"
            ) from exc

    def unload(self) -> None:
        self._engine = None
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

        tagged_text = inflection_to_tags(job.inflection) + job.text
        references = self._build_references(job)
        try:
            audio = self._run_inference(tagged_text, references)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise EngineOOMError(
                    "GPU out of memory in Fish. Fish S2 is VRAM-hungry; close other "
                    "GPU apps or use shorter spans (24GB is recommended for S2)."
                ) from exc
            raise EngineError(f"Fish synthesis failed: {exc}") from exc
        finally:
            free_cuda()

        if abs(job.inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, job.inflection.speed)
        return audio

    def _build_references(self, job: SegmentJob):
        """Zero-shot clone reference for direct mode; None for the hybrid perf stage."""
        if job.engine_params.get("role") == "perf" or not job.voice_profile:
            return []  # stage-1 performance render uses Fish's own/default voice
        try:
            from fish_speech.utils.schema import ServeReferenceAudio

            ref_path = job.voice_profile.reference_path
            with open(ref_path, "rb") as f:
                return [ServeReferenceAudio(audio=f.read(), text="")]
        except Exception as exc:
            log.debug("Fish reference unavailable (%s); using zero-shot default voice", exc)
            return []

    def _run_inference(self, text: str, references) -> np.ndarray:
        from fish_speech.utils.schema import ServeTTSRequest

        req = ServeTTSRequest(
            text=text,
            references=references,
            format="wav",
            temperature=0.7,
            top_p=0.7,
            repetition_penalty=1.2,
        )
        chunks: list[np.ndarray] = []
        for result in self._engine.inference(req):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            # InferenceResult.audio is typically (sample_rate, np.ndarray)
            if isinstance(audio, tuple) and len(audio) == 2:
                self._sr = int(audio[0])
                chunks.append(to_mono_float32(np.asarray(audio[1])))
            else:
                chunks.append(to_mono_float32(np.asarray(audio)))
        if not chunks:
            raise EngineError("Fish produced no audio for this segment.")
        return np.concatenate(chunks).astype(np.float32)
