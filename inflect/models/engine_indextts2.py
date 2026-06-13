"""IndexTTS-2 engine — the primary "Final" quality tier.

Wraps the real ``indextts.infer_v2.IndexTTS2`` API (verified against
github.com/index-tts/index-tts):

    IndexTTS2(cfg_path, model_dir, use_fp16, use_cuda_kernel, use_deepspeed)
    tts.infer(spk_audio_prompt, text, output_path, emo_audio_prompt=None,
              emo_alpha=1.0, emo_vector=None, use_emo_text=False,
              emo_text=None, use_random=False, verbose=True)

``emo_vector`` is the 8-element ``[happy, angry, sad, afraid, disgusted,
melancholic, surprised, calm]`` list — the same order as our ``EMOTIONS``.
``infer`` writes a wav to ``output_path``; we read it back to a numpy array.
Duration control is not yet enabled upstream, so speed is a post time-stretch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..audio.dsp import time_stretch, to_mono_float32
from ..config import get_config
from ..document.segmenter import SegmentJob
from ..document.spans import Inflection
from ..logging_setup import get_logger
from .engine_base import EngineError, EngineOOMError, Quality, TTSEngine, free_cuda, is_cuda_oom

log = get_logger("indextts2")

HF_REPO = "IndexTeam/IndexTTS-2"


def map_inflection(inflection: Inflection) -> dict:
    """Translate an :class:`Inflection` into IndexTTS-2 infer kwargs.

    Precedence for the emotion source: emotion-reference audio (used by the
    Phase-6 hybrid stage) > numeric vector (deterministic) > emo-text (T2E).
    """
    kwargs: dict = {
        "emo_audio_prompt": None,
        "emo_vector": None,
        "use_emo_text": False,
        "emo_text": None,
        "emo_alpha": float(inflection.emo_alpha),
    }
    vec = inflection.emotion_vector
    has_vec = vec is not None and any(v > 0 for v in vec)

    if inflection.emo_audio:
        kwargs["emo_audio_prompt"] = inflection.emo_audio
    elif has_vec:
        kwargs["emo_vector"] = [float(v) for v in vec]
        if inflection.emo_text:
            log.info("seg has both emo_vector and emo_text; preferring the vector "
                     "(deterministic), ignoring emo_text")
    elif inflection.emo_text:
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = inflection.emo_text
    return kwargs


class IndexTTS2Engine(TTSEngine):
    name = "indextts2"
    display_name = "IndexTTS-2 (Final)"
    quality = Quality.FINAL
    requires_cuda = True  # runs on CPU but extremely slowly

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
        self._tts = None
        self._sr = 22050  # updated from the first rendered wav
        self._cfg = get_config()

    @property
    def _model_dir(self):
        return self._cfg.paths.indextts2

    def models_present(self) -> bool:
        return (self._model_dir / "config.yaml").exists()

    def ensure_models(self, progress_cb: Callable[[str, int], None] | None = None) -> None:
        if self.models_present():
            return
        if progress_cb:
            progress_cb("Downloading IndexTTS-2 weights (first run, several GB)…", 0)
        try:
            import os

            from huggingface_hub import snapshot_download

            if self._cfg.settings.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self._cfg.settings.hf_endpoint
            snapshot_download(repo_id=HF_REPO, local_dir=str(self._model_dir))
        except Exception as exc:
            raise EngineError(f"Failed to download IndexTTS-2 weights: {exc}") from exc
        if not self.models_present():
            raise EngineError(
                f"IndexTTS-2 weights incomplete in {self._model_dir} (config.yaml missing)."
            )
        if progress_cb:
            progress_cb("IndexTTS-2 weights ready.", 100)

    def load(self) -> None:
        if self._loaded:
            return
        self.ensure_models()
        use_cuda = self.device == "cuda"
        try:
            from indextts.infer_v2 import IndexTTS2

            self._tts = IndexTTS2(
                cfg_path=str(self._model_dir / "config.yaml"),
                model_dir=str(self._model_dir),
                use_fp16=bool(self._cfg.settings.use_fp16 and use_cuda),
                use_cuda_kernel=bool(self._cfg.settings.use_cuda_kernel and use_cuda),
            )
            self._loaded = True
            log.info("IndexTTS-2 loaded (fp16=%s, cuda_kernel=%s)",
                     self._cfg.settings.use_fp16 and use_cuda,
                     self._cfg.settings.use_cuda_kernel and use_cuda)
        except ImportError as exc:
            raise EngineError(
                "IndexTTS-2 ('indextts') is not installed. Install it from "
                "github.com/index-tts/index-tts (see requirements.txt)."
            ) from exc
        except Exception as exc:
            raise EngineError(f"Failed to load IndexTTS-2: {exc}") from exc

    def unload(self) -> None:
        self._tts = None
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
        if not job.voice_profile or not getattr(job.voice_profile, "reference_path", None):
            raise EngineError("IndexTTS-2 needs a voice profile reference clip.")

        import soundfile as sf

        ref = job.voice_profile.reference_path
        out_path = self._cfg.paths.tmp / f"idx_{job.hash}.wav"
        kwargs = map_inflection(job.inflection)
        try:
            self._tts.infer(
                spk_audio_prompt=ref,
                text=job.text,
                output_path=str(out_path),
                verbose=False,
                **kwargs,
            )
            audio, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
            self._sr = int(sr)
            audio = to_mono_float32(audio)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise EngineOOMError(
                    "GPU out of memory in IndexTTS-2. Close other GPU apps, enable "
                    "fp16, or split the text into shorter spans."
                ) from exc
            raise EngineError(f"IndexTTS-2 synthesis failed: {exc}") from exc
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            free_cuda()

        # Duration control isn't enabled upstream; apply speed as a time-stretch.
        if abs(job.inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, job.inflection.speed)
        return audio
