"""Speech-activity detection and clean-clip selection.

Wraps Silero VAD to find speech regions, then scores candidate 10–20 s windows
by speech continuity × RMS consistency × absence of clipping and returns the
top few. The scoring/selection math is pure (numpy only) so it is unit-tested
without the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_setup import get_logger
from ..audio.dsp import resample, spectral_flatness, to_mono_float32

log = get_logger("vad")

MIN_CLIP_S = 10.0
MAX_CLIP_S = 20.0
SILERO_SR = 16000  # Silero VAD operates at 16 kHz


@dataclass
class ClipCandidate:
    start_s: float
    end_s: float
    score: float
    speech_ratio: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class AnalysisResult:
    speech_ratio: float          # fraction of the whole audio that is speech
    flatness: float              # mean spectral flatness (music/noise hint)
    recommend_isolation: bool    # heuristic: run Demucs?
    candidates: list[ClipCandidate]
    regions: list[tuple[float, float]]


# --------------------------------------------------------------------------
# Silero model wrapper
# --------------------------------------------------------------------------
class SileroVAD:
    """Lazy wrapper around Silero VAD (pip ``silero-vad`` or torch.hub)."""

    def __init__(self) -> None:
        self._model = None
        self._get_ts = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model = load_silero_vad()
            self._get_ts = get_speech_timestamps
        except Exception:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
            )
            self._model = model
            self._get_ts = utils[0]  # get_speech_timestamps
        log.info("Silero VAD loaded")

    def detect_regions(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Return speech regions as ``(start_s, end_s)`` tuples."""
        self.load()
        import torch

        mono = resample(to_mono_float32(audio), sr, SILERO_SR)
        tensor = torch.from_numpy(mono)
        stamps = self._get_ts(tensor, self._model, sampling_rate=SILERO_SR)
        return [(s["start"] / SILERO_SR, s["end"] / SILERO_SR) for s in stamps]


# --------------------------------------------------------------------------
# Pure scoring / selection
# --------------------------------------------------------------------------
def speech_ratio(regions: list[tuple[float, float]], total_s: float) -> float:
    if total_s <= 0:
        return 0.0
    covered = sum(max(0.0, e - s) for s, e in regions)
    return min(1.0, covered / total_s)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def score_window(
    audio: np.ndarray,
    sr: int,
    regions: list[tuple[float, float]],
    start_s: float,
    end_s: float,
    frame_s: float = 0.1,
) -> tuple[float, float]:
    """Score a candidate window. Returns ``(score, speech_ratio_in_window)``.

    score = speech_continuity × rms_consistency × (1 − clipping_fraction)
    """
    audio = to_mono_float32(audio)
    i0 = max(0, int(start_s * sr))
    i1 = min(audio.shape[0], int(end_s * sr))
    window = audio[i0:i1]
    win_dur = (i1 - i0) / sr if sr else 0.0
    if window.size == 0 or win_dur <= 0:
        return 0.0, 0.0

    # 1) speech continuity within the window
    covered = sum(_overlap(start_s, end_s, s, e) for s, e in regions)
    continuity = min(1.0, covered / win_dur)

    # 2) RMS consistency across frames (1 - coefficient of variation)
    frame = max(1, int(frame_s * sr))
    n_frames = max(1, window.shape[0] // frame)
    rms_vals = np.array([
        np.sqrt(np.mean(window[k * frame:(k + 1) * frame] ** 2) + 1e-12)
        for k in range(n_frames)
    ])
    voiced = rms_vals[rms_vals > rms_vals.max() * 0.1] if rms_vals.size else rms_vals
    if voiced.size == 0 or voiced.mean() <= 1e-9:
        consistency = 0.0
    else:
        cov = float(np.std(voiced) / voiced.mean())
        consistency = float(np.clip(1.0 - cov, 0.0, 1.0))

    # 3) clipping penalty
    clip_frac = float(np.mean(np.abs(window) > 0.99))

    score = continuity * consistency * (1.0 - clip_frac)
    return float(score), continuity


def select_candidates(
    audio: np.ndarray,
    sr: int,
    regions: list[tuple[float, float]],
    min_s: float = MIN_CLIP_S,
    max_s: float = MAX_CLIP_S,
    top_k: int = 3,
    hop_s: float = 0.5,
) -> list[ClipCandidate]:
    """Return up to ``top_k`` non-overlapping best windows, best first."""
    audio = to_mono_float32(audio)
    total_s = audio.shape[0] / sr if sr else 0.0
    if total_s <= 0:
        return []

    win_len = min(max_s, total_s)
    if total_s >= min_s:
        win_len = max(win_len, min_s)

    starts: list[float] = []
    t = 0.0
    while t + win_len <= total_s + 1e-6:
        starts.append(round(t, 3))
        t += hop_s
    if not starts:  # audio shorter than a full window
        starts = [0.0]
        win_len = total_s

    scored = [
        ClipCandidate(s, min(s + win_len, total_s),
                      *score_window(audio, sr, regions, s, min(s + win_len, total_s)))
        for s in starts
    ]
    scored.sort(key=lambda c: c.score, reverse=True)

    chosen: list[ClipCandidate] = []
    for cand in scored:
        if len(chosen) >= top_k:
            break
        if all(_overlap(cand.start_s, cand.end_s, c.start_s, c.end_s) <= 0 for c in chosen):
            chosen.append(cand)
    return chosen


def analyze(
    audio: np.ndarray,
    sr: int,
    regions: list[tuple[float, float]] | None = None,
    vad: SileroVAD | None = None,
) -> AnalysisResult:
    """Full analysis: regions (via Silero if not supplied), ratio, candidates."""
    audio = to_mono_float32(audio)
    total_s = audio.shape[0] / sr if sr else 0.0
    if regions is None:
        vad = vad or SileroVAD()
        regions = vad.detect_regions(audio, sr)

    ratio = speech_ratio(regions, total_s)
    flatness = spectral_flatness(audio, sr)
    # Recommend isolation when speech is sparse or the signal looks musical/noisy.
    recommend = ratio < 0.6 or flatness > 0.3
    candidates = select_candidates(audio, sr, regions)
    return AnalysisResult(
        speech_ratio=ratio,
        flatness=flatness,
        recommend_isolation=recommend,
        candidates=candidates,
        regions=regions,
    )
