"""Assemble per-segment renders into one seamless, loudness-normalized track.

Each rendered segment is resampled to a common rate, its trailing pause is
appended, and consecutive blocks are joined with a 15 ms equal-power crossfade.
The final mix is normalized to -16 LUFS and limited to -1 dBTP.

The math here is pure numpy (+ optional pyloudnorm) and is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio.dsp import db_to_linear, peak, resample, silence
from ..logging_setup import get_logger

log = get_logger("assemble")


@dataclass
class RenderedSegment:
    audio: np.ndarray       # mono float32 at ``sr``
    sr: int
    pause_after_ms: int = 0


def equal_power_crossfade(prev: np.ndarray, nxt: np.ndarray, n: int) -> np.ndarray:
    """Join two buffers with an ``n``-sample equal-power (cos/sin) crossfade."""
    prev = np.asarray(prev, dtype=np.float32)
    nxt = np.asarray(nxt, dtype=np.float32)
    if prev.size == 0:
        return nxt.copy()
    if nxt.size == 0:
        return prev.copy()
    m = int(min(n, prev.size, nxt.size))
    if m <= 0:
        return np.concatenate([prev, nxt])
    t = (np.arange(m, dtype=np.float32) + 0.5) / m
    fade_out = np.cos(t * np.pi / 2.0)   # 1 -> 0
    fade_in = np.sin(t * np.pi / 2.0)    # 0 -> 1  (fade_out^2 + fade_in^2 == 1)
    overlap = prev[-m:] * fade_out + nxt[:m] * fade_in
    return np.concatenate([prev[:-m], overlap, nxt[m:]]).astype(np.float32)


def normalize_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    true_peak_dbtp: float = -1.0,
) -> np.ndarray:
    """Normalize to ``target_lufs`` and scale below ``true_peak_dbtp``."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    out = audio
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(out.astype(np.float64))
        if np.isfinite(loudness):
            out = pyln.normalize.loudness(out.astype(np.float64), loudness, target_lufs)
            out = out.astype(np.float32)
    except Exception as exc:  # too short for an integrated measurement, etc.
        log.debug("LUFS normalize skipped: %s", exc)

    # True-peak limit: scale down (no hard clipping) if we exceed the ceiling.
    limit = db_to_linear(true_peak_dbtp)
    pk = peak(out)
    if pk > limit and pk > 0:
        out = out * (limit / pk)
    return np.ascontiguousarray(out, dtype=np.float32)


def assemble(
    segments: list[RenderedSegment],
    target_sr: int,
    crossfade_ms: int = 15,
    target_lufs: float = -16.0,
    true_peak_dbtp: float = -1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Concatenate ``segments`` into one mono float32 track at ``target_sr``."""
    if not segments:
        return np.zeros(0, dtype=np.float32)

    n_fade = max(0, int(crossfade_ms / 1000.0 * target_sr))
    blocks: list[np.ndarray] = []
    for seg in segments:
        audio = resample(seg.audio, seg.sr, target_sr) if seg.audio.size else \
            np.zeros(0, dtype=np.float32)
        pause = silence(seg.pause_after_ms, target_sr)
        block = np.concatenate([audio, pause]) if pause.size else audio
        if block.size:
            blocks.append(block.astype(np.float32))

    if not blocks:
        return np.zeros(0, dtype=np.float32)

    mix = blocks[0]
    for block in blocks[1:]:
        mix = equal_power_crossfade(mix, block, n_fade)

    if normalize:
        mix = normalize_loudness(mix, target_sr, target_lufs, true_peak_dbtp)
    return mix.astype(np.float32)
