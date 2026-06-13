"""Low-level audio helpers shared by ingest, engines and assembly.

Pure-numpy implementations so the math is testable without torch/librosa, but
the higher-quality resampler is used when ``librosa`` or ``torchaudio`` is
available.
"""

from __future__ import annotations

import numpy as np


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Coerce any int/float, mono/stereo array to mono float32 in [-1, 1]."""
    a = np.asarray(audio)
    if a.dtype.kind in "iu":  # integer PCM -> scale to [-1, 1]
        max_val = float(np.iinfo(a.dtype).max)
        a = a.astype(np.float32) / max_val
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        # (frames, channels) or (channels, frames) -> average to mono
        axis = 1 if a.shape[0] >= a.shape[1] else 0
        a = a.mean(axis=axis)
    return np.ascontiguousarray(a.reshape(-1), dtype=np.float32)


def rms(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def peak(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float32)
    return float(np.max(np.abs(a))) if a.size else 0.0


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def linear_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono ``audio`` from ``src_sr`` to ``dst_sr``.

    Prefers librosa (band-limited) then torchaudio, falling back to numpy
    linear interpolation so the function always works.
    """
    audio = to_mono_float32(audio)
    if src_sr == dst_sr or audio.size == 0:
        return audio

    try:
        import librosa

        return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    except Exception:
        pass

    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(audio).unsqueeze(0)
        out = AF.resample(t, src_sr, dst_sr)
        return out.squeeze(0).numpy().astype(np.float32)
    except Exception:
        pass

    # numpy linear-interpolation fallback
    duration = audio.shape[0] / float(src_sr)
    n_out = int(round(duration * dst_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_t = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def silence(ms: int, sr: int) -> np.ndarray:
    n = int(round(max(0, ms) / 1000.0 * sr))
    return np.zeros(n, dtype=np.float32)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change tempo by ``rate`` (``>1`` faster) preserving pitch.

    Uses librosa's phase vocoder when available; otherwise falls back to naive
    resample-based stretch (which also shifts pitch — acceptable as a fallback).
    """
    audio = to_mono_float32(audio)
    if abs(rate - 1.0) < 1e-3 or audio.size == 0:
        return audio
    try:
        import librosa

        return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
    except Exception:
        # Resample-based fallback: re-time without a vocoder.
        idx = np.arange(0, audio.shape[0], rate)
        idx = idx[idx < audio.shape[0]]
        return np.interp(idx, np.arange(audio.shape[0]), audio).astype(np.float32)


def peak_envelope(audio: np.ndarray, n_buckets: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample to a min/max envelope for waveform display.

    Returns ``(mins, maxs)`` of length ``<= n_buckets``.
    """
    a = to_mono_float32(audio)
    if a.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if a.size <= n_buckets:
        return a.copy(), a.copy()
    bucket = a.size // n_buckets
    trimmed = a[: bucket * n_buckets].reshape(n_buckets, bucket)
    return trimmed.min(axis=1).astype(np.float32), trimmed.max(axis=1).astype(np.float32)


def spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """Mean spectral flatness in [0, 1]; high values suggest noise/music."""
    a = to_mono_float32(audio)
    if a.size < 512:
        return 0.0
    try:
        import librosa

        sf = librosa.feature.spectral_flatness(y=a)
        return float(np.mean(sf))
    except Exception:
        # crude FFT-based flatness (geometric mean / arithmetic mean of power)
        spec = np.abs(np.fft.rfft(a * np.hanning(a.shape[0]))) ** 2 + 1e-12
        gmean = np.exp(np.mean(np.log(spec)))
        amean = np.mean(spec)
        return float(gmean / amean)
