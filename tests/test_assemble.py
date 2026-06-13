"""Tests for crossfade concatenation, pause insertion and loudness normalize."""

from __future__ import annotations

import numpy as np

from inflect.audio.dsp import db_to_linear
from inflect.synth.assemble import (
    RenderedSegment,
    assemble,
    equal_power_crossfade,
    normalize_loudness,
)

SR = 24000


def test_crossfade_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(800, dtype=np.float32)
    out = equal_power_crossfade(a, b, 100)
    assert out.shape[0] == 1000 + 800 - 100


def test_crossfade_with_empty():
    a = np.ones(10, dtype=np.float32)
    assert np.array_equal(equal_power_crossfade(a, np.zeros(0), 5), a)
    assert np.array_equal(equal_power_crossfade(np.zeros(0), a, 5), a)


def test_crossfade_is_equal_power():
    n = 200
    ones = np.ones(n + 50, dtype=np.float32)
    zeros = np.zeros(n + 50, dtype=np.float32)
    # both overlap regions sit at [50:50+n] (50-sample lead from prev[:-m])
    fade_out = equal_power_crossfade(ones, zeros, n)[50 : 50 + n]
    fade_in = equal_power_crossfade(zeros, ones, n)[50 : 50 + n]
    power = fade_out**2 + fade_in**2
    assert np.allclose(power, 1.0, atol=1e-4)


def test_crossfade_clamps_to_short_buffer():
    a = np.ones(30, dtype=np.float32)
    b = np.ones(20, dtype=np.float32)
    out = equal_power_crossfade(a, b, 100)  # n > both
    assert out.shape[0] == 30 + 20 - 20  # m clamped to 20


def test_assemble_crossfade_reduces_length():
    seg = RenderedSegment(np.ones(SR, dtype=np.float32), SR, 0)
    out = assemble([seg, seg], SR, crossfade_ms=15, normalize=False)
    n_fade = int(0.015 * SR)
    assert out.shape[0] == 2 * SR - n_fade


def test_assemble_inserts_pause():
    seg1 = RenderedSegment(np.ones(SR, dtype=np.float32), SR, pause_after_ms=500)
    seg2 = RenderedSegment(np.ones(SR, dtype=np.float32), SR, 0)
    out = assemble([seg1, seg2], SR, crossfade_ms=15, normalize=False)
    n_fade = int(0.015 * SR)
    pause = int(0.5 * SR)
    # block1 = SR + pause, block2 = SR, joined with one crossfade
    assert out.shape[0] == (SR + pause) + SR - n_fade


def test_assemble_resamples_to_target():
    seg = RenderedSegment(np.ones(12000, dtype=np.float32), 12000, 0)  # 1 s @ 12 kHz
    out = assemble([seg], SR, normalize=False)
    assert abs(out.shape[0] - SR) <= 2  # ~1 s @ 24 kHz


def test_assemble_empty():
    assert assemble([], SR).shape[0] == 0


def test_assemble_silent_segment_only_pause():
    silent = RenderedSegment(np.zeros(0, dtype=np.float32), SR, pause_after_ms=300)
    out = assemble([silent], SR, normalize=False)
    assert out.shape[0] == int(0.3 * SR)
    assert np.allclose(out, 0.0)


def test_normalize_respects_true_peak():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * 3) * 0.9).astype(np.float32)
    out = normalize_loudness(audio, SR, target_lufs=-16.0, true_peak_dbtp=-1.0)
    limit = db_to_linear(-1.0)
    assert np.max(np.abs(out)) <= limit + 1e-4


def test_normalize_hits_target_lufs():
    import pyloudnorm as pyln

    t = np.arange(SR * 3) / SR
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # quiet tone
    out = normalize_loudness(audio, SR, target_lufs=-16.0, true_peak_dbtp=-1.0)
    measured = pyln.Meter(SR).integrated_loudness(out.astype(np.float64))
    # within ~1.5 LU (true-peak scaling can pull it slightly under target)
    assert -18.0 <= measured <= -14.5


def test_normalize_empty():
    assert normalize_loudness(np.zeros(0, dtype=np.float32), SR).shape[0] == 0
