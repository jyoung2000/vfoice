"""Tests for the pure VAD scoring / clip-selection logic (no model)."""

from __future__ import annotations

import numpy as np

from inflect.ingest.vad import (
    score_window,
    select_candidates,
    speech_ratio,
)

SR = 16000


def _tone(seconds: float, amp: float = 0.3, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_speech_ratio():
    assert speech_ratio([(0, 5), (10, 15)], 20.0) == 0.5
    assert speech_ratio([], 10.0) == 0.0
    assert speech_ratio([(0, 30)], 10.0) == 1.0  # clamped


def test_clean_window_scores_higher_than_clipped():
    clean = _tone(20)
    clipped = np.clip(_tone(20, amp=1.6), -1.0, 1.0)
    audio = np.concatenate([clean, clipped])
    regions = [(0.0, 40.0)]  # all speech: continuity equal, clipping decides
    clean_score, _ = score_window(audio, SR, regions, 0.0, 20.0)
    clip_score, _ = score_window(audio, SR, regions, 20.0, 40.0)
    assert clean_score > clip_score
    assert clean_score > 0.5


def test_steady_beats_gappy_on_consistency():
    steady = _tone(20, amp=0.3)
    # gappy: alternate 0.5 s tone / 0.5 s near-silence
    chunks = []
    for k in range(40):
        chunks.append(_tone(0.5, amp=0.3) if k % 2 == 0 else _tone(0.5, amp=0.01))
    gappy = np.concatenate(chunks)
    audio = np.concatenate([steady, gappy])
    regions = [(0.0, 40.0)]
    steady_score, _ = score_window(audio, SR, regions, 0.0, 20.0)
    gappy_score, _ = score_window(audio, SR, regions, 20.0, 40.0)
    assert steady_score > gappy_score


def test_select_picks_clean_region_first():
    clean = _tone(20, amp=0.3)
    clipped = np.clip(_tone(20, amp=1.8), -1.0, 1.0)
    audio = np.concatenate([clean, clipped])
    regions = [(0.0, 40.0)]
    cands = select_candidates(audio, SR, regions, min_s=10, max_s=20, top_k=3)
    assert cands
    best = cands[0]
    # best window should sit inside the clean first half
    assert best.start_s < 10.0
    assert 10.0 <= best.duration <= 20.0 + 1e-6


def test_candidates_are_non_overlapping():
    audio = _tone(60, amp=0.3)
    regions = [(0.0, 60.0)]
    cands = select_candidates(audio, SR, regions, min_s=10, max_s=20, top_k=3)
    cands_sorted = sorted(cands, key=lambda c: c.start_s)
    for a, b in zip(cands_sorted, cands_sorted[1:]):
        assert a.end_s <= b.start_s + 1e-6


def test_short_audio_single_candidate():
    audio = _tone(4, amp=0.3)  # shorter than min_s
    regions = [(0.0, 4.0)]
    cands = select_candidates(audio, SR, regions, min_s=10, max_s=20, top_k=3)
    assert len(cands) == 1
    assert cands[0].start_s == 0.0
    assert abs(cands[0].end_s - 4.0) < 0.1


def test_empty_audio():
    assert select_candidates(np.zeros(0, dtype=np.float32), SR, []) == []
