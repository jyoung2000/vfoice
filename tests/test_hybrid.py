"""Phase 6: Fish tag translation + hybrid pipeline orchestration (fake engines)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inflect.document.segmenter import SegmentJob
from inflect.document.spans import EMOTIONS, Inflection
from inflect.models.engine_base import TTSEngine
from inflect.models.engine_fish import inflection_to_tags
from inflect.models.model_manager import ModelManager
from inflect.synth.pipeline import SynthesisPipeline


def _vec(name: str, value: float) -> list[float]:
    v = [0.0] * len(EMOTIONS)
    v[EMOTIONS.index(name)] = value
    return v


# --------------------------------------------------------------------------
# inflection_to_tags
# --------------------------------------------------------------------------
def test_tags_neutral_empty():
    assert inflection_to_tags(Inflection()) == ""


def test_tags_emo_text_verbatim():
    assert inflection_to_tags(Inflection(emo_text="whispering")) == "[whispering] "


def test_tags_intensity_tiers():
    assert "slightly annoyed" in inflection_to_tags(Inflection(emotion_vector=_vec("angry", 0.4)))
    assert "[angry] " == inflection_to_tags(Inflection(emotion_vector=_vec("angry", 0.6)))
    assert "furious" in inflection_to_tags(Inflection(emotion_vector=_vec("angry", 0.9)))


def test_tags_below_threshold_ignored():
    assert inflection_to_tags(Inflection(emotion_vector=_vec("happy", 0.2))) == ""


def test_tags_top_two_emotions():
    vec = _vec("angry", 0.9)
    vec[EMOTIONS.index("surprised")] = 0.5
    tag = inflection_to_tags(Inflection(emotion_vector=vec))
    assert "furious" in tag and "surprised" in tag


def test_tags_speed_appended():
    assert "fast" in inflection_to_tags(Inflection(emo_text="excited", speed=1.2))
    assert "slow" in inflection_to_tags(Inflection(emo_text="tired", speed=0.8))


def test_tags_emo_text_preferred_over_vector():
    tag = inflection_to_tags(Inflection(emotion_vector=_vec("angry", 0.9), emo_text="gentle"))
    assert tag == "[gentle] "


# --------------------------------------------------------------------------
# Hybrid pipeline orchestration (fake engines, no real models)
# --------------------------------------------------------------------------
class _FakeEngine(TTSEngine):
    def __init__(self, name, sr, load_order, synth_log, device="cpu"):
        super().__init__(device)
        self.name = name
        self._sr = sr
        self._load_order = load_order
        self._synth_log = synth_log

    def load(self):
        if not self._loaded:
            self._load_order.append(self.name)
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def sample_rate(self):
        return self._sr

    def synthesize(self, job):
        self._synth_log.append((self.name, job.text, job.inflection.emo_audio))
        return np.full(self._sr // 5, 0.1, dtype=np.float32)  # 0.2 s tone


def _make_pipeline(tmp_path):
    load_order: list[str] = []
    synth_log: list[tuple] = []
    manager = ModelManager(device="cpu")
    manager._engines = {
        "fish": _FakeEngine("fish", 44100, load_order, synth_log),
        "indextts2": _FakeEngine("indextts2", 22050, load_order, synth_log),
        "chatterbox": _FakeEngine("chatterbox", 24000, load_order, synth_log),
    }
    settings = SimpleNamespace(crossfade_ms=15, target_lufs=-16.0, true_peak_dbtp=-1.0)
    pipeline = SynthesisPipeline(manager, tmp_path / "cache", settings)
    return pipeline, load_order, synth_log


def test_hybrid_expands_and_transfers(tmp_path):
    pipeline, load_order, synth_log = _make_pipeline(tmp_path)
    profile = SimpleNamespace(id="p1", reference_path=str(tmp_path / "ref.wav"))
    hybrid_job = SegmentJob(0, "Be dramatic!", Inflection(emotion_vector=_vec("angry", 0.9)),
                            profile, "hybrid")

    mix, sr = pipeline.render([hybrid_job])
    assert mix.size > 0

    # Fish ran first (stage 1), IndexTTS-2 second (stage 2) -> one swap.
    assert load_order == ["fish", "indextts2"]

    # Stage 1 wrote a perf wav; stage 2 consumed it as emo_audio.
    perf = pipeline.perf_path_for(hybrid_job)
    assert perf.exists()
    idx_calls = [c for c in synth_log if c[0] == "indextts2"]
    assert idx_calls and idx_calls[0][2] == str(perf)  # emo_audio == perf wav


def test_mixed_document_minimizes_swaps(tmp_path):
    pipeline, load_order, _ = _make_pipeline(tmp_path)
    profile = SimpleNamespace(id="p1", reference_path=str(tmp_path / "ref.wav"))
    jobs = [
        SegmentJob(0, "Plain draft line.", Inflection(), profile, "chatterbox"),
        SegmentJob(1, "Dramatic hybrid line!", Inflection(emo_text="furious"),
                   profile, "hybrid"),
        SegmentJob(2, "Final quality line.", Inflection(), profile, "indextts2"),
    ]
    mix, sr = pipeline.render(jobs)
    assert mix.size > 0
    # fish (stage1) -> indextts2 (direct + stage2) -> chatterbox = 3 loads, 2 swaps
    assert load_order == ["fish", "indextts2", "chatterbox"]
    assert len(load_order) <= 3


def test_hybrid_caches_perf(tmp_path):
    pipeline, load_order, synth_log = _make_pipeline(tmp_path)
    profile = SimpleNamespace(id="p1", reference_path=str(tmp_path / "ref.wav"))
    job = SegmentJob(0, "Cache me.", Inflection(emo_text="sad"), profile, "hybrid")

    pipeline.render([job])
    fish_calls_first = len([c for c in synth_log if c[0] == "fish"])
    # second render: perf wav is cached, so Fish should not synth again
    pipeline.render([job])
    fish_calls_second = len([c for c in synth_log if c[0] == "fish"])
    assert fish_calls_second == fish_calls_first


def test_force_seg_id_rerenders_both_hybrid_stages(tmp_path):
    pipeline, _, synth_log = _make_pipeline(tmp_path)
    profile = SimpleNamespace(id="p1", reference_path=str(tmp_path / "ref.wav"))
    job = SegmentJob(0, "Re-render me.", Inflection(emo_text="sad"), profile, "hybrid")

    pipeline.render([job])
    fish_first = len([c for c in synth_log if c[0] == "fish"])
    # forcing the segment must re-run the Fish performance (stage 1), not just stage 2
    pipeline.render([job], force_seg_ids={0})
    fish_second = len([c for c in synth_log if c[0] == "fish"])
    assert fish_second == fish_first + 1


def test_identical_hash_segments_keep_distinct_pauses(tmp_path):
    """Two segments with identical audio-hash but different pauses must both
    keep their pause — proves the mix is keyed by position, not by hash."""
    pipeline, _, _ = _make_pipeline(tmp_path)
    profile = SimpleNamespace(id="p1", reference_path=str(tmp_path / "ref.wav"))
    # chatterbox fake outputs sr//5 = 4800 samples @ 24000 = 0.2 s
    a = SegmentJob(0, "hi", Inflection(pause_after_ms=0), profile, "chatterbox")
    b = SegmentJob(1, "hi", Inflection(pause_after_ms=500), profile, "chatterbox")
    assert a.hash == b.hash  # hash excludes pause -> they collide by hash

    mix, sr = pipeline.render([a, b])
    n_fade = int(0.015 * sr)
    expected = 4800 + (4800 + int(0.5 * sr)) - n_fade  # both pauses present
    assert abs(mix.size - expected) <= 1
