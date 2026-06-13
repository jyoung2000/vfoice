"""Tests for the pure inflection->engine-parameter mapping (no models loaded)."""

from __future__ import annotations

from inflect.document.spans import EMOTIONS, Inflection
from inflect.models.engine_base import EngineOOMError, is_cuda_oom
from inflect.models.engine_chatterbox import derive_chatterbox_params
from inflect.models.engine_indextts2 import map_inflection


def _vec(name: str, value: float) -> list[float]:
    v = [0.0] * len(EMOTIONS)
    v[EMOTIONS.index(name)] = value
    return v


# --------------------------------------------------------------------------
# Chatterbox
# --------------------------------------------------------------------------
def test_chatterbox_neutral_default():
    exag, cfg = derive_chatterbox_params(Inflection(), {})
    assert exag == 0.5
    assert cfg == 0.5


def test_chatterbox_scales_with_emotion_and_alpha():
    strong = derive_chatterbox_params(Inflection(emotion_vector=_vec("angry", 1.0),
                                                  emo_alpha=1.0), {})[0]
    weak = derive_chatterbox_params(Inflection(emotion_vector=_vec("angry", 0.3),
                                               emo_alpha=0.5), {})[0]
    assert strong > weak > 0.5


def test_chatterbox_emo_text_bumps_expressiveness():
    exag, _ = derive_chatterbox_params(Inflection(emo_text="whispering"), {})
    assert exag == 0.6


def test_chatterbox_params_override():
    exag, cfg = derive_chatterbox_params(
        Inflection(emotion_vector=_vec("happy", 1.0)),
        {"exaggeration": 0.42, "cfg_weight": 0.7},
    )
    assert exag == 0.42
    assert cfg == 0.7


# --------------------------------------------------------------------------
# IndexTTS-2
# --------------------------------------------------------------------------
def test_indextts2_vector_path():
    kw = map_inflection(Inflection(emotion_vector=_vec("surprised", 0.8), emo_alpha=0.9))
    assert kw["emo_vector"][EMOTIONS.index("surprised")] == 0.8
    assert kw["use_emo_text"] is False
    assert kw["emo_text"] is None
    assert kw["emo_alpha"] == 0.9


def test_indextts2_emo_text_path():
    kw = map_inflection(Inflection(emo_text="almost crying"))
    assert kw["use_emo_text"] is True
    assert kw["emo_text"] == "almost crying"
    assert kw["emo_vector"] is None


def test_indextts2_vector_preferred_over_text():
    kw = map_inflection(Inflection(emotion_vector=_vec("sad", 0.6), emo_text="happy"))
    assert kw["emo_vector"] is not None
    assert kw["use_emo_text"] is False  # vector wins, text ignored


def test_indextts2_emo_audio_takes_precedence():
    kw = map_inflection(Inflection(emotion_vector=_vec("angry", 0.9),
                                   emo_audio="/tmp/perf.wav", emo_alpha=0.7))
    assert kw["emo_audio_prompt"] == "/tmp/perf.wav"
    assert kw["emo_vector"] is None      # audio reference overrides the vector
    assert kw["use_emo_text"] is False
    assert kw["emo_alpha"] == 0.7


def test_indextts2_neutral():
    kw = map_inflection(Inflection())
    assert kw["emo_vector"] is None
    assert kw["emo_audio_prompt"] is None
    assert kw["use_emo_text"] is False


# --------------------------------------------------------------------------
# is_cuda_oom precedence (regression: 'alloc'/'cuda' must be grouped)
# --------------------------------------------------------------------------
def test_is_cuda_oom_detects_variants():
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert is_cuda_oom(RuntimeError("cuda oom while allocating"))
    assert is_cuda_oom(RuntimeError("failed to allocate 2GB on device cuda:0"))


def test_is_cuda_oom_ignores_unrelated():
    assert not is_cuda_oom(ValueError("bad shape"))
    # 'alloc' alone (no 'cuda') must NOT trip the grouped clause
    assert not is_cuda_oom(RuntimeError("could not allocate host buffer"))


def test_engine_oom_is_engine_error_subclass():
    from inflect.models.engine_base import EngineError

    assert issubclass(EngineOOMError, EngineError)
