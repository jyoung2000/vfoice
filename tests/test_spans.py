"""Tests for the inflection span model: split, clear and edit-remap logic."""

from __future__ import annotations

from inflect.document.spans import (
    EMOTIONS,
    Document,
    Inflection,
    InflectionSpan,
)


def _doc(text: str = "0123456789ABCDEF") -> Document:
    return Document(text=text)


def _ranges(doc: Document) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in doc.sorted_spans()]


def _infl(tag: str) -> Inflection:
    # emo_text used purely as an identity marker in tests
    return Inflection(emo_text=tag)


def _no_overlaps(doc: Document) -> bool:
    spans = doc.sorted_spans()
    return all(a.end <= b.start for a, b in zip(spans, spans[1:]))


# --------------------------------------------------------------------------
# apply_inflection — the six rich-text cases
# --------------------------------------------------------------------------
def test_apply_to_empty():
    doc = _doc()
    doc.apply_inflection(2, 5, _infl("A"))
    assert _ranges(doc) == [(2, 5)]


def test_exact_match_replaces():
    doc = _doc()
    doc.apply_inflection(2, 5, _infl("A"))
    doc.apply_inflection(2, 5, _infl("B"))
    assert _ranges(doc) == [(2, 5)]
    assert doc.span_at(3).inflection.emo_text == "B"


def test_insert_inside_splits_into_three():
    doc = _doc()
    doc.apply_inflection(2, 8, _infl("A"))
    doc.apply_inflection(4, 6, _infl("B"))
    assert _ranges(doc) == [(2, 4), (4, 6), (6, 8)]
    assert doc.span_at(2).inflection.emo_text == "A"
    assert doc.span_at(4).inflection.emo_text == "B"
    assert doc.span_at(7).inflection.emo_text == "A"
    assert _no_overlaps(doc)


def test_overlap_left_truncates_existing():
    # existing covers the right side; new overlaps its left edge
    doc = _doc()
    doc.apply_inflection(4, 10, _infl("A"))
    doc.apply_inflection(2, 6, _infl("B"))
    assert _ranges(doc) == [(2, 6), (6, 10)]
    assert doc.span_at(2).inflection.emo_text == "B"
    assert doc.span_at(6).inflection.emo_text == "A"


def test_overlap_right_truncates_existing():
    doc = _doc()
    doc.apply_inflection(2, 8, _infl("A"))
    doc.apply_inflection(6, 12, _infl("B"))
    assert _ranges(doc) == [(2, 6), (6, 12)]
    assert doc.span_at(2).inflection.emo_text == "A"
    assert doc.span_at(6).inflection.emo_text == "B"


def test_engulf_removes_existing():
    doc = _doc()
    doc.apply_inflection(4, 6, _infl("A"))
    doc.apply_inflection(2, 8, _infl("B"))
    assert _ranges(doc) == [(2, 8)]
    assert doc.span_at(5).inflection.emo_text == "B"


def test_adjacent_keeps_both():
    doc = _doc()
    doc.apply_inflection(2, 5, _infl("A"))
    doc.apply_inflection(5, 8, _infl("B"))
    assert _ranges(doc) == [(2, 5), (5, 8)]
    assert _no_overlaps(doc)


def test_engulf_multiple_existing():
    doc = _doc()
    doc.apply_inflection(1, 3, _infl("A"))
    doc.apply_inflection(4, 6, _infl("B"))
    doc.apply_inflection(7, 9, _infl("C"))
    doc.apply_inflection(2, 8, _infl("X"))  # eats middle, trims neighbours
    assert _ranges(doc) == [(1, 2), (2, 8), (8, 9)]
    assert doc.span_at(1).inflection.emo_text == "A"
    assert doc.span_at(5).inflection.emo_text == "X"
    assert doc.span_at(8).inflection.emo_text == "C"


def test_reversed_range_normalized():
    doc = _doc()
    doc.apply_inflection(7, 3, _infl("A"))  # passed end<start
    assert _ranges(doc) == [(3, 7)]


def test_apply_clamps_to_text_length():
    doc = _doc("abc")
    doc.apply_inflection(1, 99, _infl("A"))
    assert _ranges(doc) == [(1, 3)]


def test_zero_length_apply_is_noop():
    doc = _doc()
    assert doc.apply_inflection(4, 4, _infl("A")) is None
    assert _ranges(doc) == []


# --------------------------------------------------------------------------
# clear_inflection
# --------------------------------------------------------------------------
def test_clear_splits_span():
    doc = _doc()
    doc.apply_inflection(2, 10, _infl("A"))
    doc.clear_inflection(4, 6)
    assert _ranges(doc) == [(2, 4), (6, 10)]


def test_clear_entire_span():
    doc = _doc()
    doc.apply_inflection(2, 6, _infl("A"))
    doc.clear_inflection(0, 10)
    assert _ranges(doc) == []


# --------------------------------------------------------------------------
# remap_on_edit
# --------------------------------------------------------------------------
def test_remap_insert_before_shifts_right():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = "XXX" + doc.text
    doc.remap_on_edit(0, 0, 3)
    assert _ranges(doc) == [(7, 11)]


def test_remap_insert_inside_grows_span():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[:6] + "YY" + doc.text[6:]
    doc.remap_on_edit(6, 0, 2)
    assert _ranges(doc) == [(4, 10)]


def test_remap_insert_at_start_excludes_new_text():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[:4] + "ZZ" + doc.text[4:]
    doc.remap_on_edit(4, 0, 2)  # right gravity: span moves, new text outside
    assert _ranges(doc) == [(6, 10)]


def test_remap_insert_at_end_excludes_new_text():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[:8] + "QQ" + doc.text[8:]
    doc.remap_on_edit(8, 0, 2)  # left gravity: end stays, new text outside
    assert _ranges(doc) == [(4, 8)]


def test_remap_delete_before_shifts_left():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[2:]
    doc.remap_on_edit(0, 2, 0)
    assert _ranges(doc) == [(2, 6)]


def test_remap_delete_overlapping_start():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[:2] + doc.text[6:]
    doc.remap_on_edit(2, 4, 0)  # deletes [2,6): trims span head
    assert _ranges(doc) == [(2, 4)]


def test_remap_delete_collapses_span():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.text = doc.text[:3] + doc.text[9:]
    doc.remap_on_edit(3, 6, 0)  # deletes [3,9): engulfs the span entirely
    assert _ranges(doc) == []


def test_remap_replace_trims_tail():
    doc = _doc()
    doc.apply_inflection(4, 8, _infl("A"))
    doc.remap_on_edit(6, 4, 3)  # replace [6,10) with 3 chars
    assert _ranges(doc) == [(4, 6)]


# --------------------------------------------------------------------------
# Inflection helpers
# --------------------------------------------------------------------------
def test_inflection_normalized_clamps():
    infl = Inflection(emotion_vector=[2.0] * 8, emo_alpha=5, speed=9, pause_after_ms=-3)
    n = infl.normalized()
    assert n.emotion_vector == [1.0] * 8
    assert n.emo_alpha == 1.0
    assert n.speed == 1.5
    assert n.pause_after_ms == 0


def test_inflection_vector_wrong_length_raises():
    import pytest

    with pytest.raises(ValueError):
        Inflection(emotion_vector=[0.5, 0.5]).normalized()


def test_inflection_is_neutral():
    assert Inflection().is_neutral()
    assert not Inflection(speed=1.2).is_neutral()
    assert not Inflection(emo_text="sad").is_neutral()
    assert not Inflection(emotion_vector=[0, 1, 0, 0, 0, 0, 0, 0]).is_neutral()


def test_inflection_canonical_ignores_pause():
    a = Inflection(emo_text="x", pause_after_ms=0).canonical()
    b = Inflection(emo_text="x", pause_after_ms=500).canonical()
    assert a == b  # pause is applied at assembly, not baked into the segment


def test_inflection_summary_ranks_emotions():
    vec = [0.0] * 8
    vec[EMOTIONS.index("angry")] = 0.7
    vec[EMOTIONS.index("surprised")] = 0.3
    summary = Inflection(emotion_vector=vec, speed=1.1).summary()
    assert "angry 0.70" in summary
    assert "surprised 0.30" in summary
    assert "1.1×" in summary


def test_inflection_roundtrip():
    infl = Inflection(emotion_vector=[0.1] * 8, emo_text="soft", emo_alpha=0.5,
                      speed=1.2, pause_after_ms=200, engine="fish")
    assert Inflection.from_dict(infl.to_dict()) == infl


def test_span_roundtrip():
    span = InflectionSpan(2, 5, _infl("A"), color_idx=3)
    assert InflectionSpan.from_dict(span.to_dict()) == span
