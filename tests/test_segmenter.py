"""Tests for document segmentation and segment-cache hashing."""

from __future__ import annotations

from dataclasses import dataclass

from inflect.document.segmenter import (
    MAX_SEGMENT_CHARS,
    SegmentJob,
    segment_document,
)
from inflect.document.spans import Document, Inflection


@dataclass
class _Profile:
    id: str


def test_no_spans_single_segment():
    doc = Document(text="Hello there. How are you?")
    jobs = segment_document(doc)
    assert len(jobs) == 1
    assert jobs[0].text == doc.text
    assert jobs[0].seg_id == 0


def test_cut_at_span_boundaries():
    doc = Document(text="Hello world goodbye")
    #                    0     6     12
    doc.apply_inflection(6, 11, Inflection(emo_text="angry"))  # "world"
    jobs = segment_document(doc)
    texts = [j.text for j in jobs]
    assert texts == ["Hello ", "world", " goodbye"]
    assert jobs[1].inflection.emo_text == "angry"
    assert jobs[0].inflection.emo_text is None
    # seg_ids are contiguous and ordered
    assert [j.seg_id for j in jobs] == [0, 1, 2]


def test_default_inflection_applied_to_unstyled_runs():
    doc = Document(text="abc def", default_inflection=Inflection(speed=1.3))
    doc.apply_inflection(0, 3, Inflection(emo_text="happy"))
    jobs = segment_document(doc)
    assert jobs[0].inflection.emo_text == "happy"
    # the trailing " def" run uses the document default
    assert jobs[-1].inflection.speed == 1.3


def test_whitespace_gap_without_pause_dropped():
    doc = Document(text="A    B")
    doc.apply_inflection(0, 1, Inflection(emo_text="x"))
    doc.apply_inflection(5, 6, Inflection(emo_text="y"))
    jobs = segment_document(doc)
    # middle "    " gap has neutral default + no pause -> dropped
    assert [j.text for j in jobs] == ["A", "B"]


def test_pause_only_segment_retained():
    doc = Document(text="A    B")
    doc.apply_inflection(0, 1, Inflection(emo_text="x"))
    # give the whitespace gap an explicit pause -> must be kept as a silent seg
    doc.apply_inflection(1, 5, Inflection(pause_after_ms=400))
    jobs = segment_document(doc)
    silent = [j for j in jobs if j.is_silent]
    assert len(silent) == 1
    assert silent[0].inflection.pause_after_ms == 400


def test_long_run_split_at_sentences():
    sentence = "This is a fairly long sentence that keeps going. "
    text = sentence * 12  # well over MAX_SEGMENT_CHARS
    doc = Document(text=text)
    jobs = segment_document(doc)
    assert len(jobs) > 1
    assert all(len(j.text) <= MAX_SEGMENT_CHARS for j in jobs)
    # Re-joining the pieces reconstructs the original run exactly.
    assert "".join(j.text for j in jobs) == text


def test_long_run_pause_only_on_last_piece():
    sentence = "Sentence number filler text here. "
    text = sentence * 15
    doc = Document(text=text)
    doc.apply_inflection(0, len(text), Inflection(emo_text="sad", pause_after_ms=500))
    jobs = segment_document(doc)
    assert len(jobs) > 1
    assert all(j.inflection.pause_after_ms == 0 for j in jobs[:-1])
    assert jobs[-1].inflection.pause_after_ms == 500
    assert all(j.inflection.emo_text == "sad" for j in jobs)


def test_hard_wrap_oversized_single_sentence():
    text = "word " * 200  # one 1000-char run with no sentence terminators
    doc = Document(text=text)
    jobs = segment_document(doc)
    assert all(len(j.text) <= MAX_SEGMENT_CHARS for j in jobs)
    assert "".join(j.text for j in jobs) == text


def test_char_ranges_map_back_to_text():
    doc = Document(text="Hello world goodbye")
    doc.apply_inflection(6, 11, Inflection(emo_text="angry"))  # "world"
    jobs = segment_document(doc)
    for j in jobs:
        assert doc.text[j.char_start:j.char_end] == j.text
    # the styled segment maps exactly onto the span range
    world = next(j for j in jobs if j.text == "world")
    assert (world.char_start, world.char_end) == (6, 11)


def test_per_span_engine_override():
    doc = Document(text="alpha beta")
    doc.apply_inflection(0, 5, Inflection(engine="fish"))
    jobs = segment_document(doc, engine="indextts2")
    assert jobs[0].engine == "fish"     # span override wins
    assert jobs[-1].engine == "indextts2"  # default for the rest


# --------------------------------------------------------------------------
# hashing / caching
# --------------------------------------------------------------------------
def test_hash_changes_with_text():
    a = SegmentJob(0, "hello", Inflection(), None, "indextts2")
    b = SegmentJob(0, "world", Inflection(), None, "indextts2")
    assert a.hash != b.hash


def test_hash_changes_with_inflection():
    a = SegmentJob(0, "hi", Inflection(emo_text="happy"), None, "indextts2")
    b = SegmentJob(0, "hi", Inflection(emo_text="angry"), None, "indextts2")
    assert a.hash != b.hash


def test_hash_changes_with_engine():
    a = SegmentJob(0, "hi", Inflection(), None, "indextts2")
    b = SegmentJob(0, "hi", Inflection(), None, "chatterbox")
    assert a.hash != b.hash


def test_hash_changes_with_profile():
    a = SegmentJob(0, "hi", Inflection(), _Profile("p1"), "indextts2")
    b = SegmentJob(0, "hi", Inflection(), _Profile("p2"), "indextts2")
    assert a.hash != b.hash


def test_hash_changes_with_engine_params():
    a = SegmentJob(0, "hi", Inflection(), None, "chatterbox", {"cfg_weight": 0.5})
    b = SegmentJob(0, "hi", Inflection(), None, "chatterbox", {"cfg_weight": 0.7})
    assert a.hash != b.hash


def test_hash_stable_across_instances():
    a = SegmentJob(0, "hi", Inflection(emo_text="x"), _Profile("p"), "fish", {"k": 1})
    b = SegmentJob(9, "hi", Inflection(emo_text="x"), _Profile("p"), "fish", {"k": 1})
    # seg_id is not part of the cache identity
    assert a.hash == b.hash


def test_hash_ignores_pause_only_difference():
    a = SegmentJob(0, "hi", Inflection(pause_after_ms=0), None, "indextts2")
    b = SegmentJob(0, "hi", Inflection(pause_after_ms=900), None, "indextts2")
    assert a.hash == b.hash  # pause handled at assembly, audio identical
