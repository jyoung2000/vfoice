"""Tests for .inflect project save/load round-tripping and migration."""

from __future__ import annotations

import json

import pytest

from inflect.document.project_io import (
    PROJECT_SUFFIX,
    SCHEMA_VERSION,
    Project,
    load_project,
    save_project,
)
from inflect.document.spans import Document, Inflection


def _sample_project() -> Project:
    doc = Document(text="The quick brown fox.", voice_profile_id="voice-123")
    doc.default_inflection = Inflection(speed=1.1)
    doc.apply_inflection(4, 9, Inflection(emotion_vector=[0.0, 0.8, 0, 0, 0, 0, 0, 0],
                                          emo_text="tense", pause_after_ms=250))
    return Project(document=doc, name="Demo", engine="indextts2")


def test_save_adds_suffix(tmp_path):
    path = save_project(_sample_project(), tmp_path / "myproj")
    assert path.suffix == PROJECT_SUFFIX
    assert path.exists()


def test_round_trip_preserves_document(tmp_path):
    proj = _sample_project()
    path = save_project(proj, tmp_path / "p.inflect")
    loaded = load_project(path)

    assert loaded.name == "Demo"
    assert loaded.engine == "indextts2"
    assert loaded.document.text == proj.document.text
    assert loaded.document.voice_profile_id == "voice-123"
    assert loaded.document.default_inflection.speed == 1.1

    orig_spans = proj.document.sorted_spans()
    new_spans = loaded.document.sorted_spans()
    assert len(new_spans) == len(orig_spans) == 1
    assert (new_spans[0].start, new_spans[0].end) == (4, 9)
    assert new_spans[0].inflection.emo_text == "tense"
    assert new_spans[0].inflection.emotion_vector[1] == 0.8
    assert new_spans[0].inflection.pause_after_ms == 250


def test_save_sets_timestamps(tmp_path):
    proj = _sample_project()
    assert proj.created == ""
    save_project(proj, tmp_path / "p.inflect")
    assert proj.created
    assert proj.modified


def test_written_file_has_schema_version(tmp_path):
    path = save_project(_sample_project(), tmp_path / "p.inflect")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION


def test_newer_schema_rejected(tmp_path):
    path = tmp_path / "future.inflect"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "document": {}}))
    with pytest.raises(ValueError):
        load_project(path)


def test_load_minimal_document(tmp_path):
    path = tmp_path / "min.inflect"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                "document": {"text": "hi"}}))
    proj = load_project(path)
    assert proj.document.text == "hi"
    assert proj.document.spans == []
