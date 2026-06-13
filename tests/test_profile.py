"""Tests for the voice profile library CRUD."""

from __future__ import annotations

import numpy as np

from inflect.ingest.profile import VoiceLibrary, VoiceProfile


def _audio(seconds: float = 1.0, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_add_and_list(tmp_path):
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_profile("Alice", _audio(2.0), 24000, source_filename="a.mp4", consent=True)
    assert isinstance(p, VoiceProfile)
    assert p.consent is True
    assert abs(p.duration - 2.0) < 0.05
    assert (tmp_path / "voices" / p.id / "reference.wav").exists()
    assert len(lib) == 1
    assert lib.get(p.id).name == "Alice"


def test_index_persisted_and_reloaded(tmp_path):
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_profile("Bob", _audio(), 24000)
    # fresh library instance reads from index.json
    lib2 = VoiceLibrary.open(tmp_path / "voices")
    assert len(lib2) == 1
    assert lib2.get(p.id).name == "Bob"


def test_rename_and_notes(tmp_path):
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_profile("Carol", _audio(), 24000)
    lib.rename(p.id, "Caroline")
    lib.update_notes(p.id, "studio mic")
    lib2 = VoiceLibrary.open(tmp_path / "voices")
    assert lib2.get(p.id).name == "Caroline"
    assert lib2.get(p.id).notes == "studio mic"


def test_delete_removes_folder(tmp_path):
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_profile("Dave", _audio(), 24000)
    folder = tmp_path / "voices" / p.id
    assert folder.exists()
    lib.delete(p.id)
    assert not folder.exists()
    assert len(lib) == 0


def test_rebuild_from_folders_when_index_missing(tmp_path):
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_profile("Eve", _audio(), 24000)
    # nuke the index; library should rebuild by scanning meta.json files
    (tmp_path / "voices" / "index.json").unlink()
    lib2 = VoiceLibrary.open(tmp_path / "voices")
    assert len(lib2) == 1
    assert lib2.get(p.id).name == "Eve"


def test_add_from_wav(tmp_path):
    import soundfile as sf

    src = tmp_path / "src.wav"
    sf.write(str(src), _audio(1.5), 24000)
    lib = VoiceLibrary.open(tmp_path / "voices")
    p = lib.add_from_wav("Frank", src, source_filename="src.wav", consent=True)
    assert abs(p.duration - 1.5) < 0.05
    assert p.exists


def test_profile_roundtrip():
    p = VoiceProfile(id="x", name="N", reference_path="/tmp/r.wav", duration=3.0, consent=True)
    assert VoiceProfile.from_dict(p.to_dict()) == p
