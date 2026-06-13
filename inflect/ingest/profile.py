"""Voice profiles and the persistent voice library.

A profile is a folder ``voices/<uuid>/`` containing ``reference.wav`` (the
cloning reference clip) and ``meta.json``. A top-level ``voices/index.json``
lists all profiles for fast loading. The library is the CRUD layer over both.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing numpy at module load
    import numpy as np

REFERENCE_FILENAME = "reference.wav"
META_FILENAME = "meta.json"


@dataclass
class VoiceProfile:
    """A reusable cloned-voice reference."""

    id: str
    name: str
    reference_path: str          # absolute path to reference.wav
    source_filename: str = ""    # original media file name
    duration: float = 0.0        # seconds
    sample_rate: int = 24000
    created: str = ""
    notes: str = ""
    consent: bool = False        # "I have permission to clone this voice"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceProfile":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def exists(self) -> bool:
        return Path(self.reference_path).exists()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class VoiceLibrary:
    """CRUD over the on-disk voice library."""

    voices_dir: Path
    index_path: Path
    _profiles: dict[str, VoiceProfile] = field(default_factory=dict)

    @classmethod
    def open(cls, voices_dir: Path, index_path: Path | None = None) -> "VoiceLibrary":
        voices_dir = Path(voices_dir)
        voices_dir.mkdir(parents=True, exist_ok=True)
        index_path = Path(index_path) if index_path else voices_dir / "index.json"
        lib = cls(voices_dir=voices_dir, index_path=index_path)
        lib._load()
        return lib

    # ----- queries ----------------------------------------------------------
    def list(self) -> list[VoiceProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.name.lower())

    def get(self, profile_id: str) -> VoiceProfile | None:
        return self._profiles.get(profile_id)

    def __len__(self) -> int:
        return len(self._profiles)

    # ----- mutation ---------------------------------------------------------
    def add_profile(
        self,
        name: str,
        audio: "np.ndarray",
        sample_rate: int,
        source_filename: str = "",
        consent: bool = False,
        notes: str = "",
    ) -> VoiceProfile:
        """Create a profile from an in-memory mono waveform."""
        import soundfile as sf

        profile_id = uuid.uuid4().hex[:12]
        folder = self.voices_dir / profile_id
        folder.mkdir(parents=True, exist_ok=True)
        ref_path = folder / REFERENCE_FILENAME
        sf.write(str(ref_path), audio, sample_rate)
        duration = float(len(audio)) / float(sample_rate) if sample_rate else 0.0

        profile = VoiceProfile(
            id=profile_id,
            name=name.strip() or f"Voice {profile_id}",
            reference_path=str(ref_path),
            source_filename=source_filename,
            duration=round(duration, 2),
            sample_rate=sample_rate,
            created=_now_iso(),
            notes=notes,
            consent=consent,
        )
        self._write_meta(profile)
        self._profiles[profile_id] = profile
        self._save()
        return profile

    def add_from_wav(
        self,
        name: str,
        wav_path: Path,
        source_filename: str = "",
        consent: bool = False,
        notes: str = "",
    ) -> VoiceProfile:
        """Create a profile by copying an existing reference wav."""
        import soundfile as sf

        info = sf.info(str(wav_path))
        profile_id = uuid.uuid4().hex[:12]
        folder = self.voices_dir / profile_id
        folder.mkdir(parents=True, exist_ok=True)
        ref_path = folder / REFERENCE_FILENAME
        shutil.copyfile(wav_path, ref_path)

        profile = VoiceProfile(
            id=profile_id,
            name=name.strip() or f"Voice {profile_id}",
            reference_path=str(ref_path),
            source_filename=source_filename,
            duration=round(info.frames / info.samplerate, 2) if info.samplerate else 0.0,
            sample_rate=info.samplerate,
            created=_now_iso(),
            notes=notes,
            consent=consent,
        )
        self._write_meta(profile)
        self._profiles[profile_id] = profile
        self._save()
        return profile

    def rename(self, profile_id: str, new_name: str) -> None:
        profile = self._profiles.get(profile_id)
        if not profile:
            raise KeyError(profile_id)
        profile.name = new_name.strip() or profile.name
        self._write_meta(profile)
        self._save()

    def update_notes(self, profile_id: str, notes: str) -> None:
        profile = self._profiles.get(profile_id)
        if not profile:
            raise KeyError(profile_id)
        profile.notes = notes
        self._write_meta(profile)
        self._save()

    def delete(self, profile_id: str) -> None:
        profile = self._profiles.pop(profile_id, None)
        if not profile:
            raise KeyError(profile_id)
        folder = self.voices_dir / profile_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        self._save()

    # ----- persistence ------------------------------------------------------
    def _write_meta(self, profile: VoiceProfile) -> None:
        meta_path = self.voices_dir / profile.id / META_FILENAME
        meta_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    def _save(self) -> None:
        payload = {"profiles": [p.to_dict() for p in self.list()]}
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def _load(self) -> None:
        self._profiles.clear()
        if not self.index_path.exists():
            self._rebuild_from_folders()
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._rebuild_from_folders()
            return
        for entry in data.get("profiles", []):
            try:
                profile = VoiceProfile.from_dict(entry)
                self._profiles[profile.id] = profile
            except (TypeError, KeyError):
                continue

    def _rebuild_from_folders(self) -> None:
        """Reconstruct the index by scanning ``voices/*/meta.json``."""
        for meta in self.voices_dir.glob(f"*/{META_FILENAME}"):
            try:
                profile = VoiceProfile.from_dict(json.loads(meta.read_text(encoding="utf-8")))
                self._profiles[profile.id] = profile
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        if self._profiles:
            self._save()
