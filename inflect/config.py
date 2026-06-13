"""Central configuration: filesystem paths and persisted settings.

This is the single source of global state in Inflect Studio (per the project
constraints). It is deliberately Qt-free so it can be imported by workers,
engines and tests without pulling in the GUI.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

APP_NAME = "InflectStudio"


def _default_home() -> Path:
    """Per-platform application data directory (overridable via INFLECT_HOME)."""
    env = os.environ.get("INFLECT_HOME")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    # macOS / Linux
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


@dataclass(frozen=True)
class Paths:
    """Resolved directories. Created lazily via :meth:`ensure`."""

    home: Path

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def indextts2(self) -> Path:
        return self.models / "indextts2"

    @property
    def chatterbox(self) -> Path:
        return self.models / "chatterbox"

    @property
    def fish(self) -> Path:
        return self.models / "fish"

    @property
    def voices(self) -> Path:
        return self.home / "voices"

    @property
    def voices_index(self) -> Path:
        return self.voices / "index.json"

    @property
    def cache(self) -> Path:
        return self.home / "project_cache"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs / "inflect.log"

    @property
    def settings_file(self) -> Path:
        return self.home / "settings.json"

    @property
    def tmp(self) -> Path:
        return self.home / "tmp"

    def ensure(self) -> "Paths":
        for d in (self.models, self.indextts2, self.chatterbox, self.fish,
                  self.voices, self.cache, self.logs, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Settings:
    """User-tunable settings, persisted as JSON in the app home dir."""

    ffmpeg_path: str = "ffmpeg"          # resolved on PATH unless overridden
    output_device: str | None = None     # sounddevice device name; None = default
    default_engine: str = "indextts2"    # "chatterbox" (Draft) | "indextts2" (Final)
    use_fp16: bool = True
    use_cuda_kernel: bool = True
    enable_demucs: bool = True
    target_lufs: float = -16.0
    true_peak_dbtp: float = -1.0
    crossfade_ms: int = 15
    hf_endpoint: str | None = None        # optional HF mirror endpoint
    accept_fish_license: bool = False     # Phase 6 research-license acknowledgement

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**known)
            except (json.JSONDecodeError, TypeError):
                pass  # fall through to defaults on a corrupt settings file
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class Config:
    """Bundles resolved :class:`Paths` and loaded :class:`Settings`."""

    paths: Paths
    settings: Settings

    def save_settings(self) -> None:
        self.settings.save(self.paths.settings_file)

    def device(self) -> str:
        """Preferred torch device: 'cuda' when available, else 'cpu'."""
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide config singleton (paths ensured, settings loaded)."""
    paths = Paths(home=_default_home()).ensure()
    settings = Settings.load(paths.settings_file)
    return Config(paths=paths, settings=settings)
