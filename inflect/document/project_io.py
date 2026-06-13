"""Versioned save/load for ``.inflect`` project files.

A project bundles the editor document plus a little metadata. The format is
plain JSON with a ``schema_version`` so older files can be migrated forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .spans import Document

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".inflect"


@dataclass
class Project:
    """An in-memory project: a document plus bookkeeping metadata."""

    document: Document = field(default_factory=Document)
    name: str = "Untitled"
    created: str = ""
    modified: str = ""
    # Default engine mode for the document: "chatterbox" (Draft) or
    # "indextts2" (Final). Per-span overrides live on each Inflection.
    engine: str = "indextts2"

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "engine": self.engine,
            "document": self.document.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        d = _migrate(d)
        return cls(
            name=d.get("name", "Untitled"),
            created=d.get("created", ""),
            modified=d.get("modified", ""),
            engine=d.get("engine", "indextts2"),
            document=Document.from_dict(d.get("document", {})),
        )


def _migrate(d: dict) -> dict:
    """Forward-migrate older schema versions. No-op for the current version."""
    version = int(d.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Project schema_version {version} is newer than supported "
            f"{SCHEMA_VERSION}; please update Inflect Studio."
        )
    # Future migrations: if version < SCHEMA_VERSION, transform here.
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_project(project: Project, path: str | Path) -> Path:
    """Write ``project`` to ``path`` (adding the ``.inflect`` suffix)."""
    path = Path(path)
    if path.suffix != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    if not project.created:
        project.created = _now_iso()
    project.modified = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic-ish write to avoid corrupting on crash
    return path


def load_project(path: str | Path) -> Project:
    """Read a ``.inflect`` file into a :class:`Project`."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Project.from_dict(data)
