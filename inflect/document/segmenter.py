"""Split a :class:`Document` into cacheable :class:`SegmentJob` units.

Cuts happen at every inflection-span boundary so that each segment has a single
constant delivery. Long constant runs are further split at sentence boundaries
so the TTS engine receives manageable chunks. Each job carries a stable
``hash`` so the synthesis pipeline can cache rendered wavs and only re-render
the segments that actually changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .spans import Document, Inflection

# Soft cap on characters per segment. Runs longer than this are split at
# sentence boundaries; a single sentence longer than this is hard-wrapped.
MAX_SEGMENT_CHARS = 400

# Sentence terminators followed by whitespace or end-of-text, allowing a
# closing quote/bracket to ride along with the terminator.
_SENTENCE_RE = re.compile(r'[^.?!…]*[.?!…]+["”\')\]]*(?:\s+|$)', re.UNICODE)


class _HasId(Protocol):
    id: str


@dataclass
class SegmentJob:
    """A single unit of synthesis work.

    ``voice_profile`` is anything exposing an ``.id`` (a ``VoiceProfile``) or
    ``None``. ``hash`` uniquely identifies the rendered audio for caching.
    """

    seg_id: int
    text: str
    inflection: Inflection
    voice_profile: _HasId | None
    engine: str
    engine_params: dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    char_start: int = 0  # source offset in the document (not part of the hash)
    char_end: int = 0

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = self.compute_hash()

    @property
    def profile_id(self) -> str:
        return getattr(self.voice_profile, "id", None) or ""

    @property
    def is_silent(self) -> bool:
        """A pause-only segment with no speakable text."""
        return self.text.strip() == ""

    def compute_hash(self) -> str:
        payload = "|".join(
            [
                self.text,
                self.inflection.canonical(),
                self.profile_id,
                self.engine,
                json.dumps(self.engine_params, sort_keys=True, separators=(",", ":")),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving terminators and trailing space."""
    matches = [m.group(0) for m in _SENTENCE_RE.finditer(text)]
    consumed = sum(len(m) for m in matches)
    if consumed < len(text):
        # Trailing fragment with no terminator (e.g. unfinished sentence).
        matches.append(text[consumed:])
    return [m for m in matches if m]


def _hard_wrap(chunk: str, limit: int) -> list[str]:
    """Wrap an over-long sentence at whitespace nearest ``limit``."""
    out: list[str] = []
    rest = chunk
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = limit  # no space found; split mid-word as a last resort
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


def _pack_run(text: str, limit: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split a constant-inflection run into <=``limit``-char pieces.

    Short runs pass through untouched; long runs are split at sentence
    boundaries, greedily packing consecutive sentences up to ``limit``.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_wrap(sentence, limit))
            continue
        if current and len(current) + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def _boundaries(doc: Document) -> list[int]:
    """All cut points: 0, span edges, end-of-text — sorted and de-duplicated."""
    pts = {0, len(doc.text)}
    for span in doc.spans:
        pts.add(max(0, min(span.start, len(doc.text))))
        pts.add(max(0, min(span.end, len(doc.text))))
    return sorted(pts)


def segment_document(
    doc: Document,
    voice_profile: _HasId | None = None,
    engine: str = "indextts2",
    engine_params: dict[str, Any] | None = None,
) -> list[SegmentJob]:
    """Turn ``doc`` into an ordered list of :class:`SegmentJob`.

    The per-span ``engine`` override (Phase 6) takes precedence over the
    ``engine`` argument, which acts as the document default.
    """
    engine_params = engine_params or {}
    jobs: list[SegmentJob] = []
    boundaries = _boundaries(doc)
    seg_id = 0

    for a, b in zip(boundaries, boundaries[1:]):
        if a >= b:
            continue
        span = doc.span_at(a)
        inflection = span.inflection if span else doc.default_inflection
        seg_engine = inflection.engine or engine
        run_text = doc.text[a:b]
        pieces = _pack_run(run_text)

        offset = a
        for i, piece in enumerate(pieces):
            is_last = i == len(pieces) - 1
            # Only the final piece of a run keeps the trailing pause so we
            # never insert pauses inside a single delivery run.
            piece_infl = inflection if is_last else replace(inflection, pause_after_ms=0)
            if piece.strip() == "" and piece_infl.pause_after_ms == 0:
                offset += len(piece)
                continue  # drop empty whitespace gaps with no pause
            jobs.append(
                SegmentJob(
                    seg_id=seg_id,
                    text=piece,
                    inflection=piece_infl.normalized(),
                    voice_profile=voice_profile,
                    engine=seg_engine,
                    engine_params=dict(engine_params),
                    char_start=offset,
                    char_end=offset + len(piece),
                )
            )
            offset += len(piece)
            seg_id += 1

    return jobs
