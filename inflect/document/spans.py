"""Core document data model for Inflect Studio.

Defines the inflection / span / document structures plus the rich-text-style
logic for applying, clearing and remapping inflection spans. This module is
pure Python (stdlib only) so it can be unit-tested without any heavy deps.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace

# IndexTTS-2 emotion dimension order. Do not reorder: the vector index maps
# directly onto the engine's 8-dim emotion input.
EMOTIONS: list[str] = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]

# Number of distinct underline colors cycled through in the editor.
NUM_SPAN_COLORS = 8


@dataclass
class Inflection:
    """How a span of text should be spoken.

    All fields are optional / defaulted so an empty ``Inflection`` means
    "use the engine's neutral delivery".
    """

    emotion_vector: list[float] | None = None  # len 8, each 0.0-1.0, or None
    emo_text: str | None = None                # free-form delivery description -> IndexTTS-2 T2E
    emo_audio: str | None = None               # path to an emotion reference wav (optional)
    emo_alpha: float = 0.8                      # blend strength of emotion vs neutral
    speed: float = 1.0                          # 0.5-1.5, duration control / time-stretch
    pause_after_ms: int = 0                     # silence appended after this span
    engine: str | None = None                   # None = document default engine (Phase 6)

    # ----- validation / normalization -------------------------------------
    def normalized(self) -> "Inflection":
        """Return a copy with values clamped to valid ranges."""
        vec = self.emotion_vector
        if vec is not None:
            if len(vec) != len(EMOTIONS):
                raise ValueError(
                    f"emotion_vector must have {len(EMOTIONS)} entries, got {len(vec)}"
                )
            vec = [min(1.0, max(0.0, float(v))) for v in vec]
        return Inflection(
            emotion_vector=vec,
            emo_text=(self.emo_text or None),
            emo_audio=(self.emo_audio or None),
            emo_alpha=min(1.0, max(0.0, float(self.emo_alpha))),
            speed=min(1.5, max(0.5, float(self.speed))),
            pause_after_ms=max(0, int(self.pause_after_ms)),
            engine=(self.engine or None),
        )

    def is_neutral(self) -> bool:
        """True when this inflection carries no styling at all."""
        has_vec = self.emotion_vector is not None and any(v > 0 for v in self.emotion_vector)
        return not (
            has_vec
            or self.emo_text
            or self.emo_audio
            or self.speed != 1.0
            or self.pause_after_ms != 0
            or self.engine
        )

    # ----- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "emotion_vector": self.emotion_vector,
            "emo_text": self.emo_text,
            "emo_audio": self.emo_audio,
            "emo_alpha": self.emo_alpha,
            "speed": self.speed,
            "pause_after_ms": self.pause_after_ms,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Inflection":
        vec = d.get("emotion_vector")
        return cls(
            emotion_vector=list(vec) if vec is not None else None,
            emo_text=d.get("emo_text"),
            emo_audio=d.get("emo_audio"),
            emo_alpha=float(d.get("emo_alpha", 0.8)),
            speed=float(d.get("speed", 1.0)),
            pause_after_ms=int(d.get("pause_after_ms", 0)),
            engine=d.get("engine"),
        )

    def canonical(self) -> str:
        """Stable JSON string used for segment cache hashing.

        Excludes ``pause_after_ms`` because pauses are inserted during assembly
        rather than baked into the rendered segment audio, so two segments that
        differ only in trailing pause can share a cached wav.
        """
        payload = self.to_dict()
        payload.pop("pause_after_ms", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def summary(self) -> str:
        """Short human-readable label, e.g. 'angry 0.7 · surprised 0.3 · 1.1×'."""
        parts: list[str] = []
        if self.emotion_vector is not None:
            ranked = sorted(
                ((EMOTIONS[i], v) for i, v in enumerate(self.emotion_vector) if v > 0.01),
                key=lambda kv: kv[1],
                reverse=True,
            )
            parts.extend(f"{name} {val:.2f}" for name, val in ranked[:3])
        if self.emo_text:
            parts.append(f'"{self.emo_text}"')
        if self.emo_audio:
            parts.append("perf-ref")
        if abs(self.speed - 1.0) > 1e-3:
            parts.append(f"{self.speed:.2g}×")
        if self.pause_after_ms:
            parts.append(f"+{self.pause_after_ms}ms")
        if self.engine:
            parts.append(f"[{self.engine}]")
        return " · ".join(parts) if parts else "default"


@dataclass
class InflectionSpan:
    """An inflection applied to ``text[start:end]`` (end exclusive)."""

    start: int
    end: int
    inflection: Inflection
    color_idx: int = 0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("span offsets must be non-negative")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "color_idx": self.color_idx,
            "inflection": self.inflection.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InflectionSpan":
        return cls(
            start=int(d["start"]),
            end=int(d["end"]),
            color_idx=int(d.get("color_idx", 0)),
            inflection=Inflection.from_dict(d.get("inflection", {})),
        )


def _subtract(span: InflectionSpan, start: int, end: int) -> list[InflectionSpan]:
    """Return the fragments of ``span`` that are *not* covered by ``[start, end)``.

    Produces 0, 1 or 2 fragments. Each fragment is a deep copy that preserves
    the original span's inflection and color, so this single helper handles
    every overlap case (no-overlap, engulf, overlap-left/right, contains).
    """
    frags: list[InflectionSpan] = []
    # Left fragment: part of span before the cleared range.
    if span.start < start:
        frags.append(
            InflectionSpan(span.start, min(span.end, start),
                           copy.deepcopy(span.inflection), span.color_idx)
        )
    # Right fragment: part of span after the cleared range.
    if span.end > end:
        frags.append(
            InflectionSpan(max(span.start, end), span.end,
                           copy.deepcopy(span.inflection), span.color_idx)
        )
    return [f for f in frags if f.length > 0]


@dataclass
class Document:
    """The full text plus its non-overlapping inflection spans."""

    text: str = ""
    spans: list[InflectionSpan] = field(default_factory=list)
    default_inflection: Inflection = field(default_factory=Inflection)
    voice_profile_id: str | None = None

    # ----- span queries -----------------------------------------------------
    def sorted_spans(self) -> list[InflectionSpan]:
        return sorted(self.spans, key=lambda s: s.start)

    def span_at(self, pos: int) -> InflectionSpan | None:
        """Span whose range contains ``pos`` (caret-friendly: also matches the
        span ending exactly at ``pos`` when ``pos`` is at end-of-text)."""
        for span in self.spans:
            if span.contains(pos):
                return span
        return None

    def inflection_at(self, pos: int) -> Inflection:
        """Effective inflection at ``pos`` — the span's, or the doc default."""
        span = self.span_at(pos)
        return span.inflection if span else self.default_inflection

    # ----- span mutation ----------------------------------------------------
    def apply_inflection(
        self, start: int, end: int, inflection: Inflection, color_idx: int = 0
    ) -> InflectionSpan | None:
        """Apply ``inflection`` to ``[start, end)``, splitting any overlapping
        spans like rich-text formatting. Returns the newly created span."""
        start, end = self._clamp_range(start, end)
        if start >= end:
            return None
        rebuilt: list[InflectionSpan] = []
        for span in self.spans:
            rebuilt.extend(_subtract(span, start, end))
        new_span = InflectionSpan(start, end, inflection.normalized(), color_idx)
        rebuilt.append(new_span)
        rebuilt.sort(key=lambda s: s.start)
        self.spans = rebuilt
        return new_span

    def clear_inflection(self, start: int, end: int) -> None:
        """Remove inflection coverage over ``[start, end)``, splitting spans."""
        start, end = self._clamp_range(start, end)
        if start >= end:
            return
        rebuilt: list[InflectionSpan] = []
        for span in self.spans:
            rebuilt.extend(_subtract(span, start, end))
        rebuilt.sort(key=lambda s: s.start)
        self.spans = rebuilt

    def remap_on_edit(self, position: int, removed: int, added: int) -> None:
        """Shift/shrink span offsets after a text edit.

        Mirrors Qt's ``contentsChange(position, charsRemoved, charsAdded)``:
        the range ``[position, position+removed)`` is replaced by ``added``
        characters. Spans that fully collapse are dropped.
        """
        if removed == 0 and added == 0:
            return
        n = len(self.text)
        rebuilt: list[InflectionSpan] = []
        for span in self.spans:
            new_start = _remap_start(span.start, position, removed, added)
            new_end = _remap_end(span.end, position, removed, added)
            new_start = max(0, min(new_start, n))
            new_end = max(0, min(new_end, n))
            if new_end > new_start:
                rebuilt.append(replace(span, start=new_start, end=new_end))
        rebuilt.sort(key=lambda s: s.start)
        self.spans = rebuilt

    # ----- helpers ----------------------------------------------------------
    def _clamp_range(self, start: int, end: int) -> tuple[int, int]:
        n = len(self.text)
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if start > end:
            start, end = end, start
        return start, end

    # ----- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "voice_profile_id": self.voice_profile_id,
            "default_inflection": self.default_inflection.to_dict(),
            "spans": [s.to_dict() for s in self.sorted_spans()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            text=d.get("text", ""),
            voice_profile_id=d.get("voice_profile_id"),
            default_inflection=Inflection.from_dict(d.get("default_inflection", {})),
            spans=[InflectionSpan.from_dict(s) for s in d.get("spans", [])],
        )


def _remap_start(x: int, position: int, removed: int, added: int) -> int:
    """Remap a span *start* offset after a replace edit (right gravity).

    A pure insertion exactly at the start pushes the start forward, so newly
    typed text at the left boundary is *not* absorbed into the span.
    """
    end_del = position + removed
    if x <= position:
        if removed == 0 and x == position:
            return x + added  # right gravity at the insertion point
        return x
    if x >= end_del:
        return x - removed + added
    return position + added  # head deleted: start at the new content boundary


def _remap_end(x: int, position: int, removed: int, added: int) -> int:
    """Remap a span *end* offset after a replace edit (left gravity).

    A pure insertion exactly at the end leaves the end put, so text typed at
    the right boundary is *not* absorbed; text typed strictly inside grows it.
    """
    end_del = position + removed
    if x <= position:
        return x  # left gravity at the insertion point
    if x >= end_del:
        return x - removed + added
    return position  # tail deleted: end at the deletion point
