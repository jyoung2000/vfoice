"""Render a list of SegmentJobs into one assembled track.

Jobs are grouped by engine and rendered one engine at a time (so model swaps
are minimized — exactly the discipline a 12 GB GPU needs), then reassembled in
the original document order. Every segment is cached on disk by its hash, so
editing one phrase only re-renders that one segment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..document.segmenter import SegmentJob
from ..logging_setup import get_logger
from ..models.model_manager import ModelManager
from .assemble import RenderedSegment, assemble

log = get_logger("pipeline")

ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


@dataclass
class SegmentLayout:
    """Where a segment landed in the assembled mix (for timeline markers)."""

    seg_id: int
    start_s: float
    end_s: float
    char_start: int
    char_end: int
    job_hash: str


@dataclass
class _RenderUnit:
    """One render task. ``kind`` is 'final' (its audio is the segment) or
    'perf' (a hybrid stage-1 performance written to ``perf_path``).

    ``out_index`` is the position in the original ``jobs`` list this unit's
    audio belongs to (final units only); ``seg_id`` is the document segment id
    used for force/re-render targeting.
    """

    engine: str
    kind: str
    job: SegmentJob
    seg_id: int
    out_index: int
    perf_path: Path | None = None


class CancelledError(RuntimeError):
    """Raised when synthesis is cancelled between segments."""


# Render engines in this order so model swaps are minimized and, crucially,
# Fish (hybrid stage 1) always runs before IndexTTS-2 (hybrid stage 2).
_ENGINE_ORDER = ["fish", "indextts2", "chatterbox"]


def _engine_rank(name: str) -> int:
    return _ENGINE_ORDER.index(name) if name in _ENGINE_ORDER else len(_ENGINE_ORDER)


class SynthesisPipeline:
    def __init__(self, manager: ModelManager, cache_dir: Path, settings) -> None:
        self.manager = manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.last_layout: list[SegmentLayout] = []

    # ----- caching ----------------------------------------------------------
    def cache_path(self, job_hash: str) -> Path:
        return self.cache_dir / f"{job_hash}.wav"

    def is_cached(self, job: SegmentJob) -> bool:
        return job.is_silent or self.cache_path(job.hash).exists()

    def invalidate(self, job_hash: str) -> None:
        path = self.cache_path(job_hash)
        if path.exists():
            path.unlink()

    # ----- rendering --------------------------------------------------------
    def _render_job(self, job: SegmentJob, force: bool) -> RenderedSegment:
        if job.is_silent:
            return RenderedSegment(np.zeros(0, dtype=np.float32), self._engine_sr(job.engine),
                                   job.inflection.pause_after_ms)

        import soundfile as sf

        path = self.cache_path(job.hash)
        if path.exists() and not force:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            return RenderedSegment(np.asarray(audio, dtype=np.float32).reshape(-1), int(sr),
                                   job.inflection.pause_after_ms)

        engine = self.manager.get_engine(job.engine)
        t0 = time.perf_counter()
        audio = engine.synthesize(job)
        dt = time.perf_counter() - t0
        sr = engine.sample_rate
        log.info("seg %d rendered: %d chars, %.2fs audio in %.2fs (%s) vram=%.0fMB",
                 job.seg_id, len(job.text), len(audio) / sr if sr else 0, dt, job.engine,
                 ModelManager.vram_allocated_mb())
        if audio.size:
            sf.write(str(path), audio, sr)
        return RenderedSegment(audio, sr, job.inflection.pause_after_ms)

    def _plan_units(self, jobs: list[SegmentJob]) -> list[_RenderUnit]:
        """Expand hybrid jobs into (Fish perf + IndexTTS-2 final) unit pairs."""
        units: list[_RenderUnit] = []
        for idx, job in enumerate(jobs):
            if job.engine == "hybrid":
                stage1, stage2, perf_path = self._make_hybrid_stages(job)
                units.append(_RenderUnit("fish", "perf", stage1, job.seg_id, idx, perf_path))
                units.append(_RenderUnit("indextts2", "final", stage2, job.seg_id, idx))
            else:
                units.append(_RenderUnit(job.engine, "final", job, job.seg_id, idx))
        return units

    def _make_hybrid_stages(self, job: SegmentJob) -> tuple[SegmentJob, SegmentJob, Path]:
        """Build the Fish performance job and the IndexTTS-2 timbre job."""
        stage1 = SegmentJob(
            seg_id=job.seg_id, text=job.text, inflection=job.inflection,
            voice_profile=None, engine="fish",
            engine_params={**job.engine_params, "role": "perf"},
        )
        perf_path = self.cache_dir / f"perf_{stage1.hash}.wav"
        # Stage 2: clone the user's timbre, drive emotion from the performance wav.
        stage2_infl = replace(
            job.inflection, emo_audio=str(perf_path), emo_text=None, emotion_vector=None)
        stage2 = SegmentJob(
            seg_id=job.seg_id, text=job.text, inflection=stage2_infl,
            voice_profile=job.voice_profile, engine="indextts2",
            engine_params=job.engine_params,
            char_start=job.char_start, char_end=job.char_end,
        )
        return stage1, stage2, perf_path

    def _render_perf(self, job: SegmentJob, perf_path: Path, force: bool) -> None:
        """Render a hybrid stage-1 performance to ``perf_path`` (cached)."""
        import soundfile as sf

        if perf_path.exists() and not force:
            return
        engine = self.manager.get_engine(job.engine)
        audio = engine.synthesize(job)
        if audio.size:
            sf.write(str(perf_path), audio, engine.sample_rate)
        log.info("hybrid perf rendered seg %d -> %s", job.seg_id, perf_path.name)

    def perf_path_for(self, job: SegmentJob) -> Path:
        """Public: the stage-1 performance wav path for a hybrid ``job`` (for audition)."""
        stage1, _, perf_path = self._make_hybrid_stages(job)
        return perf_path

    def _engine_sr(self, engine_name: str) -> int:
        eng = self.manager._engines.get(engine_name)
        return eng.sample_rate if eng and eng.is_loaded else 24000

    def render(
        self,
        jobs: list[SegmentJob],
        progress_cb: ProgressCb | None = None,
        should_cancel: CancelCb | None = None,
        force_seg_ids: set[int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Render all jobs and return ``(mix, sample_rate)``.

        Hybrid jobs expand into a Fish stage-1 'performance' render and an
        IndexTTS-2 stage-2 render that uses it as the emotion reference. Units
        are rendered grouped by engine (Fish → IndexTTS-2 → Chatterbox) so model
        swaps stay minimal (≤ 2 for a typical mixed document).

        ``force_seg_ids`` forces a re-render (ignoring the cache) of those
        document segments — covering *both* hybrid stages, since they share the
        segment's id.
        """
        if not jobs:
            return np.zeros(0, dtype=np.float32), 24000

        force_seg_ids = force_seg_ids or set()
        units = self._plan_units(jobs)
        # Stable sort by engine rank keeps document order within an engine.
        units.sort(key=lambda u: _engine_rank(u.engine))

        # Keyed by out_index (position in `jobs`) — guaranteed unique and, unlike
        # the job hash, preserves per-segment pauses (hash excludes pause).
        rendered: dict[int, RenderedSegment] = {}
        total = len(units)
        for done, unit in enumerate(units, start=1):
            if should_cancel and should_cancel():
                self.manager.unload_all()
                raise CancelledError()
            force = unit.seg_id in force_seg_ids
            if unit.kind == "perf":
                self._render_perf(unit.job, unit.perf_path, force=force)
            else:
                rendered[unit.out_index] = self._render_job(unit.job, force=force)
            if progress_cb:
                progress_cb(done, total, f"Rendering segment {done}/{total}")

        ordered = [rendered[i] for i in range(len(jobs))]
        target_sr = max((s.sr for s in ordered if s.audio.size), default=24000)
        self.last_layout = self._compute_layout(jobs, ordered, target_sr)
        if progress_cb:
            progress_cb(total, total, "Assembling…")
        mix = assemble(
            ordered,
            target_sr=target_sr,
            crossfade_ms=self.settings.crossfade_ms,
            target_lufs=self.settings.target_lufs,
            true_peak_dbtp=self.settings.true_peak_dbtp,
        )
        return mix, target_sr

    def _compute_layout(self, jobs, ordered, target_sr: int) -> list[SegmentLayout]:
        """Approximate each segment's [start, end] in the crossfaded mix."""
        n_fade = max(0, int(self.settings.crossfade_ms / 1000.0 * target_sr))
        layout: list[SegmentLayout] = []
        pos = 0
        for i, (job, seg) in enumerate(zip(jobs, ordered)):
            audio_len = round(seg.audio.size * target_sr / seg.sr) if seg.audio.size else 0
            pause_len = round(seg.pause_after_ms / 1000.0 * target_sr)
            block_len = audio_len + pause_len
            start = pos if i == 0 else max(0, pos)
            end = start + block_len
            layout.append(SegmentLayout(
                seg_id=job.seg_id,
                start_s=start / target_sr,
                end_s=end / target_sr,
                char_start=getattr(job, "char_start", 0),
                char_end=getattr(job, "char_end", 0),
                job_hash=job.hash,
            ))
            pos = max(0, end - n_fade)  # next block overlaps by the crossfade
        return layout
