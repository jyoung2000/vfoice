"""Render a list of SegmentJobs into one assembled track.

Jobs are grouped by engine and rendered one engine at a time (so model swaps
are minimized — exactly the discipline a 12 GB GPU needs), then reassembled in
the original document order. Every segment is cached on disk by its hash, so
editing one phrase only re-renders that one segment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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


class CancelledError(RuntimeError):
    """Raised when synthesis is cancelled between segments."""


def _ordered_engines(jobs: list[SegmentJob]) -> list[str]:
    """Unique engine names in first-appearance order."""
    seen: list[str] = []
    for job in jobs:
        if job.engine not in seen:
            seen.append(job.engine)
    return seen


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

    def _engine_sr(self, engine_name: str) -> int:
        eng = self.manager._engines.get(engine_name)
        return eng.sample_rate if eng and eng.is_loaded else 24000

    def render(
        self,
        jobs: list[SegmentJob],
        progress_cb: ProgressCb | None = None,
        should_cancel: CancelCb | None = None,
        force_hashes: set[str] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Render all jobs and return ``(mix, sample_rate)``."""
        if not jobs:
            return np.zeros(0, dtype=np.float32), 24000

        force_hashes = force_hashes or set()
        rendered: dict[int, RenderedSegment] = {}
        total = len(jobs)
        done = 0

        # Group by engine; render one engine's whole batch before swapping.
        by_engine: dict[str, list[SegmentJob]] = {}
        for job in jobs:
            by_engine.setdefault(job.engine, []).append(job)

        for engine_name in _ordered_engines(jobs):
            for job in by_engine[engine_name]:
                if should_cancel and should_cancel():
                    self.manager.unload_all()
                    raise CancelledError()
                seg = self._render_job(job, force=job.hash in force_hashes)
                rendered[job.seg_id] = seg
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"Rendering segment {done}/{total}")

        ordered = [rendered[job.seg_id] for job in jobs]
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

    def render_one(self, job: SegmentJob, force: bool = True) -> tuple[np.ndarray, int]:
        """Render a single segment (for per-segment preview / re-render)."""
        if force:
            self.invalidate(job.hash)
        seg = self._render_job(job, force=force)
        return seg.audio, seg.sr
