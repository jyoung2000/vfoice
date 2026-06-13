"""QObject worker that runs the synthesis pipeline off the GUI thread.

Emits progress, completion, cancellation and failure signals. Cancellation is a
simple thread-safe flag checked by the pipeline between segments.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from ..document.segmenter import SegmentJob
from ..logging_setup import get_logger
from .pipeline import CancelledError, SynthesisPipeline

log = get_logger("worker")


class SynthWorker(QObject):
    progress = Signal(int, int, str)   # done, total, message
    finished = Signal(object, int)     # mix (np.ndarray), sample_rate
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        pipeline: SynthesisPipeline,
        jobs: list[SegmentJob],
        force_seg_ids: set[int] | None = None,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._jobs = jobs
        self._force = force_seg_ids or set()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            # Make sure required engine weights are present before rendering.
            engines = {job.engine for job in self._jobs}
            for name in engines:
                if self._cancel.is_set():
                    raise CancelledError()
                if not self._pipeline.manager.is_downloaded(name):
                    self.progress.emit(0, len(self._jobs), f"Downloading {name} model…")
                    self._pipeline.manager.ensure_downloaded(
                        name, lambda msg, pct: self.progress.emit(0, len(self._jobs), msg)
                    )

            mix, sr = self._pipeline.render(
                self._jobs,
                progress_cb=lambda d, t, m: self.progress.emit(d, t, m),
                should_cancel=self._cancel.is_set,
                force_seg_ids=self._force,
            )
            self.finished.emit(mix, sr)
        except CancelledError:
            log.info("synthesis cancelled")
            self.cancelled.emit()
        except Exception as exc:
            log.exception("synthesis failed")
            self.failed.emit(str(exc))


class PreviewWorker(QObject):
    """Render a single segment (Inspector 'Preview this segment' / audition).

    Routes through the full pipeline so hybrid jobs (Fish → IndexTTS-2) work the
    same as direct ones.
    """

    finished = Signal(object, int)
    failed = Signal(str)

    def __init__(self, pipeline: SynthesisPipeline, job: SegmentJob, force: bool = True) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._job = job
        self._force = force

    def run(self) -> None:
        try:
            engines = {"fish", "indextts2"} if self._job.engine == "hybrid" else {self._job.engine}
            for name in engines:
                if not self._pipeline.manager.is_downloaded(name):
                    self._pipeline.manager.ensure_downloaded(name)
            force = {self._job.seg_id} if self._force else None
            mix, sr = self._pipeline.render([self._job], force_seg_ids=force)
            self.finished.emit(mix, sr)
        except Exception as exc:
            log.exception("preview failed")
            self.failed.emit(str(exc))
