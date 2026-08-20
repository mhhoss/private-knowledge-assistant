"""Background ingestion job tracking (ADR-17).

`POST /documents` no longer embeds inline: it hands the uploaded files to a background
job and returns immediately, so the API stays responsive while the (CPU-bound, slow)
embedding backend works through them. This module owns exactly that concern — job
identity, per-file progress, and the background runner — and delegates every actual
unit of ingestion work to `rag/engine.ingest_file` (invariant 3 stays intact: this
module sequences *jobs*, not retrieval/generation, and never reimplements per-file
ingestion, dedup, or compensation logic).

In-memory only, per process, by design: a job is gone after a restart. This is safe
because nothing is ever written to the vector store until a file's chunks are fully
embedded (ADR-7) — a restart mid-job loses that job's tracking, never leaves partial
data behind. See ADR-17 for the full rationale.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from app.rag import engine
from app.rag.indexer import IngestOutcome

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from app.storage.vector_store import VectorStore

# Held for a job's entire file loop (`run_ingestion_job`) so at most one job ever
# embeds at a time. The embedding backend already serializes requests internally
# (a single local model instance), so this doesn't change what the backend can do — it
# keeps this process's own behavior deterministic and sequential rather than opening
# concurrent HTTP calls into a backend never asked to handle them.
_ingestion_lock = threading.Lock()


class FileStatus(StrEnum):
    """Per-file progress within a job.

    `INDEXED`/`ALREADY_INDEXED`/`FAILED` mirror `indexer.IngestStatus` exactly — those
    three come straight from a real `IngestOutcome`. `QUEUED`/`PROCESSING` are job-only
    states with no domain equivalent. `SKIPPED` is also job-only: a file whose turn
    never came because cancellation was requested first (see `JobStore.request_cancel`)
    — never a real ingestion attempt, so it carries no `IngestOutcome` either.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ALREADY_INDEXED = "already_indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


_TERMINAL_FILE_STATUSES = frozenset(
    {
        FileStatus.INDEXED,
        FileStatus.ALREADY_INDEXED,
        FileStatus.FAILED,
        FileStatus.SKIPPED,
    }
)


class JobStatus(StrEnum):
    """Derived from `IngestionJob.started_at`/`finished_at` — never stored directly."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class FileProgress:
    """One file's progress within a job."""

    filename: str
    status: FileStatus = FileStatus.QUEUED
    document_id: str | None = None
    chunk_count: int = 0
    error: str | None = None

    @classmethod
    def from_outcome(cls, outcome: IngestOutcome) -> FileProgress:
        return cls(
            filename=outcome.filename,
            status=FileStatus(outcome.status.value),
            document_id=outcome.document_id,
            chunk_count=outcome.chunk_count,
            error=outcome.error,
        )


@dataclass
class IngestionJob:
    """One ingestion request's lifecycle: queued -> running -> completed.

    Timestamps use `time.monotonic()` — they exist only to derive status and an
    evidence-based ETA within this process's lifetime, never to persist or compare
    across restarts.
    """

    job_id: str
    files: list[FileProgress]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def total(self) -> int:
        return len(self.files)

    @property
    def completed_count(self) -> int:
        return sum(1 for f in self.files if f.status in _TERMINAL_FILE_STATUSES)

    @property
    def current_filename(self) -> str | None:
        for f in self.files:
            if f.status is FileStatus.PROCESSING:
                return f.filename
        return None

    @property
    def status(self) -> JobStatus:
        if self.finished_at is not None:
            return JobStatus.COMPLETED
        if self.started_at is not None:
            return JobStatus.RUNNING
        return JobStatus.QUEUED

    @property
    def eta_seconds(self) -> float | None:
        """Estimated remaining time, extrapolated only from this job's own completed
        files so far. `None` before the first file finishes or once the job is done —
        never a guess made before there is real progress to base it on."""
        completed = self.completed_count
        remaining = self.total - completed
        if completed == 0 or remaining <= 0 or self.started_at is None:
            return None
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return round((elapsed / completed) * remaining, 1)


class JobStore:
    """In-memory, per-process ingestion job registry. See module docstring."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def create(self, filenames: list[str]) -> IngestionJob:
        job = IngestionJob(
            job_id=uuid.uuid4().hex,
            files=[FileProgress(filename=name) for name in filenames],
            created_at=time.monotonic(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._cancel_events[job.job_id] = threading.Event()
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def request_cancel(self, job_id: str) -> bool:
        """Ask a running job to stop starting new files (files already in progress
        still finish normally — see `run_ingestion_job`). Returns `False` if the job
        id is unknown or the job has already completed."""
        with self._lock:
            job = self._jobs.get(job_id)
            event = self._cancel_events.get(job_id)
            if job is None or event is None or job.finished_at is not None:
                return False
            event.set()
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(job_id)
            return event.is_set() if event is not None else False

    def mark_started(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.started_at is None:
                job.started_at = time.monotonic()

    def mark_processing(self, job_id: str, index: int) -> None:
        with self._lock:
            self._jobs[job_id].files[index].status = FileStatus.PROCESSING

    def record_outcome(self, job_id: str, index: int, outcome: IngestOutcome) -> None:
        with self._lock:
            self._jobs[job_id].files[index] = FileProgress.from_outcome(outcome)

    def mark_skipped(self, job_id: str, index: int) -> None:
        """A file whose turn never came because cancellation was requested first."""
        with self._lock:
            file = self._jobs[job_id].files[index]
            file.status = FileStatus.SKIPPED
            file.error = None

    def mark_finished(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].finished_at = time.monotonic()


def run_ingestion_job(
    *,
    job_store: JobStore,
    job_id: str,
    store: VectorStore,
    embed_model: BaseEmbedding,
    files: list[tuple[str, bytes]],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Process one job's files in order, in a background thread (ADR-17).

    Delegates each file to `engine.ingest_file` unchanged — the same load/parse/chunk/
    index path, the same per-file compensation (ADR-7), the same dedup (ADR-3) as
    before this module existed. Only the bookkeeping (job/file progress) is new.

    A cancellation request (`JobStore.request_cancel`) is checked before each
    not-yet-started file; a file already mid-embedding always finishes normally —
    cancellation only ever shortens the *remaining* queue, the same rule the UI's
    "Cancel remaining" affordance already documented before this module existed.
    """
    with _ingestion_lock:
        job_store.mark_started(job_id)
        for index, (filename, content) in enumerate(files):
            if job_store.is_cancelled(job_id):
                job_store.mark_skipped(job_id, index)
                continue
            job_store.mark_processing(job_id, index)
            outcome = engine.ingest_file(
                store=store,
                embed_model=embed_model,
                filename=filename,
                content=content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            job_store.record_outcome(job_id, index, outcome)
        job_store.mark_finished(job_id)
