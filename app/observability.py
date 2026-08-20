"""Structured logging: the only place that decides how the app logs and where to.

Every other module gets its own logger the normal way (`logging.getLogger(__name__)`)
and calls `log_event` to emit one structured record — this module only owns formatting
and destination (stdout, JSON lines), not what gets logged or when (ADR-15).

Invariant: a logged event is metadata only — filenames, ids, counts, scores, statuses,
durations. Document text, query text, answer text, and credentials must never appear in
a log record (see ADR-15; this mirrors the same rule `config.py`'s `mask_secret`
already enforces for API responses).
"""

from __future__ import annotations

import json
import logging
import sys

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """One JSON object per line: parseable by any log shipper, readable by a human
    piping through `jq`, no schema beyond "these fields always exist"."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once, at process startup. Safe to call more than
    once (e.g. re-entrant test/app startup) — every call after the first is a no-op,
    so handlers are never duplicated."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def log_event(
    logger: logging.Logger, level: int, message: str, /, **fields: object
) -> None:
    """Log one structured event. The only entry point call sites should use for a
    structured event, so every one lands in the same `{message, ...fields}` shape
    instead of each call site re-building `extra={"fields": ...}` itself.
    """
    logger.log(level, message, extra={"fields": fields})
