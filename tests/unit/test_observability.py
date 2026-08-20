"""Structured logging: format, idempotent configuration, and the no-content invariant
(ADR-15) — a logged event is metadata only, never document/query/answer text."""

from __future__ import annotations

import json
import logging

import pytest

from app.observability import _JsonFormatter, configure_logging, log_event
from tests.conftest import log_fields


@pytest.fixture
def logger() -> logging.Logger:
    """A throwaway named logger, isolated from the real app loggers and from other
    tests — `caplog` captures via the root handler regardless of this logger's own
    name, so tests never depend on load order or leak into each other."""
    return logging.getLogger("tests.observability")


class TestJsonFormatter:
    def test_formats_a_plain_record_as_one_json_object(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="something happened",
            args=(),
            exc_info=None,
        )
        payload = json.loads(_JsonFormatter().format(record))

        assert payload["message"] == "something happened"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert "timestamp" in payload

    def test_merges_structured_fields_into_the_same_object(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="document ingested",
            args=(),
            exc_info=None,
        )
        record.fields = {"document_id": "abc123", "chunk_count": 4}

        payload = json.loads(_JsonFormatter().format(record))

        assert payload["document_id"] == "abc123"
        assert payload["chunk_count"] == 4
        assert payload["message"] == "document ingested"

    def test_a_record_with_no_fields_still_formats_cleanly(self) -> None:
        record = logging.LogRecord(
            name="app.test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="plain warning",
            args=(),
            exc_info=None,
        )
        payload = json.loads(_JsonFormatter().format(record))
        assert payload["level"] == "WARNING"


class TestLogEvent:
    def test_emits_one_record_with_the_message_and_fields(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=logger.name):
            log_event(logger, logging.INFO, "query answered", retrieved_count=3)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.getMessage() == "query answered"
        assert log_fields(record) == {"retrieved_count": 3}

    def test_respects_the_given_level(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            log_event(logger, logging.WARNING, "provider request failed", route="/query")

        assert caplog.records[0].levelname == "WARNING"

    def test_never_needs_the_message_to_contain_the_field_values(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fields are carried as structured data, not string-interpolated into the
        message — this is what keeps a log line safe to write without having to
        remember to redact anything inside the message string itself."""
        with caplog.at_level(logging.INFO, logger=logger.name):
            log_event(
                logger,
                logging.INFO,
                "document ingested",
                document_id="doc-1",
                filename="report.pdf",
            )

        record = caplog.records[0]
        assert record.getMessage() == "document ingested"
        assert "doc-1" not in record.getMessage()
        assert log_fields(record)["document_id"] == "doc-1"


class TestConfigureLogging:
    def test_is_idempotent_and_never_duplicates_handlers(self) -> None:
        root = logging.getLogger()
        before = len(root.handlers)

        configure_logging("INFO")
        configure_logging("INFO")
        configure_logging("DEBUG")

        # However many handlers the first call added, a second and third call must
        # not add any more — `_CONFIGURED` makes every call after the first a no-op.
        added = len(root.handlers) - before
        assert added in (0, 1)
        configure_logging("INFO")
        assert len(root.handlers) - before == added
