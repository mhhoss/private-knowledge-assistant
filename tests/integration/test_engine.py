"""Orchestration: per-file ingestion, and query → retrieve → generate → cite.

`engine.py` is the only module under test that touches every layer at once, so these
tests exercise it against real components throughout: `VectorStore`, `StubEmbedding`,
and `StubLLM` (never a real provider).
"""

from __future__ import annotations

import logging

import pytest

from app.documents.loader import UnsupportedFileTypeError
from app.rag.engine import answer_query, ingest_file, ingest_files
from app.rag.indexer import IngestStatus
from app.storage.vector_store import VectorStore
from tests.conftest import StubEmbedding, StubLLM, log_fields
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

ENGLISH_TEXT = "Kubernetes cluster costs increased significantly this quarter."
PERSIAN_TEXT = "هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت."
MIXED_TEXT = "تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند."
UNRELATED_TEXT = "The cafeteria menu changed to include more vegetarian options."


def do_ingest(
    store: VectorStore, embed_model: StubEmbedding, filename: str, content: bytes
):
    return ingest_file(
        store=store,
        embed_model=embed_model,
        filename=filename,
        content=content,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


class TestIngestFile:
    def test_indexes_an_english_pdf(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(store, embed_model, "report.pdf", build_pdf([ENGLISH_TEXT]))
        assert outcome.status is IngestStatus.INDEXED
        assert outcome.chunk_count > 0
        (doc,) = store.list_documents()
        assert doc.filename == "report.pdf"

    def test_indexes_a_persian_docx(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(
            store, embed_model, "گزارش.docx", build_docx([PERSIAN_TEXT])
        )
        assert outcome.status is IngestStatus.INDEXED
        (doc,) = store.list_documents()
        assert doc.filename == "گزارش.docx"

    def test_unsupported_extension_becomes_a_failed_outcome_not_an_exception(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(store, embed_model, "notes.txt", b"irrelevant")
        assert outcome.status is IngestStatus.FAILED
        assert outcome.filename == "notes.txt"
        assert outcome.error


    def test_corrupt_file_of_a_supported_type_becomes_a_failed_outcome(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(store, embed_model, "broken.pdf", b"not a real pdf")
        assert outcome.status is IngestStatus.FAILED
        assert store.count() == 0

    def test_document_with_no_extractable_text_is_a_failed_outcome(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(store, embed_model, "blank.docx", build_docx([]))
        assert outcome.status is IngestStatus.FAILED
        assert "No extractable text" in (outcome.error or "")
        assert outcome.document_id is not None  # identity is still known
        assert store.count() == 0

    def test_reuploading_identical_content_is_already_indexed(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        content = build_pdf([ENGLISH_TEXT])
        first = do_ingest(store, embed_model, "report.pdf", content)
        second = do_ingest(store, embed_model, "report_copy.pdf", content)

        assert first.status is IngestStatus.INDEXED
        assert second.status is IngestStatus.ALREADY_INDEXED
        (doc,) = store.list_documents()
        assert doc.filename == "report.pdf"  # ADR-3: first filename wins


# Reproduces `15_abyari_ch3.pdf`'s/`16_apartmani3_ch2.pdf`'s real broken-font
# signature (Amuzeh custom encoding): a `ToUnicode` CMap that maps glyphs to C0
# control codepoints instead of real characters, built at the byte level like every
# other PDF fixture here (see `tests/pdf_fixtures.py`) rather than injected as
# pre-extracted text. ~8.9% non-text density - above the real family's measured
# ceiling (6.4%, ADR-19), so it exercises the accepted side of the recalibrated 15%
# rejection threshold with margin on both sides: this must now index (slowly, via
# ADR-19's batch/timeout tuning), not be rejected.
_MODERATELY_GARBLED_WORD = "abcdefgh\x06ijklmnop\x18qrstuvwx\x0fyzABCDEF\x19GHIJKLMN\x1d"
MODERATELY_GARBLED_PDF_TEXT = " ".join([_MODERATELY_GARBLED_WORD] * 10)

# ~48% non-text density - far beyond anything in the real corpus, the "all-noise"
# case ADR-19 still rejects outright regardless of batch/timeout tuning.
_SEVERELY_GARBLED_WORD = "".join(f"{c}\x06" for c in "ABCDEFGHIJ")
SEVERELY_GARBLED_PDF_TEXT = " ".join([_SEVERELY_GARBLED_WORD] * 10)

GARBLED_PDF_PAGE_WIDTH = 40000  # wide enough that -layout keeps every repeated word


class TestIngestFilePathologicalText:
    def test_moderately_broken_font_pdf_matching_the_real_family_still_indexes(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(
            store,
            embed_model,
            "16_apartmani3_ch2.pdf",
            build_pdf([MODERATELY_GARBLED_PDF_TEXT], page_width=GARBLED_PDF_PAGE_WIDTH),
        )
        assert outcome.status is IngestStatus.INDEXED
        assert outcome.chunk_count > 0

    def test_severely_broken_font_pdf_becomes_a_failed_outcome_not_a_hang(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcome = do_ingest(
            store,
            embed_model,
            "broken.pdf",
            build_pdf([SEVERELY_GARBLED_PDF_TEXT], page_width=GARBLED_PDF_PAGE_WIDTH),
        )
        assert outcome.status is IngestStatus.FAILED
        assert outcome.filename == "broken.pdf"
        assert outcome.error and "corrupted" in outcome.error
        assert store.count() == 0

    def test_severe_failure_never_reaches_the_store(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        do_ingest(
            store,
            embed_model,
            "broken.pdf",
            build_pdf([SEVERELY_GARBLED_PDF_TEXT], page_width=GARBLED_PDF_PAGE_WIDTH),
        )
        assert store.list_documents() == []

    def test_a_normal_pdf_still_indexes_after_the_severely_broken_one_fails(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        do_ingest(
            store,
            embed_model,
            "broken.pdf",
            build_pdf([SEVERELY_GARBLED_PDF_TEXT], page_width=GARBLED_PDF_PAGE_WIDTH),
        )
        outcome = do_ingest(store, embed_model, "report.pdf", build_pdf([ENGLISH_TEXT]))

        assert outcome.status is IngestStatus.INDEXED
        (doc,) = store.list_documents()
        assert doc.filename == "report.pdf"


class TestIngestFiles:
    def test_files_are_ingested_independently(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcomes = ingest_files(
            store=store,
            embed_model=embed_model,
            files=[
                ("report.pdf", build_pdf([ENGLISH_TEXT])),
                ("notes.txt", b"unsupported"),
                ("گزارش.pdf", build_pdf([PERSIAN_TEXT])),
            ],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        by_name = {outcome.filename: outcome for outcome in outcomes}
        assert by_name["report.pdf"].status is IngestStatus.INDEXED
        assert by_name["notes.txt"].status is IngestStatus.FAILED
        assert by_name["گزارش.pdf"].status is IngestStatus.INDEXED
        assert {doc.filename for doc in store.list_documents()} == {
            "report.pdf",
            "گزارش.pdf",
        }

    def test_a_bad_file_never_blocks_or_rolls_back_the_others(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcomes = ingest_files(
            store=store,
            embed_model=embed_model,
            files=[
                ("a.pdf", build_pdf(["Alpha document."])),
                ("corrupt.pdf", b"garbage"),
                ("b.pdf", build_pdf(["Beta document."])),
            ],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        statuses = [o.status for o in outcomes]
        assert statuses == [
            IngestStatus.INDEXED,
            IngestStatus.FAILED,
            IngestStatus.INDEXED,
        ]
        assert store.count() > 0

    def test_empty_file_list_returns_empty_outcome_list(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        outcomes = ingest_files(
            store=store,
            embed_model=embed_model,
            files=[],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        assert outcomes == []


class TestAnswerQuery:
    def test_answers_from_relevant_english_context(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        do_ingest(store, embed_model, "report.pdf", build_pdf([ENGLISH_TEXT]))
        do_ingest(store, embed_model, "unrelated.pdf", build_pdf([UNRELATED_TEXT]))
        llm.response = "Kubernetes costs rose significantly."

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="How did Kubernetes cluster costs change?",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is False
        assert result.answer == "Kubernetes costs rose significantly."
        assert any(source.filename == "report.pdf" for source in result.sources)
        assert llm.call_count == 1

    def test_answers_from_relevant_persian_context(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        do_ingest(store, embed_model, "گزارش.pdf", build_pdf([PERSIAN_TEXT]))
        llm.response = "هزینه دوازده درصد افزایش یافت."

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="هزینه خوشه کوبرنتیز چقدر تغییر کرد؟",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is False
        assert result.sources[0].filename == "گزارش.pdf"

    def test_answers_from_mixed_language_context(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        do_ingest(store, embed_model, "mixed.docx", build_docx([MIXED_TEXT]))
        llm.response = "تیم زیرساخت از Kubernetes استفاده می‌کند."

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="زیرساخت از چه چیزی برای Kubernetes استفاده می‌کند؟",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is False
        assert result.sources[0].document_id

    def test_empty_knowledge_base_refuses_without_calling_the_llm(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        llm.response = "this must never be seen"

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="Anything at all?",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is True
        assert result.sources == []
        assert llm.call_count == 0

    def test_high_cutoff_refuses_without_calling_the_llm(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        """A `min_score` no candidate can clear reproduces the retrieval-only refusal
        path end to end: the cutoff empties retrieval, and `generate()` never calls
        the LLM for empty context (ADR-4)."""
        do_ingest(store, embed_model, "unrelated.pdf", build_pdf([UNRELATED_TEXT]))
        llm.response = "this must never be seen"

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.999,
        )

        assert result.is_refusal is True
        assert llm.call_count == 0

    def test_model_signaled_refusal_still_calls_the_llm_but_drops_sources(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        """Weak-but-passing context (ADR-4's prompt-level layer) does reach the LLM."""
        do_ingest(store, embed_model, "unrelated.pdf", build_pdf([UNRELATED_TEXT]))
        llm.response = "[[INSUFFICIENT_CONTEXT]]"

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is True
        assert result.sources == []
        assert llm.call_count == 1

    def test_top_k_and_min_score_are_forwarded_to_retrieval(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        do_ingest(
            store, embed_model, "a.pdf", build_pdf(["Alpha document about servers."])
        )
        do_ingest(
            store, embed_model, "b.pdf", build_pdf(["Beta document about servers."])
        )
        do_ingest(
            store, embed_model, "c.pdf", build_pdf(["Gamma document about servers."])
        )
        llm.response = "answer"

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="servers",
            top_k=2,
            min_score=0.0,
        )

        assert len(result.sources) == 2


class TestEndToEndIngestThenQuery:
    def test_a_freshly_ingested_document_is_immediately_queryable(
        self, store: VectorStore, embed_model: StubEmbedding, llm: StubLLM
    ) -> None:
        outcome = do_ingest(store, embed_model, "گزارش.pdf", build_pdf([PERSIAN_TEXT]))
        assert outcome.status is IngestStatus.INDEXED
        llm.response = "پاسخ نهایی."

        result = answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query="کوبرنتیز",
            top_k=5,
            min_score=0.0,
        )

        assert result.is_refusal is False
        assert result.sources[0].document_id == outcome.document_id


class TestNoDuplicatedLayerAccess:
    def test_engine_module_imports_only_documented_collaborators(self) -> None:
        import ast
        from pathlib import Path

        source = Path("app/rag/engine.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "chromadb" not in modules
        # engine.py may know about every domain/infra module; it must not reach past
        # them into transport/UI concerns.
        forbidden_prefixes = ("fastapi", "streamlit", "httpx", "app.api", "app.schemas")
        assert not any(m.startswith(forbidden_prefixes) for m in modules)


class TestUnsupportedFileTypeIsHandledNotRaised:
    def test_ingest_file_never_raises_unsupported_file_type_error(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        # ingest_file must convert this to an outcome, unlike documents.loader.load
        # which raises it directly (verified as a contrast, not a redundant check).
        try:
            outcome = do_ingest(store, embed_model, "x.bmp", b"whatever")
        except UnsupportedFileTypeError:
            raise AssertionError(
                "engine.ingest_file must not let this escape"
            ) from None
        assert outcome.status is IngestStatus.FAILED


class TestIngestionIsLogged:
    def test_a_successful_ingestion_logs_status_and_identity(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.engine"):
            outcome = do_ingest(store, embed_model, "report.pdf", build_pdf([ENGLISH_TEXT]))

        assert outcome.status is IngestStatus.INDEXED
        records = [r for r in caplog.records if r.name == "app.rag.engine"]
        assert len(records) == 1
        assert records[0].getMessage() == "document ingested"
        assert log_fields(records[0])["status"] == "indexed"
        assert log_fields(records[0])["document_id"] == outcome.document_id
        assert log_fields(records[0])["chunk_count"] == outcome.chunk_count

    def test_a_failed_ingestion_is_logged_at_warning_without_leaking_document_text(
        self, store: VectorStore, embed_model: StubEmbedding, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.engine"):
            outcome = do_ingest(store, embed_model, "x.bmp", b"whatever")

        assert outcome.status is IngestStatus.FAILED
        record = next(r for r in caplog.records if r.name == "app.rag.engine")
        assert record.levelname == "WARNING"
        assert log_fields(record)["status"] == "failed"
        # The error message is the same one already deemed safe to show the user
        # (R-09) — never the file's own bytes/text.
        assert log_fields(record)["error"] == outcome.error

    def test_already_indexed_is_logged_as_its_own_status_not_as_a_failure(
        self, store: VectorStore, embed_model: StubEmbedding, caplog: pytest.LogCaptureFixture
    ) -> None:
        do_ingest(store, embed_model, "a.pdf", build_pdf([ENGLISH_TEXT]))
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="app.rag.engine"):
            outcome = do_ingest(store, embed_model, "a.pdf", build_pdf([ENGLISH_TEXT]))

        assert outcome.status is IngestStatus.ALREADY_INDEXED
        record = next(r for r in caplog.records if r.name == "app.rag.engine")
        assert record.levelname == "INFO"
        assert log_fields(record)["status"] == "already_indexed"


class TestQueryIsLogged:
    def test_a_query_logs_timing_and_retrieval_outcome_never_the_text(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        llm: StubLLM,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        do_ingest(store, embed_model, "report.pdf", build_pdf([ENGLISH_TEXT]))
        llm.response = "Costs rose [1]."
        secret_query = "What happened to Kubernetes costs this quarter?"

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="app.rag.engine"):
            result = answer_query(
                store=store,
                embed_model=embed_model,
                llm=llm,
                query=secret_query,
                top_k=5,
                min_score=0.0,
            )

        record = next(r for r in caplog.records if r.name == "app.rag.engine")
        assert record.getMessage() == "query answered"
        assert log_fields(record)["is_refusal"] is result.is_refusal
        assert log_fields(record)["source_count"] == len(result.sources)
        assert log_fields(record)["retrieved_count"] >= 1
        assert isinstance(log_fields(record)["elapsed_ms"], float)
        assert log_fields(record)["elapsed_ms"] >= 0
        # Neither the question nor the generated answer text ever appears in the log.
        assert secret_query not in json_safe(record)
        assert result.answer not in json_safe(record)

    def test_a_refusal_is_logged_with_zero_sources(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        llm: StubLLM,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.engine"):
            result = answer_query(
                store=store,
                embed_model=embed_model,
                llm=llm,
                query="anything",
                top_k=5,
                min_score=0.0,
            )

        assert result.is_refusal is True
        record = next(r for r in caplog.records if r.name == "app.rag.engine")
        assert log_fields(record)["is_refusal"] is True
        assert log_fields(record)["source_count"] == 0
        assert log_fields(record)["retrieved_count"] == 0
        assert log_fields(record)["min_score"] is None
        assert log_fields(record)["max_score"] is None


def json_safe(record: logging.LogRecord) -> str:
    """Every string a log line could actually surface: the rendered message plus its
    structured field values, joined for a single substring check."""
    return " ".join(
        [record.getMessage(), *(str(v) for v in log_fields(record).values())]
    )
