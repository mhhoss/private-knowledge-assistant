"""Public API contracts (`app/schemas/api.py`) — independent of FastAPI and of the
domain/storage types they mirror. Every schema is exercised in isolation, by
construction, never by routing a request through the app.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas import api as schemas


def errors_of(exc_info: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    return [error["msg"] for error in exc_info.value.errors()]


class TestIngestOutcome:
    def test_indexed_outcome_is_valid(self) -> None:
        outcome = schemas.IngestOutcome(
            filename="report.pdf",
            status=schemas.IngestStatus.INDEXED,
            document_id="abc123",
            chunk_count=4,
        )
        assert outcome.error is None

    def test_already_indexed_outcome_is_valid_with_zero_chunks(self) -> None:
        schemas.IngestOutcome(
            filename="report.pdf",
            status=schemas.IngestStatus.ALREADY_INDEXED,
            document_id="abc123",
        )

    def test_failed_outcome_requires_an_error_message(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            schemas.IngestOutcome(filename="broken.pdf", status=schemas.IngestStatus.FAILED)
        assert "error is required" in " ".join(errors_of(exc_info))

    def test_failed_outcome_may_have_no_document_id(self) -> None:
        """Identity is unknown when loading fails before the content hash is computed."""
        outcome = schemas.IngestOutcome(
            filename="notes.txt",
            status=schemas.IngestStatus.FAILED,
            error="Unsupported file type",
        )
        assert outcome.document_id is None

    def test_non_failed_outcome_rejects_an_error_message(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            schemas.IngestOutcome(
                filename="report.pdf",
                status=schemas.IngestStatus.INDEXED,
                error="should not be here",
            )
        assert "must be unset" in " ".join(errors_of(exc_info))

    def test_chunk_count_must_be_zero_unless_indexed(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            schemas.IngestOutcome(
                filename="report.pdf",
                status=schemas.IngestStatus.ALREADY_INDEXED,
                chunk_count=3,
            )
        assert "chunk_count must be 0" in " ".join(errors_of(exc_info))

    def test_chunk_count_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            schemas.IngestOutcome(
                filename="report.pdf", status=schemas.IngestStatus.INDEXED, chunk_count=-1
            )

    def test_blank_filename_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            schemas.IngestOutcome(filename="", status=schemas.IngestStatus.FAILED, error="x")

    def test_persian_filename_round_trips(self) -> None:
        outcome = schemas.IngestOutcome(
            filename="گزارش.pdf",
            status=schemas.IngestStatus.INDEXED,
            document_id="fa1",
            chunk_count=2,
        )
        assert outcome.model_dump()["filename"] == "گزارش.pdf"

    def test_status_accepts_the_same_string_values_as_the_domain_enum(self) -> None:
        from app.rag.indexer import IngestStatus as DomainIngestStatus

        assert {status.value for status in schemas.IngestStatus} == {
            status.value for status in DomainIngestStatus
        }


class TestIngestionResponse:
    def test_wraps_multiple_independent_outcomes(self) -> None:
        response = schemas.IngestionResponse(
            results=[
                schemas.IngestOutcome(
                    filename="a.pdf", status=schemas.IngestStatus.INDEXED, chunk_count=1
                ),
                schemas.IngestOutcome(
                    filename="b.txt", status=schemas.IngestStatus.FAILED, error="bad type"
                ),
            ]
        )
        assert len(response.results) == 2

    def test_empty_results_is_valid(self) -> None:
        schemas.IngestionResponse(results=[])


class TestDocumentSummary:
    def test_valid_summary(self) -> None:
        schemas.DocumentSummary(
            document_id="fa1", filename="گزارش.pdf", file_type="pdf", chunk_count=3
        )

    def test_chunk_count_must_be_at_least_one(self) -> None:
        """Invariant 8: a listed document is never partial."""
        with pytest.raises(ValidationError):
            schemas.DocumentSummary(
                document_id="d", filename="f.pdf", file_type="pdf", chunk_count=0
            )

    @pytest.mark.parametrize("field", ["document_id", "filename", "file_type"])
    def test_blank_identity_fields_are_rejected(self, field: str) -> None:
        values = {
            "document_id": "d",
            "filename": "f.pdf",
            "file_type": "pdf",
            "chunk_count": 1,
        }
        values[field] = ""
        with pytest.raises(ValidationError):
            schemas.DocumentSummary(**values)


class TestDocumentListResponse:
    def test_empty_knowledge_base_is_valid(self) -> None:
        schemas.DocumentListResponse(documents=[])

    def test_lists_documents_regardless_of_language(self) -> None:
        response = schemas.DocumentListResponse(
            documents=[
                schemas.DocumentSummary(
                    document_id="en1", filename="report.pdf", file_type="pdf", chunk_count=2
                ),
                schemas.DocumentSummary(
                    document_id="fa1", filename="گزارش.pdf", file_type="pdf", chunk_count=1
                ),
            ]
        )
        assert {doc.filename for doc in response.documents} == {"report.pdf", "گزارش.pdf"}


class TestDeleteDocumentResponse:
    def test_deleted_true_when_something_was_removed(self) -> None:
        schemas.DeleteDocumentResponse(document_id="d1", deleted=True)

    def test_deleted_false_is_valid_not_an_error(self) -> None:
        """Deletion is idempotent: nothing to remove is a normal outcome (R-07)."""
        schemas.DeleteDocumentResponse(document_id="missing", deleted=False)

    def test_blank_document_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            schemas.DeleteDocumentResponse(document_id="", deleted=True)


class TestResetResponse:
    def test_has_a_default_message(self) -> None:
        assert schemas.ResetResponse().message


class TestQueryRequest:
    def test_english_query_is_valid(self) -> None:
        schemas.QueryRequest(query="What were the Q4 costs?")

    def test_persian_query_is_valid(self) -> None:
        schemas.QueryRequest(query="هزینه‌های سه‌ماهه چهارم چقدر بود؟")

    def test_mixed_language_query_is_valid(self) -> None:
        schemas.QueryRequest(query="هزینه Kubernetes چقدر بود؟")

    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            schemas.QueryRequest(query="")

    def test_whitespace_only_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            schemas.QueryRequest(query="   \n\t  ")

    def test_query_content_is_not_otherwise_transformed(self) -> None:
        """Normalization is a retrieval/generation concern (invariant 7), not a
        request-validation one — the schema must not mutate the query text."""
        raw = "  What about Kubernetes?  "
        assert schemas.QueryRequest(query=raw).query == raw


class TestCitation:
    def test_valid_citation(self) -> None:
        schemas.Citation(
            document_id="fa1",
            filename="گزارش.pdf",
            file_type="pdf",
            chunk_id="0001",
            excerpt="هزینه دوازده درصد افزایش یافت.",
        )

    @pytest.mark.parametrize("field", ["document_id", "filename", "file_type", "chunk_id"])
    def test_blank_identity_fields_are_rejected(self, field: str) -> None:
        values = {
            "document_id": "d",
            "filename": "f.pdf",
            "file_type": "pdf",
            "chunk_id": "0000",
            "excerpt": "text",
        }
        values[field] = ""
        with pytest.raises(ValidationError):
            schemas.Citation(**values)

    def test_excerpt_may_be_any_length_including_empty(self) -> None:
        """Excerpt fidelity, not shape, is the requirement — no length constraint."""
        schemas.Citation(
            document_id="d", filename="f.pdf", file_type="pdf", chunk_id="0000", excerpt=""
        )


class TestAnswerResponse:
    def _citation(self) -> schemas.Citation:
        return schemas.Citation(
            document_id="d", filename="f.pdf", file_type="pdf", chunk_id="0000", excerpt="x"
        )

    def test_grounded_answer_with_sources_is_valid(self) -> None:
        schemas.AnswerResponse(answer="12%.", sources=[self._citation()], is_refusal=False)

    def test_refusal_with_no_sources_is_valid(self) -> None:
        schemas.AnswerResponse(
            answer="I don't have enough information.", sources=[], is_refusal=True
        )

    def test_refusal_cannot_carry_sources(self) -> None:
        """Groundedness rule 4: a refusal has nothing to cite."""
        with pytest.raises(ValidationError) as exc_info:
            schemas.AnswerResponse(
                answer="refused", sources=[self._citation()], is_refusal=True
            )
        assert "must not carry sources" in " ".join(errors_of(exc_info))

    def test_non_refusal_must_carry_at_least_one_source(self) -> None:
        """Groundedness rule 4: every real answer must cite something."""
        with pytest.raises(ValidationError) as exc_info:
            schemas.AnswerResponse(answer="12%.", sources=[], is_refusal=False)
        assert "must carry at least one source" in " ".join(errors_of(exc_info))

    def test_multiple_sources_across_documents_are_valid(self) -> None:
        other = schemas.Citation(
            document_id="d2", filename="g.pdf", file_type="pdf", chunk_id="0000", excerpt="y"
        )
        schemas.AnswerResponse(
            answer="combined", sources=[self._citation(), other], is_refusal=False
        )

    def test_persian_answer_and_citation_round_trip(self) -> None:
        citation = schemas.Citation(
            document_id="fa1",
            filename="گزارش.pdf",
            file_type="pdf",
            chunk_id="0000",
            excerpt="هزینه دوازده درصد افزایش یافت.",
        )
        response = schemas.AnswerResponse(
            answer="هزینه دوازده درصد افزایش یافت.", sources=[citation], is_refusal=False
        )
        dumped = response.model_dump()
        assert dumped["sources"][0]["filename"] == "گزارش.pdf"


class TestErrorResponse:
    def test_valid_error(self) -> None:
        schemas.ErrorResponse(detail="Document not found.")

    def test_blank_detail_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ErrorResponse(detail="")


class TestSchemasAreIndependentOfFastApiAndDomainTypes:
    def test_module_does_not_import_fastapi(self) -> None:
        import ast
        from pathlib import Path

        source = Path("app/schemas/api.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        forbidden = {
            "fastapi",
            "chromadb",
            "app.storage.vector_store",
            "app.rag.indexer",
            "app.rag.retriever",
            "app.rag.generator",
            "app.rag.engine",
        }
        assert not (modules & forbidden), modules & forbidden

    def test_every_public_schema_is_a_plain_pydantic_model(self) -> None:
        public_names = [name for name in dir(schemas) if not name.startswith("_")]
        model_classes = [
            getattr(schemas, name)
            for name in public_names
            if isinstance(getattr(schemas, name), type)
            and issubclass(getattr(schemas, name), BaseModel)
        ]
        assert model_classes, "expected at least one schema class"
        for model_class in model_classes:
            assert BaseModel in model_class.__mro__
