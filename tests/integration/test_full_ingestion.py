"""End-to-end ingestion: file bytes → loader → parser → processor → indexer → store.

Complements `tests/integration/test_ingestion.py` (which starts from pre-built `Chunk`s)
by exercising real PDF/DOCX extraction as part of the pipeline, in English, Persian, and
mixed-language documents (R-10).
"""

from __future__ import annotations

import pytest

from app.documents.loader import UnsupportedFileTypeError, load
from app.documents.processor import process_document
from app.rag.indexer import IngestStatus, index_document
from app.storage.vector_store import VectorStore
from tests.conftest import StubEmbedding
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

PERSIAN_REPORT = (
    "این گزارش هزینه‌های زیرساخت را پوشش می‌دهد. "
    "هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت."
)
MIXED_REPORT = "تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند."


def load_and_process(*, filename: str, content: bytes):
    """The load+parse+chunk half of the pipeline a future `rag/engine.py` will run."""
    document = load(filename=filename, content=content)
    chunks = process_document(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        raw_text=document.raw_text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return document, chunks


class TestEndToEndIngestion:
    def test_english_pdf(self, store: VectorStore, embed_model: StubEmbedding) -> None:
        content = build_pdf(["The quarterly infrastructure report is final."])
        document, chunks = load_and_process(filename="report.pdf", content=content)

        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)

        assert outcome.status is IngestStatus.INDEXED
        (indexed,) = store.list_documents()
        assert indexed.document_id == document.document_id
        assert indexed.filename == "report.pdf"
        assert indexed.file_type == "pdf"

    def test_persian_pdf(self, store: VectorStore, embed_model: StubEmbedding) -> None:
        content = build_pdf([PERSIAN_REPORT])
        _document, chunks = load_and_process(filename="گزارش.pdf", content=content)
        assert chunks, "Persian text must survive extraction + normalization"

        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)

        assert outcome.status is IngestStatus.INDEXED
        (indexed,) = store.list_documents()
        assert indexed.filename == "گزارش.pdf"

    def test_mixed_language_docx(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        content = build_docx([MIXED_REPORT])
        _document, chunks = load_and_process(filename="mixed.docx", content=content)

        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)

        assert outcome.status is IngestStatus.INDEXED
        assert "Kubernetes" in chunks[0].text
        assert "PostgreSQL" in chunks[0].text

    def test_docx_table_content_is_indexed(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        content = build_docx(
            ["Infrastructure costs by team"],
            tables=[[["Team", "Cost"], ["زیرساخت", "۱۲۰"]]],
        )
        _, chunks = load_and_process(filename="costs.docx", content=content)
        combined = " ".join(chunk.text for chunk in chunks)
        assert "زیرساخت" in combined
        assert "۱۲۰" in combined

    def test_reuploading_identical_bytes_under_a_new_name_is_already_indexed(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        content = build_pdf([PERSIAN_REPORT])
        _, first_chunks = load_and_process(filename="original.pdf", content=content)
        index_document(store=store, embed_model=embed_model, chunks=first_chunks)

        _, second_chunks = load_and_process(
            filename="renamed_copy.pdf", content=content
        )
        outcome = index_document(
            store=store, embed_model=embed_model, chunks=second_chunks
        )

        assert outcome.status is IngestStatus.ALREADY_INDEXED
        (indexed,) = store.list_documents()
        assert indexed.filename == "original.pdf"  # ADR-3: first filename wins

    def test_multiple_documents_in_different_languages_coexist(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        english_content = build_pdf(["Infrastructure spending report."])
        persian_content = build_pdf([PERSIAN_REPORT])

        for filename, content in [
            ("english.pdf", english_content),
            ("گزارش.pdf", persian_content),
        ]:
            _, chunks = load_and_process(filename=filename, content=content)
            index_document(store=store, embed_model=embed_model, chunks=chunks)

        filenames = {doc.filename for doc in store.list_documents()}
        assert filenames == {"english.pdf", "گزارش.pdf"}


class TestNoExtractableText:
    """A document with no text is a caller-visible signal, not a silent empty index."""

    def test_blank_docx_yields_no_chunks(self) -> None:
        document, chunks = load_and_process(
            filename="blank.docx", content=build_docx([])
        )
        assert document.raw_text == ""
        assert chunks == []

    def test_whitespace_only_pdf_yields_no_chunks(self) -> None:
        _document, chunks = load_and_process(
            filename="scan.pdf", content=build_pdf(["   "])
        )
        assert chunks == []


class TestUnsupportedUpload:
    def test_unsupported_extension_never_reaches_the_store(
        self, store: VectorStore
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            load(filename="notes.txt", content=b"irrelevant content")
        assert store.count() == 0
