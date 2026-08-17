"""Ingestion flow: document → chunks → index, with per-file outcomes (R-09).

Covers English, Persian, and mixed-language documents (R-10) against a temporary
Chroma path and a stub embedding model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.documents.processor import Chunk, process_document
from app.rag import indexer
from app.rag.indexer import IngestStatus, index_document, index_documents
from app.storage.vector_store import EmbeddingMismatchError, VectorStore
from tests.conftest import STUB_FINGERPRINT, StubEmbedding

ENGLISH_TEXT = (
    "The quarterly report covers infrastructure costs. "
    "Kubernetes cluster spending increased by twelve percent."
)
PERSIAN_TEXT = (
    "گزارش فصلی هزینه‌های زیرساخت را پوشش می‌دهد. "
    "هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت."
)
MIXED_TEXT = "تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند."


def make_chunks(
    document_id: str, filename: str, text: str, file_type: str = "pdf"
) -> list[Chunk]:
    return process_document(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        raw_text=text,
        chunk_size=120,
        chunk_overlap=20,
    )


class TestIndexing:
    @pytest.mark.parametrize(
        ("document_id", "filename", "text"),
        [
            ("en1", "report.pdf", ENGLISH_TEXT),
            ("fa1", "گزارش.pdf", PERSIAN_TEXT),
            ("mx1", "mixed.docx", MIXED_TEXT),
        ],
    )
    def test_indexes_documents_in_either_language(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        document_id: str,
        filename: str,
        text: str,
    ) -> None:
        chunks = make_chunks(document_id, filename, text)
        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)

        assert outcome.status is IngestStatus.INDEXED
        assert outcome.document_id == document_id
        assert outcome.chunk_count == len(chunks) == store.count()

        (indexed,) = store.list_documents()
        assert indexed.filename == filename
        assert indexed.chunk_count == len(chunks)

    def test_indexes_multiple_documents_independently(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        documents = [
            make_chunks("en1", "report.pdf", ENGLISH_TEXT),
            make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
        ]
        outcomes = index_documents(
            store=store, embed_model=embed_model, documents=documents
        )
        assert [outcome.status for outcome in outcomes] == [
            IngestStatus.INDEXED,
            IngestStatus.INDEXED,
        ]
        assert {doc.document_id for doc in store.list_documents()} == {"en1", "fa1"}

    def test_reindexing_identical_content_embeds_nothing(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        chunks = make_chunks("en1", "report.pdf", ENGLISH_TEXT)
        index_document(store=store, embed_model=embed_model, chunks=chunks)
        before = store.count()

        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)

        assert outcome.status is IngestStatus.ALREADY_INDEXED
        assert outcome.chunk_count == 0
        assert store.count() == before

    def test_empty_chunk_list_is_a_caller_error(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        with pytest.raises(ValueError):
            index_document(store=store, embed_model=embed_model, chunks=[])

    def test_stored_text_excludes_source_metadata(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        """Metadata is for citation and deletion, never part of the indexed text."""
        chunks = make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT)
        index_document(store=store, embed_model=embed_model, chunks=chunks)

        stored = store._collection.get(include=["documents"])
        documents = stored["documents"]
        assert documents is not None
        assert set(documents) == {chunk.text for chunk in chunks}


class TestFailureIsolation:
    @staticmethod
    def _partially_write_then_fail(
        monkeypatch: pytest.MonkeyPatch, bad_id: str
    ) -> None:
        """Make one document write its first chunk and then fail, as a provider might."""
        real_write = indexer._write

        def flaky_write(*, store, embed_model, chunks):  # type: ignore[no-untyped-def]
            if chunks[0].document_id == bad_id:
                real_write(store=store, embed_model=embed_model, chunks=chunks[:1])
                raise RuntimeError("embedding provider unavailable")
            real_write(store=store, embed_model=embed_model, chunks=chunks)

        monkeypatch.setattr(indexer, "_write", flaky_write)

    def test_one_failure_does_not_roll_back_other_files(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._partially_write_then_fail(monkeypatch, bad_id="fa1")
        documents = [
            make_chunks("en1", "report.pdf", ENGLISH_TEXT),
            make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
            make_chunks("mx1", "mixed.docx", MIXED_TEXT),
        ]

        outcomes = index_documents(
            store=store, embed_model=embed_model, documents=documents
        )

        by_name = {outcome.filename: outcome for outcome in outcomes}
        assert by_name["report.pdf"].status is IngestStatus.INDEXED
        assert by_name["mixed.docx"].status is IngestStatus.INDEXED
        assert by_name["گزارش.pdf"].status is IngestStatus.FAILED
        assert "embedding provider unavailable" in (by_name["گزارش.pdf"].error or "")
        assert {doc.document_id for doc in store.list_documents()} == {"en1", "mx1"}

    def test_failed_document_leaves_no_chunks_behind(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invariant 8: compensation removes the partial write."""
        self._partially_write_then_fail(monkeypatch, bad_id="fa1")

        outcome = index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
        )

        assert outcome.status is IngestStatus.FAILED
        assert store.document_exists("fa1") is False
        assert store.count() == 0

    def test_failure_outcome_carries_no_document_text(
        self,
        store: VectorStore,
        embed_model: StubEmbedding,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._partially_write_then_fail(monkeypatch, bad_id="fa1")
        outcome = index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
        )
        assert PERSIAN_TEXT[:20] not in (outcome.error or "")


class TestPersistenceAndDeletion:
    def test_index_survives_reopening_the_store(
        self, chroma_path: Path, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
        )
        expected = store.count()

        reopened = VectorStore(
            path=chroma_path,
            collection_name="test_kb",
            embedding_fingerprint=STUB_FINGERPRINT,
        )
        assert reopened.count() == expected
        assert reopened.document_exists("fa1")

    def test_delete_removes_only_the_target_document(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        index_documents(
            store=store,
            embed_model=embed_model,
            documents=[
                make_chunks("en1", "report.pdf", ENGLISH_TEXT),
                make_chunks("fa1", "گزارش.pdf", PERSIAN_TEXT),
            ],
        )
        store.delete_document("en1")

        assert store.document_exists("en1") is False
        assert store.document_exists("fa1") is True

    def test_deleting_an_unknown_document_is_a_no_op(self, store: VectorStore) -> None:
        store.delete_document("missing")
        assert store.count() == 0

    def test_reset_clears_everything(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("en1", "report.pdf", ENGLISH_TEXT),
        )
        store.reset()

        assert store.count() == 0
        assert store.list_documents() == []


class TestEmbeddingCompatibility:
    def test_mismatched_model_is_refused(
        self, chroma_path: Path, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("en1", "report.pdf", ENGLISH_TEXT),
        )

        with pytest.raises(EmbeddingMismatchError, match="not comparable"):
            VectorStore(
                path=chroma_path,
                collection_name="test_kb",
                embedding_fingerprint="different-model",
            )

    def test_empty_collection_adopts_the_current_model(
        self, chroma_path: Path, store: VectorStore
    ) -> None:
        reopened = VectorStore(
            path=chroma_path,
            collection_name="test_kb",
            embedding_fingerprint="different-model",
        )
        assert reopened.count() == 0

    def test_reset_allows_switching_models(
        self, chroma_path: Path, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        index_document(
            store=store,
            embed_model=embed_model,
            chunks=make_chunks("en1", "report.pdf", ENGLISH_TEXT),
        )
        store.reset()

        switched = VectorStore(
            path=chroma_path,
            collection_name="test_kb",
            embedding_fingerprint="different-model",
        )
        assert switched.count() == 0
