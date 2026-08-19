"""`VectorStore`'s embedding-fingerprint bookkeeping (ADR-8), including the runtime
`adopt_embedding_fingerprint` path a settings update relies on to stay index-safe."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.documents.processor import process_document
from app.rag.indexer import index_document
from app.storage.vector_store import EmbeddingMismatchError, VectorStore
from tests.conftest import STUB_FINGERPRINT, StubEmbedding


def _index_one_document(store: VectorStore, embed_model: StubEmbedding) -> None:
    chunks = process_document(
        document_id="doc-1",
        filename="report.pdf",
        file_type="pdf",
        raw_text="The quarterly report covers infrastructure costs.",
        chunk_size=200,
        chunk_overlap=20,
    )
    outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)
    assert outcome.status.value == "indexed"


class TestEmbeddingFingerprintProperty:
    def test_reports_the_fingerprint_the_store_was_opened_with(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(
            path=tmp_path / "chroma",
            collection_name="test_kb",
            embedding_fingerprint="model-a",
        )
        assert store.embedding_fingerprint == "model-a"


class TestAdoptEmbeddingFingerprint:
    def test_is_a_no_op_when_the_fingerprint_already_matches(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(
            path=tmp_path / "chroma",
            collection_name="test_kb",
            embedding_fingerprint=STUB_FINGERPRINT,
        )
        store.adopt_embedding_fingerprint(STUB_FINGERPRINT)
        assert store.embedding_fingerprint == STUB_FINGERPRINT

    def test_adopts_a_new_fingerprint_when_the_collection_is_empty(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(
            path=tmp_path / "chroma", collection_name="test_kb", embedding_fingerprint="old"
        )
        store.adopt_embedding_fingerprint("new")
        assert store.embedding_fingerprint == "new"
        assert store.count() == 0

    def test_a_document_can_still_be_indexed_after_adopting_a_new_fingerprint(
        self, tmp_path: Path
    ) -> None:
        """Adoption must leave a genuinely usable collection, not just flip a flag."""
        store = VectorStore(
            path=tmp_path / "chroma", collection_name="test_kb", embedding_fingerprint="old"
        )
        store.adopt_embedding_fingerprint("new")
        _index_one_document(store, StubEmbedding())
        assert store.count() > 0

    def test_refuses_when_chunks_exist_under_a_different_model(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(
            path=tmp_path / "chroma",
            collection_name="test_kb",
            embedding_fingerprint=STUB_FINGERPRINT,
        )
        _index_one_document(store, StubEmbedding())

        with pytest.raises(EmbeddingMismatchError, match=STUB_FINGERPRINT):
            store.adopt_embedding_fingerprint("a-different-model")

        # Refused, not partially applied (ADR-8): both the fingerprint and the
        # existing data must be exactly as they were before the call.
        assert store.embedding_fingerprint == STUB_FINGERPRINT
        assert store.count() == 1

    def test_is_a_no_op_when_fingerprint_matches_even_with_existing_chunks(
        self, tmp_path: Path
    ) -> None:
        store = VectorStore(
            path=tmp_path / "chroma",
            collection_name="test_kb",
            embedding_fingerprint=STUB_FINGERPRINT,
        )
        _index_one_document(store, StubEmbedding())

        store.adopt_embedding_fingerprint(STUB_FINGERPRINT)

        assert store.count() == 1
