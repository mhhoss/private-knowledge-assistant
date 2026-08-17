"""Retrieval: query → relevant chunks → groundedness cutoff (R-04, ADR-4).

Uses the real `VectorStore` and the deterministic `StubEmbedding`, exercised through the
actual ingestion path so retrieval is tested against genuinely indexed chunks, not
hand-inserted vectors.
"""

from __future__ import annotations

from app.documents.processor import process_document
from app.rag.indexer import index_document
from app.rag.retriever import RetrievedChunk, retrieve
from app.storage.vector_store import VectorStore
from tests.conftest import StubEmbedding

KUBERNETES_EN = "Kubernetes cluster costs increased significantly this quarter."
KUBERNETES_FA = "هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت."
UNRELATED_EN = "The cafeteria menu changed to include more vegetarian options."


def seed(
    store: VectorStore, embed_model: StubEmbedding, documents: list[tuple]
) -> None:
    """Index a set of (document_id, filename, file_type, text) documents."""
    for document_id, filename, file_type, text in documents:
        chunks = process_document(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            raw_text=text,
            chunk_size=500,
            chunk_overlap=0,
        )
        outcome = index_document(store=store, embed_model=embed_model, chunks=chunks)
        assert outcome.status.value == "indexed"


class TestRelevantRetrieval:
    def test_retrieves_relevant_english_chunk(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(
            store,
            embed_model,
            [
                ("en1", "report.pdf", "pdf", KUBERNETES_EN),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.0,
        )

        assert results
        assert results[0].document_id == "en1"
        assert results[0].filename == "report.pdf"

    def test_retrieves_relevant_persian_chunk(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(
            store,
            embed_model,
            [
                ("fa1", "گزارش.pdf", "pdf", KUBERNETES_FA),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="هزینه خوشه کوبرنتیز",
            top_k=5,
            min_score=0.0,
        )

        assert results
        assert results[0].document_id == "fa1"
        assert results[0].filename == "گزارش.pdf"

    def test_retrieves_relevant_chunk_for_mixed_language_query(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        mixed_doc = "تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند."
        seed(
            store,
            embed_model,
            [
                ("mx1", "mixed.docx", "docx", mixed_doc),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes زیرساخت",
            top_k=5,
            min_score=0.0,
        )

        assert results
        assert results[0].document_id == "mx1"

    def test_cross_lingual_query_still_matches_via_shared_english_tokens(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        """A Persian query containing an English term matches an English document."""
        seed(
            store,
            embed_model,
            [
                ("en1", "report.pdf", "pdf", KUBERNETES_EN),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="هزینه Kubernetes چقدر است",
            top_k=5,
            min_score=0.0,
        )

        assert results
        assert results[0].document_id == "en1"


class TestScoreCutoff:
    def test_min_score_zero_keeps_every_candidate(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(
            store,
            embed_model,
            [
                ("en1", "report.pdf", "pdf", KUBERNETES_EN),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.0,
        )
        assert len(results) == 2

    def test_high_min_score_drops_weak_matches(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(
            store,
            embed_model,
            [
                ("en1", "report.pdf", "pdf", KUBERNETES_EN),
                ("other", "unrelated.pdf", "pdf", UNRELATED_EN),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.99,
        )
        assert results == []

    def test_cutoff_is_deterministic_across_repeated_calls(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(store, embed_model, [("en1", "report.pdf", "pdf", KUBERNETES_EN)])

        first = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.5,
        )
        second = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.5,
        )
        assert first == second

    def test_cutoff_uses_greater_or_equal(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        """A chunk scoring exactly at the threshold is kept, not dropped."""
        seed(store, embed_model, [("en1", "report.pdf", "pdf", KUBERNETES_EN)])
        baseline = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=1,
            min_score=0.0,
        )
        threshold = baseline[0].score

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=1,
            min_score=threshold,
        )
        assert len(results) == 1


class TestNoResults:
    def test_empty_store_returns_no_results(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="anything at all",
            top_k=5,
            min_score=0.0,
        )
        assert results == []

    def test_blank_query_returns_no_results_without_querying_the_store(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(store, embed_model, [("en1", "report.pdf", "pdf", KUBERNETES_EN)])
        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="   ",
            top_k=5,
            min_score=0.0,
        )
        assert results == []

    def test_everything_below_cutoff_yields_empty_list_not_an_error(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(store, embed_model, [("other", "unrelated.pdf", "pdf", UNRELATED_EN)])
        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="Kubernetes cluster costs",
            top_k=5,
            min_score=0.999,
        )
        assert results == []


class TestMetadataPreservation:
    def test_result_carries_full_citation_metadata(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(store, embed_model, [("fa1", "گزارش.pdf", "pdf", KUBERNETES_FA)])

        (result,) = retrieve(
            store=store,
            embed_model=embed_model,
            query="کوبرنتیز",
            top_k=5,
            min_score=0.0,
        )

        assert isinstance(result, RetrievedChunk)
        assert result.document_id == "fa1"
        assert result.filename == "گزارش.pdf"
        assert result.file_type == "pdf"
        assert result.chunk_id == "0000"
        assert result.text == KUBERNETES_FA
        assert isinstance(result.score, float)

    def test_top_k_limits_result_count(
        self, store: VectorStore, embed_model: StubEmbedding
    ) -> None:
        seed(
            store,
            embed_model,
            [
                ("en1", "a.pdf", "pdf", "Alpha document about servers."),
                ("en2", "b.pdf", "pdf", "Beta document about servers."),
                ("en3", "c.pdf", "pdf", "Gamma document about servers."),
            ],
        )

        results = retrieve(
            store=store,
            embed_model=embed_model,
            query="servers",
            top_k=2,
            min_score=0.0,
        )
        assert len(results) == 2


class TestEmbeddingFingerprintCompatibility:
    def test_retrieval_respects_the_fingerprint_check_at_store_open(
        self, chroma_path, embed_model: StubEmbedding
    ) -> None:
        """Retrieval never runs against a store opened with a mismatched fingerprint —
        `VectorStore.__init__` already refuses to open one (invariant 9); this pins
        that retrieval performs no fingerprint check of its own and relies on it.
        """
        import pytest

        from app.storage.vector_store import EmbeddingMismatchError

        store = VectorStore(
            path=chroma_path,
            collection_name="fp_kb",
            embedding_fingerprint="model-a",
        )
        seed(store, embed_model, [("en1", "report.pdf", "pdf", KUBERNETES_EN)])

        with pytest.raises(EmbeddingMismatchError):
            VectorStore(
                path=chroma_path,
                collection_name="fp_kb",
                embedding_fingerprint="model-b",
            )
