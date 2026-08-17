"""Embeds processed chunks and writes them to the vector store.

Ingestion is per-document with compensation, not transactional (ADR-7): each document is
indexed in one attempt, and any failure deletes whatever that attempt wrote before the
failure is reported, so a document is either fully indexed or absent (invariant 8).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from app.documents.processor import Chunk
from app.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding


class IngestStatus(StrEnum):
    """Per-document ingestion outcome (R-09)."""

    INDEXED = "indexed"
    ALREADY_INDEXED = "already_indexed"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestOutcome:
    """Result for one file. Carries no document text, only identity and status."""

    filename: str
    status: IngestStatus
    document_id: str | None = None
    chunk_count: int = 0
    """Chunks written by this operation; zero unless the status is `indexed`."""
    error: str | None = None

    @classmethod
    def failure(
        cls, *, filename: str, error: str, document_id: str | None = None
    ) -> IngestOutcome:
        return cls(
            filename=filename,
            status=IngestStatus.FAILED,
            document_id=document_id,
            error=error,
        )


def index_document(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    chunks: Sequence[Chunk],
) -> IngestOutcome:
    """Index one document's chunks, isolating and compensating any failure.

    `chunks` must be non-empty; a document with no extractable text is a caller-side
    failure outcome, since only the caller knows the filename to report.
    """
    if not chunks:
        raise ValueError("index_document requires at least one chunk")

    document_id = chunks[0].document_id
    filename = chunks[0].filename

    if store.document_exists(document_id):
        # Identical content, by ADR-3's content-derived id: nothing to re-embed.
        return IngestOutcome(
            filename=filename,
            status=IngestStatus.ALREADY_INDEXED,
            document_id=document_id,
        )

    try:
        _write(store=store, embed_model=embed_model, chunks=chunks)
    except Exception as error:  # noqa: BLE001 - one file's failure must not escape
        detail = str(error) or type(error).__name__
        try:
            store.delete_document(document_id)
        except Exception as cleanup_error:  # noqa: BLE001
            detail = f"{detail} (cleanup failed: {cleanup_error})"
        return IngestOutcome.failure(
            filename=filename, error=detail, document_id=document_id
        )

    return IngestOutcome(
        filename=filename,
        status=IngestStatus.INDEXED,
        document_id=document_id,
        chunk_count=len(chunks),
    )


def index_documents(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    documents: Iterable[Sequence[Chunk]],
) -> list[IngestOutcome]:
    """Index several documents independently, in order.

    An indexing failure in one document becomes a failed outcome and never stops the
    others (ADR-7). Empty chunk lists are rejected rather than skipped, so a document
    with no extractable text cannot vanish without an outcome.
    """
    return [
        index_document(store=store, embed_model=embed_model, chunks=chunks)
        for chunks in documents
    ]


def _write(
    *, store: VectorStore, embed_model: BaseEmbedding, chunks: Sequence[Chunk]
) -> None:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    metadata_keys = list(chunks[0].metadata())
    # `document_id` is a reserved key that LlamaIndex writes from the node's source
    # relationship, so the relationship — not the metadata dict — is what makes
    # deletion and listing by document work.
    source = RelatedNodeInfo(node_id=chunks[0].document_id)
    nodes = [
        TextNode(
            id_=chunk.node_id,
            text=chunk.text,
            metadata=chunk.metadata(),
            relationships={NodeRelationship.SOURCE: source},
            # Source metadata is for citation and deletion, not for the embedding or the
            # prompt; embedding it would dilute the vector with filenames and hashes.
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        )
        for chunk in chunks
    ]
    index = VectorStoreIndex.from_vector_store(
        store.llama_vector_store, embed_model=embed_model
    )
    index.insert_nodes(nodes)
