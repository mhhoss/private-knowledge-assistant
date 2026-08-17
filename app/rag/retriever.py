"""Query → relevant chunks, filtered by the groundedness cutoff. No generation.

`generator.py` and `engine.py` are not implemented yet; this module only answers
"what did we retrieve", never "what should the answer be" (invariant 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.documents.processor import normalize_text
from app.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk, carrying everything needed to cite it (invariant 1)."""

    text: str
    score: float
    document_id: str
    filename: str
    file_type: str
    chunk_id: str


def retrieve(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    query: str,
    top_k: int,
    min_score: float,
) -> list[RetrievedChunk]:
    """Retrieve up to `top_k` chunks for `query`, dropping any scoring below `min_score`.

    An empty result means insufficient context (ADR-4) — the engine will use it to
    trigger the deterministic refusal, never as an error condition here.

    The query is normalized with the same function as document text (invariant 7), so
    an Arabic-typed query matches Persian-typed content, and vice versa. No branch here
    depends on which language `query` or the indexed chunks are in (invariant 10).
    """
    from llama_index.core import VectorStoreIndex

    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    index = VectorStoreIndex.from_vector_store(
        store.llama_vector_store, embed_model=embed_model
    )
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(normalized_query)

    return [
        RetrievedChunk(
            text=node.node.get_content(),
            score=score,
            document_id=node.node.metadata.get("document_id", ""),
            filename=node.node.metadata.get("filename", ""),
            file_type=node.node.metadata.get("file_type", ""),
            chunk_id=node.node.metadata.get("chunk_id", ""),
        )
        for node in nodes
        # A missing score is treated as failing the cutoff, not as passing it.
        if (score := node.score if node.score is not None else 0.0) >= min_score
    ]
