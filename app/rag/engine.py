"""Orchestrates the two flows this system runs: per-file ingestion, and
query → retrieve → generate → cite.

The only module that sequences other components (invariant 3). `documents/`,
`rag/indexer.py`, `rag/retriever.py`, and `rag/generator.py` never call each other or
the vector store directly outside their own defined boundary — this module threads
their inputs and outputs together without duplicating their internal logic: a
load/parse failure or an empty extraction becomes an `indexer.IngestOutcome` here
(ADR-7), and a `retriever.RetrievedChunk` becomes a `generator.ContextChunk` here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from app.documents.loader import UnsupportedFileTypeError, load
from app.documents.parser import ParsingError
from app.documents.processor import process_document
from app.rag.generator import ContextChunk, GeneratedAnswer, generate
from app.rag.indexer import IngestOutcome, index_document
from app.rag.retriever import retrieve
from app.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM


def ingest_file(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    filename: str,
    content: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestOutcome:
    """Run one uploaded file through load → parse → chunk → index (R-09).

    Every failure mode — an unsupported extension, an unreadable file, no extractable
    text, or an embedding/store failure during indexing — becomes a failed
    `IngestOutcome` for this file. This function never raises, and a failure here never
    affects any other file (ADR-7).
    """
    try:
        document = load(filename=filename, content=content)
    except (UnsupportedFileTypeError, ParsingError) as error:
        return IngestOutcome.failure(filename=filename, error=str(error))

    chunks = process_document(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        raw_text=document.raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        # A document with no extractable text (e.g. a scanned/image-only PDF) is a
        # caller error to `index_document`, since only the caller knows the filename
        # to report — this is that caller (ADR-7).
        return IngestOutcome.failure(
            filename=document.filename,
            error="No extractable text was found in this file.",
            document_id=document.document_id,
        )

    return index_document(store=store, embed_model=embed_model, chunks=chunks)


def ingest_files(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    files: Iterable[tuple[str, bytes]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[IngestOutcome]:
    """Ingest several `(filename, content)` files independently, in order (ADR-7).

    One file's failure never rolls back or blocks another's success — a well-formed
    upload request can succeed even if every file in it fails; per-file failures are
    data in the returned list, not raised errors.
    """
    return [
        ingest_file(
            store=store,
            embed_model=embed_model,
            filename=filename,
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for filename, content in files
    ]


def answer_query(
    *,
    store: VectorStore,
    embed_model: BaseEmbedding,
    llm: LLM,
    query: str,
    top_k: int,
    min_score: float,
) -> GeneratedAnswer:
    """Query → Retriever → Context → Generator → Answer + Sources.

    An empty or below-cutoff retrieval result is passed straight through to
    `generate()`, which refuses without calling the LLM (ADR-4) — this function does
    not duplicate that check; it only adapts `RetrievedChunk` to `ContextChunk`, since
    the two modules deliberately do not know about each other's types.
    """
    retrieved = retrieve(
        store=store,
        embed_model=embed_model,
        query=query,
        top_k=top_k,
        min_score=min_score,
    )
    context = [
        ContextChunk(
            text=chunk.text,
            document_id=chunk.document_id,
            filename=chunk.filename,
            file_type=chunk.file_type,
            chunk_id=chunk.chunk_id,
        )
        for chunk in retrieved
    ]
    return generate(query=query, chunks=context, llm=llm)
