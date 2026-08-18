"""Thin HTTP layer: validate, delegate to `rag/engine.py` and `storage/`, return schemas.

No chunking, retrieval, generation, or indexing logic lives here (invariant: "Routes
contain no chunking, retrieval, or generation logic", ARCHITECTURE.md Layers). This
module's only job is translating between the wire contracts in `schemas/api.py` and the
internal domain/storage types (`indexer.IngestOutcome`, `vector_store.DocumentSummary`,
`generator.Citation`/`GeneratedAnswer`) — never exposing the latter directly.

Dependency providers (`_get_settings`/`_get_vector_store`/`_get_embed_model`/`_get_llm`)
read the shared settings/store/embedding/LLM objects `app/main.py` builds once, at
startup, off `request.app.state` — they never construct their own (ADR-10). Tests that
mount this router directly (no `app/main.py` lifespan) override these dependencies to
inject stubs instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from openai import APIError

from app.config import Settings
from app.rag import engine
from app.rag.generator import Citation as DomainCitation
from app.rag.generator import GeneratedAnswer
from app.rag.indexer import IngestOutcome as DomainIngestOutcome
from app.schemas import api as schemas
from app.storage.vector_store import DocumentSummary as DomainDocumentSummary
from app.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM

router = APIRouter()


# --- dependencies ---
# Each reads an object `app/main.py`'s `lifespan` built exactly once at startup — never
# constructed here. Opening the `VectorStore` there is also where the ADR-8 embedding
# fingerprint check runs; a mismatch fails startup rather than this layer having to
# translate it into a response (ADR-10).


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _get_vector_store(request: Request) -> VectorStore:
    return request.app.state.store


def _get_embed_model(request: Request) -> BaseEmbedding:
    return request.app.state.embed_model


def _get_llm(request: Request) -> LLM:
    return request.app.state.llm


# --- domain -> schema translation ---


def _to_schema_outcome(outcome: DomainIngestOutcome) -> schemas.IngestOutcome:
    return schemas.IngestOutcome(
        filename=outcome.filename,
        status=schemas.IngestStatus(outcome.status.value),
        document_id=outcome.document_id,
        chunk_count=outcome.chunk_count,
        error=outcome.error,
    )


def _to_schema_summary(summary: DomainDocumentSummary) -> schemas.DocumentSummary:
    return schemas.DocumentSummary(
        document_id=summary.document_id,
        filename=summary.filename,
        file_type=summary.file_type,
        chunk_count=summary.chunk_count,
    )


def _to_schema_citation(citation: DomainCitation) -> schemas.Citation:
    return schemas.Citation(
        document_id=citation.document_id,
        filename=citation.filename,
        file_type=citation.file_type,
        chunk_id=citation.chunk_id,
        excerpt=citation.excerpt,
    )


def _to_schema_answer(answer: GeneratedAnswer) -> schemas.AnswerResponse:
    return schemas.AnswerResponse(
        answer=answer.answer,
        sources=[_to_schema_citation(source) for source in answer.sources],
        is_refusal=answer.is_refusal,
    )


# --- routes ---


@router.post("/documents", response_model=schemas.IngestionResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(_get_settings),
    store: VectorStore = Depends(_get_vector_store),
    embed_model: BaseEmbedding = Depends(_get_embed_model),
) -> schemas.IngestionResponse:
    """Upload and index one or more files (R-01, R-09).

    Always 200: a per-file failure is data in `results`, never a request-level error
    (ADR-7) — a well-formed request can report every file as failed.
    """
    loaded = [((file.filename or "unnamed"), await file.read()) for file in files]
    outcomes = engine.ingest_files(
        store=store,
        embed_model=embed_model,
        files=loaded,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return schemas.IngestionResponse(results=[_to_schema_outcome(o) for o in outcomes])


@router.get("/documents", response_model=schemas.DocumentListResponse)
def list_documents(
    store: VectorStore = Depends(_get_vector_store),
) -> schemas.DocumentListResponse:
    """List indexed documents (R-06). Derived entirely from store metadata (ADR-2)."""
    return schemas.DocumentListResponse(
        documents=[_to_schema_summary(doc) for doc in store.list_documents()]
    )


@router.delete(
    "/documents/{document_id}", response_model=schemas.DeleteDocumentResponse
)
def delete_document(
    document_id: str,
    store: VectorStore = Depends(_get_vector_store),
) -> schemas.DeleteDocumentResponse:
    """Delete one document by id (R-07). Idempotent: absence is not an error."""
    existed = store.document_exists(document_id)
    if existed:
        store.delete_document(document_id)
    return schemas.DeleteDocumentResponse(document_id=document_id, deleted=existed)


@router.post("/reset", response_model=schemas.ResetResponse)
def reset_knowledge_base(
    store: VectorStore = Depends(_get_vector_store),
) -> schemas.ResetResponse:
    """Clear the entire knowledge base (R-07)."""
    store.reset()
    return schemas.ResetResponse()


@router.post(
    "/query",
    response_model=schemas.AnswerResponse,
    responses={502: {"model": schemas.ErrorResponse}},
)
def query(
    request: schemas.QueryRequest,
    settings: Settings = Depends(_get_settings),
    store: VectorStore = Depends(_get_vector_store),
    embed_model: BaseEmbedding = Depends(_get_embed_model),
    llm: LLM = Depends(_get_llm),
) -> schemas.AnswerResponse:
    """Answer a question from indexed documents only (R-04, R-05), or refuse (ADR-4).

    A provider/network failure during embedding or generation surfaces as a 502 with
    `schemas.ErrorResponse`, not a raw 500 — the request was valid, the configured
    provider just could not serve it.
    """
    try:
        result = engine.answer_query(
            store=store,
            embed_model=embed_model,
            llm=llm,
            query=request.query,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        )
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The configured LLM/embedding provider could not be reached. "
            "Please try again.",
        ) from error
    return _to_schema_answer(result)
