"""Thin HTTP layer: validate, delegate to `rag/engine.py` and `storage/`, return schemas.

No chunking, retrieval, generation, or indexing logic lives here (invariant: "Routes
contain no chunking, retrieval, or generation logic", ARCHITECTURE.md Layers). This
module's only job is translating between the wire contracts in `schemas/api.py` and the
internal domain/storage types (`indexer.IngestOutcome`, `vector_store.DocumentSummary`,
`generator.Citation`/`GeneratedAnswer`) — never exposing the latter directly.

Dependency providers (`_get_settings`/`_get_vector_store`/`_get_embed_model`/`_get_llm`/
`_get_registry`) read `request.app.state` — the settings/embedding/LLM trio through the
`ProviderRegistry` `app/main.py` builds at startup, the `VectorStore` directly — and
never construct their own (ADR-10). `update_llm_settings`/`update_embedding_settings`
are the one exception: they build and probe a *replacement* client (still only via
`config.py`'s `build_llm`/`build_embedding_model`, invariant 5 intact) and, only on
success, hand it to the registry themselves — the registry is the single owner of that
mutation, not a general pattern for routes to construct clients. Tests that mount this
router directly (no `app/main.py` lifespan) override these dependencies to inject stubs
instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from openai import APIError

from app.config import (
    ProviderDescription,
    ProviderRegistry,
    Settings,
    build_embedding_model,
    build_llm,
    describe_providers,
    mask_secret,
    probe_embedding,
    probe_llm,
)
from app.rag import engine
from app.rag.generator import Citation as DomainCitation
from app.rag.generator import GeneratedAnswer
from app.rag.indexer import IngestOutcome as DomainIngestOutcome
from app.schemas import api as schemas
from app.storage.vector_store import DocumentSummary as DomainDocumentSummary
from app.storage.vector_store import EmbeddingMismatchError, VectorStore

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM

router = APIRouter()


# --- dependencies ---
# `_get_vector_store` reads the one `VectorStore` `app/main.py`'s `lifespan` builds at
# startup and never rebuilds — opening it there is also where the ADR-8 embedding
# fingerprint check runs, so a mismatch fails startup rather than this layer having to
# translate it into a response. The other three read the `ProviderRegistry` that same
# `lifespan` builds; unlike the store, its contents may be replaced at runtime by
# `update_llm_settings`/`update_embedding_settings` below (ADR-10's amendment).


def _get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.registry


def _get_settings(request: Request) -> Settings:
    return request.app.state.registry.settings


def _get_vector_store(request: Request) -> VectorStore:
    return request.app.state.store


def _get_embed_model(request: Request) -> BaseEmbedding:
    return request.app.state.registry.embed_model


def _get_llm(request: Request) -> LLM:
    return request.app.state.registry.llm


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


def _to_schema_provider(provider: ProviderDescription) -> schemas.ProviderSummary:
    return schemas.ProviderSummary(
        model=provider.model,
        host=provider.host,
        base_url=provider.base_url,
        masked_key=provider.masked_key,
        is_local=provider.is_local,
    )


def _sanitize_provider_error(detail: str, secret: str) -> str:
    """Redact a raw credential from a provider error before it leaves the process.

    Provider SDKs occasionally echo request details verbatim in exception messages;
    this is the only place a live-probe failure is rendered back to a client, so it is
    the one place this redaction has to happen (R-08's masking rule extended to the new
    write path).
    """
    if secret and secret in detail:
        return detail.replace(secret, mask_secret(secret))
    return detail


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


@router.get("/settings", response_model=schemas.SettingsResponse)
def read_settings(
    settings: Settings = Depends(_get_settings),
) -> schemas.SettingsResponse:
    """Report the provider configuration currently in effect, with credentials masked
    (R-08) — the environment/`.env` values resolved at startup, unless replaced since
    by `POST /settings/llm`/`POST /settings/embedding` (ADR-10's amendment). This route
    itself only reads `app.state.registry`; it never changes anything.
    """
    llm_provider, embedding_provider = describe_providers(settings)
    return schemas.SettingsResponse(
        llm=_to_schema_provider(llm_provider),
        embedding=_to_schema_provider(embedding_provider),
    )


@router.post("/settings/test", response_model=schemas.ConnectionTestResponse)
def test_providers(
    embed_model: BaseEmbedding = Depends(_get_embed_model),
    llm: LLM = Depends(_get_llm),
) -> schemas.ConnectionTestResponse:
    """Probe both configured providers with one real call each.

    Always 200: "the provider is unreachable" is the result being requested, not an
    error in the request, so it is reported in the body like an ingestion outcome.
    """
    return schemas.ConnectionTestResponse(
        llm=_check(lambda: probe_llm(llm)),
        embedding=_check(lambda: probe_embedding(embed_model)),
    )


@router.post(
    "/settings/llm",
    response_model=schemas.ProviderSummary,
    responses={502: {"model": schemas.ErrorResponse}},
)
def update_llm_settings(
    request: schemas.UpdateLlmSettingsRequest,
    registry: ProviderRegistry = Depends(_get_registry),
) -> schemas.ProviderSummary:
    """Replace the LLM provider at runtime: build -> probe -> commit only on success.

    Never written to `.env` — process-local until the app restarts, at which point
    `.env` is authoritative again. A failed probe leaves `registry` (and therefore
    every dependent route) completely unchanged; a request already in flight with the
    previous client simply finishes with it.
    """
    effective_api_key = request.api_key or registry.settings.llm_api_key
    new_settings = registry.settings.model_copy(
        update={
            "llm_api_key": effective_api_key,
            "llm_base_url": request.base_url,
            "llm_model": request.model,
        }
    )
    try:
        llm = build_llm(new_settings)
        probe_llm(llm)
    except Exception as error:
        detail = _sanitize_provider_error(
            str(error) or type(error).__name__, effective_api_key
        )
        raise HTTPException(status_code=502, detail=detail) from error

    registry.replace_llm(settings=new_settings, llm=llm)
    return _to_schema_provider(describe_providers(new_settings)[0])


@router.post(
    "/settings/embedding",
    response_model=schemas.ProviderSummary,
    responses={
        409: {"model": schemas.ErrorResponse},
        502: {"model": schemas.ErrorResponse},
    },
)
def update_embedding_settings(
    request: schemas.UpdateEmbeddingSettingsRequest,
    registry: ProviderRegistry = Depends(_get_registry),
    store: VectorStore = Depends(_get_vector_store),
) -> schemas.ProviderSummary:
    """Replace the embedding provider at runtime: build -> probe -> fingerprint-safe
    commit (ADR-8) -> only then update `registry`.

    Never written to `.env`. A failed probe, or a fingerprint conflict with documents
    already indexed under a different embedding model, leaves `registry` and the
    vector store completely unchanged — this never resets or deletes the knowledge
    base itself; that remains a separate, explicit `POST /reset` call.
    """
    effective_api_key = request.api_key or (registry.settings.embedding_api_key or "")
    new_settings = registry.settings.model_copy(
        update={
            "embedding_api_key": effective_api_key,
            "embedding_base_url": request.base_url,
            "embedding_model": request.model,
        }
    )
    try:
        embed_model = build_embedding_model(new_settings)
        probe_embedding(embed_model)
    except Exception as error:
        detail = _sanitize_provider_error(
            str(error) or type(error).__name__, effective_api_key
        )
        raise HTTPException(status_code=502, detail=detail) from error

    try:
        store.adopt_embedding_fingerprint(new_settings.embedding_fingerprint)
    except EmbeddingMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    registry.replace_embedding(settings=new_settings, embed_model=embed_model)
    return _to_schema_provider(describe_providers(new_settings)[1])


def _check(probe: Callable[[], None]) -> schemas.ConnectionCheck:
    try:
        probe()
    except Exception as error:  # noqa: BLE001 - any provider failure is a failed check
        return schemas.ConnectionCheck(
            ok=False, detail=str(error) or type(error).__name__
        )
    return schemas.ConnectionCheck(ok=True)
