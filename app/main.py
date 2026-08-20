"""FastAPI application construction and wiring. No business logic (ARCHITECTURE.md).

Settings, the embedding client, and the LLM client are built once here, in `lifespan`,
and handed to a `ProviderRegistry` stored on `app.state` — routes only ever read that
registry (`api/routes.py`'s dependencies), never construct a client themselves. The
registry's contents may later be replaced at runtime (`POST /settings/llm`,
`POST /settings/embedding`), which is the part of ADR-10 that decision now amends; the
`VectorStore` is not part of that registry and stays a true singleton, never rebuilt or
swapped after startup. Opening the store here is also where the ADR-8
embedding-fingerprint check runs, and a mismatch fails application startup rather than
surfacing per request (see ADR-10 for why that is the simpler, correct choice once this
module exists to own the lifecycle).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import (
    ProviderRegistry,
    Settings,
    build_embedding_model,
    build_llm,
    describe_providers,
    get_settings,
    require_credentials,
)
from app.observability import configure_logging, log_event
from app.rag.jobs import JobStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _lifespan_for(settings_factory: Callable[[], Settings]):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_factory()
        configure_logging(settings.log_level)
        embed_model = build_embedding_model(settings)
        llm = build_llm(settings)
        # The only mutable provider state in the process (ADR-10's amendment): routes
        # read `app.state.registry`, never construct a client themselves, and a runtime
        # settings update (see `api/routes.py`) replaces this object's contents only
        # after the replacement client has already been proven reachable.
        app.state.registry = ProviderRegistry(
            settings=settings, llm=llm, embed_model=embed_model
        )
        # Constructing VectorStore performs the ADR-8 fingerprint check; letting
        # EmbeddingMismatchError propagate here fails startup instead of leaving the
        # app to serve requests against an index it cannot safely read or write. The
        # store itself is never swapped or rebuilt at runtime — only the registry above
        # is mutable.
        app.state.store = VectorStore(
            path=settings.chroma_path,
            collection_name=settings.chroma_collection,
            embedding_fingerprint=settings.embedding_fingerprint,
        )
        # In-memory background-ingestion job registry (ADR-17). One per process,
        # never persisted or rebuilt at runtime — the same lifecycle as `store` above.
        app.state.job_store = JobStore()
        llm_provider, embedding_provider = describe_providers(settings)
        log_event(
            logger,
            logging.INFO,
            "startup complete",
            llm_model=llm_provider.model,
            llm_host=llm_provider.host,
            embedding_model=embedding_provider.model,
            embedding_host=embedding_provider.host,
            embedding_is_local=embedding_provider.is_local,
        )
        yield

    return lifespan


def _settings_from_environment() -> Settings:
    """The real startup path: load from the environment and fail loudly if the
    provider credentials a test never needs (stub providers) are actually blank."""
    settings = get_settings()
    require_credentials(settings)
    return settings


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. `settings` overrides the environment, for tests.

    Credential validation only runs when `settings` is omitted — tests that inject
    `Settings` directly (with stub embedding/LLM clients that never make a real
    provider call) are intentionally exempt.
    """
    settings_factory = (
        _settings_from_environment if settings is None else (lambda: settings)
    )
    app = FastAPI(
        title="Private Knowledge Assistant",
        lifespan=_lifespan_for(settings_factory),
    )
    app.include_router(router)
    return app


app = create_app()
