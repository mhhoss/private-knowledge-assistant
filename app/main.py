"""FastAPI application construction and wiring. No business logic (ARCHITECTURE.md).

Settings, the embedding client, the LLM client, and the `VectorStore` are each built
once, in `lifespan`, and stored on `app.state` — not reconstructed per request. This is
the app-scoped-singleton half of ADR-10: opening the store here, at startup, is also
where the ADR-8 embedding-fingerprint check now runs, and a mismatch fails application
startup rather than surfacing per request (see ADR-10 for why that is now the simpler,
correct choice once this module exists to own the lifecycle).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, build_embedding_model, build_llm, get_settings
from app.storage.vector_store import VectorStore


def _lifespan_for(settings_factory: Callable[[], Settings]):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_factory()
        app.state.settings = settings
        app.state.embed_model = build_embedding_model(settings)
        app.state.llm = build_llm(settings)
        # Constructing VectorStore performs the ADR-8 fingerprint check; letting
        # EmbeddingMismatchError propagate here fails startup instead of leaving the
        # app to serve requests against an index it cannot safely read or write.
        app.state.store = VectorStore(
            path=settings.chroma_path,
            collection_name=settings.chroma_collection,
            embedding_fingerprint=settings.embedding_fingerprint,
        )
        yield

    return lifespan


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. `settings` overrides the environment, for tests."""
    settings_factory = get_settings if settings is None else (lambda: settings)
    app = FastAPI(
        title="Private Knowledge Assistant",
        lifespan=_lifespan_for(settings_factory),
    )
    app.include_router(router)
    return app


app = create_app()
