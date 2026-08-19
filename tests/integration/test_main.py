"""Integration tests for `app/main.py`: app creation, wiring, and lifecycle.

`create_app(settings=...)` lets these tests control configuration without touching the
environment or `.env` (off limits per CLAUDE.md); real provider clients are constructed
(`build_embedding_model`/`build_llm`), but never called — no test here sends a chat or
embedding request. Entering `TestClient(app)` as a context manager runs `lifespan`, so
a startup failure (e.g. `EmbeddingMismatchError`) surfaces by raising out of `__enter__`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.storage.vector_store import EmbeddingMismatchError, VectorStore
from tests.pdf_fixtures import build_pdf

# Real OpenAI-recognized model names: `build_embedding_model`/`build_llm` construct
# actual provider client objects (never called over the network in these tests), and
# the client constructors validate the model name against a known-model enum.
EMBEDDING_MODEL_A = "text-embedding-3-small"
EMBEDDING_MODEL_B = "text-embedding-3-large"


def _settings(chroma_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "chroma_path": chroma_path,
        "chroma_collection": "test_kb",
        "embedding_model": EMBEDDING_MODEL_A,
        "chunk_size": 200,
        "chunk_overlap": 30,
    }
    kwargs.update(overrides)
    # pydantic-settings' `_env_file` init-only control param is defined in a manual
    # `BaseSettings.__init__` override that pyright's pydantic plugin doesn't see — it
    # synthesizes `__init__` from model fields only.
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


@pytest.fixture
def stub_providers(monkeypatch: pytest.MonkeyPatch):
    """Real `build_embedding_model`/`build_llm` construct real provider clients that
    would hit the network the moment an endpoint actually embeds or generates — off
    limits in tests. Structural/lifecycle tests don't need this; tests that exercise
    ingestion or query through the real app do.
    """
    import app.main as main_module
    from tests.conftest import StubEmbedding, StubLLM

    stub_embed_model = StubEmbedding()
    stub_llm = StubLLM()
    monkeypatch.setattr(
        main_module, "build_embedding_model", lambda settings: stub_embed_model
    )
    monkeypatch.setattr(main_module, "build_llm", lambda settings: stub_llm)
    return stub_embed_model, stub_llm


class TestAppCreation:
    def test_create_app_returns_a_fastapi_instance(self, tmp_path: Path) -> None:
        from fastapi import FastAPI

        app = create_app(settings=_settings(tmp_path / "chroma"))
        assert isinstance(app, FastAPI)

    def test_module_level_app_is_importable_without_side_effects(self) -> None:
        """Importing `app.main` must not touch the network, `.env`, or the filesystem —
        `app = create_app()` only builds the FastAPI object; `lifespan` runs later."""
        import app.main

        assert app.main.app is not None


class TestRouteRegistration:
    def test_every_documented_route_is_registered(self, tmp_path: Path) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))
        paths = app.openapi()["paths"]

        assert set(paths["/documents"]) == {"get", "post"}
        assert set(paths["/documents/{document_id}"]) == {"delete"}
        assert set(paths["/reset"]) == {"post"}
        assert set(paths["/query"]) == {"post"}


class TestStartupBehavior:
    def test_lifespan_populates_shared_state(self, tmp_path: Path) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app):
            assert isinstance(app.state.store, VectorStore)
            assert app.state.registry.settings.chroma_collection == "test_kb"
            assert app.state.registry.embed_model is not None
            assert app.state.registry.llm is not None

    def test_requests_work_once_started(self, tmp_path: Path) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            response = client.get("/documents")

        assert response.status_code == 200
        assert response.json() == {"documents": []}


class TestSharedDependencyLifecycle:
    def test_the_same_store_instance_serves_every_request(self, tmp_path: Path) -> None:
        """Application-scoped, not per-request (ADR-10): two requests must observe
        each other's writes through one shared `VectorStore`, not independent ones."""
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            store_a = app.state.store
            client.get("/documents")
            store_b = app.state.store

        assert store_a is store_b

    def test_ingested_documents_persist_across_requests_on_the_shared_store(
        self, tmp_path: Path, stub_providers: object
    ) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            client.post(
                "/documents",
                files=[("files", ("report.pdf", build_pdf(["Some content."])))],
            )
            response = client.get("/documents")

        assert response.status_code == 200
        assert len(response.json()["documents"]) == 1

    def test_embed_model_and_llm_are_each_a_single_shared_instance(
        self, tmp_path: Path
    ) -> None:
        """Still true (ADR-10's amendment): as long as nothing calls one of the
        runtime-update routes, the registry hands out the same instances every time —
        it only replaces them on an explicit, successful update (see
        `TestProviderReplacement` in `test_routes.py`)."""
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            embed_a = app.state.registry.embed_model
            llm_a = app.state.registry.llm
            client.get("/documents")
            embed_b = app.state.registry.embed_model
            llm_b = app.state.registry.llm

        assert embed_a is embed_b
        assert llm_a is llm_b


class TestProviderInitializationFailure:
    def test_llm_construction_failure_fails_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.main as main_module

        def _boom(settings: Settings) -> None:
            raise RuntimeError("provider misconfigured")

        monkeypatch.setattr(main_module, "build_llm", _boom)
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with (
            pytest.raises(RuntimeError, match="provider misconfigured"),
            TestClient(app),
        ):
            pass

    def test_embedding_model_construction_failure_fails_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.main as main_module

        def _boom(settings: Settings) -> None:
            raise RuntimeError("bad embedding config")

        monkeypatch.setattr(main_module, "build_embedding_model", _boom)
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with pytest.raises(RuntimeError, match="bad embedding config"), TestClient(app):
            pass


class TestEmbeddingMismatchAtStartup:
    def test_a_mismatched_collection_fails_startup_not_a_request(
        self, tmp_path: Path
    ) -> None:
        chroma_path = tmp_path / "chroma"
        # Build a non-empty collection under the old fingerprint, as a prior process
        # (with the old EMBEDDING_MODEL) would have left it (ADR-8).
        old_store = VectorStore(
            path=chroma_path,
            collection_name="test_kb",
            embedding_fingerprint=EMBEDDING_MODEL_B,
        )
        old_store._collection.add(
            ids=["chunk-1"],
            embeddings=[[0.1] * 8],
            metadatas=[
                {"document_id": "doc-1", "filename": "a.pdf", "file_type": "pdf"}
            ],
        )

        app = create_app(
            settings=_settings(chroma_path, embedding_model=EMBEDDING_MODEL_A)
        )

        with pytest.raises(EmbeddingMismatchError), TestClient(app):
            pass

    def test_an_empty_collection_adopts_the_new_fingerprint_and_starts_cleanly(
        self, tmp_path: Path
    ) -> None:
        chroma_path = tmp_path / "chroma"
        VectorStore(
            path=chroma_path,
            collection_name="test_kb",
            embedding_fingerprint=EMBEDDING_MODEL_B,
        )

        app = create_app(
            settings=_settings(chroma_path, embedding_model=EMBEDDING_MODEL_A)
        )

        with TestClient(app) as client:
            response = client.get("/documents")

        assert response.status_code == 200


class TestCredentialValidationAtStartup:
    def test_blank_llm_api_key_fails_startup_when_settings_come_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.main as main_module

        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(tmp_path / "chroma")
        )
        app = create_app()  # settings=None: exercises the real startup path

        with pytest.raises(RuntimeError, match="LLM_API_KEY"), TestClient(app):
            pass

    def test_explicitly_injected_settings_skip_credential_validation(
        self, tmp_path: Path
    ) -> None:
        """Tests that inject `Settings` with stub providers (never a real credential)
        must keep working — validation is only for the real environment-loaded path."""
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app):
            pass  # must not raise


class TestExistingApiBehaviorUnchanged:
    """Smoke-checks that going through the real app/lifespan (rather than the bare
    router `test_routes.py` mounts) still produces the documented contracts."""

    def test_ingestion_response_shape_is_unchanged(
        self, tmp_path: Path, stub_providers: object
    ) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            response = client.post(
                "/documents",
                files=[("files", ("report.pdf", build_pdf(["Some content."])))],
            )

        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "indexed"
        assert result["filename"] == "report.pdf"

    def test_delete_is_idempotent(self, tmp_path: Path) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            response = client.delete("/documents/does-not-exist")

        assert response.status_code == 200
        assert response.json() == {"document_id": "does-not-exist", "deleted": False}

    def test_blank_query_is_still_a_422(self, tmp_path: Path) -> None:
        app = create_app(settings=_settings(tmp_path / "chroma"))

        with TestClient(app) as client:
            response = client.post("/query", json={"query": "   "})

        assert response.status_code == 422
