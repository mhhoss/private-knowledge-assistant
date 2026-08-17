"""HTTP integration tests for `app/api/routes.py`.

Exercises the router through a real `TestClient`, with `StubEmbedding`/`StubLLM` and a
temporary `VectorStore` injected via `app.dependency_overrides` — never a real provider
or `chroma_db/`. `app/main.py` does not exist yet, so each test builds its own minimal
`FastAPI` app around the router, the way `main.py` eventually will.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.config import Settings
from app.storage.vector_store import VectorStore
from tests.conftest import StubEmbedding, StubLLM, settings_kwargs
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

ENGLISH_TEXT = "Kubernetes cluster costs increased significantly this quarter."
PERSIAN_TEXT = "هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت."


def _settings(**overrides: object) -> Settings:
    return Settings(
        **settings_kwargs(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            retrieval_top_k=5,
            retrieval_min_score=0.0,
            **overrides,
        )
    )


@pytest.fixture
def app(store: VectorStore, embed_model: StubEmbedding, llm: StubLLM) -> FastAPI:
    application = FastAPI()
    application.include_router(routes.router)
    application.dependency_overrides[routes._get_settings] = lambda: _settings()
    application.dependency_overrides[routes._get_vector_store] = lambda: store
    application.dependency_overrides[routes._get_embed_model] = lambda: embed_model
    application.dependency_overrides[routes._get_llm] = lambda: llm
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def upload(client: TestClient, files: list[tuple[str, bytes]]) -> httpx.Response:
    return client.post(
        "/documents",
        files=[("files", (name, content)) for name, content in files],
    )


class TestIngestDocuments:
    def test_single_file_ingestion(self, client: TestClient) -> None:
        response = upload(client, [("report.pdf", build_pdf([ENGLISH_TEXT]))])

        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["filename"] == "report.pdf"
        assert result["status"] == "indexed"
        assert result["chunk_count"] > 0
        assert result["error"] is None
        assert result["document_id"]

    def test_multi_file_ingestion_with_partial_failure(
        self, client: TestClient
    ) -> None:
        response = upload(
            client,
            [
                ("a.pdf", build_pdf(["Alpha document."])),
                ("notes.txt", b"unsupported extension"),
                ("گزارش.pdf", build_pdf([PERSIAN_TEXT])),
            ],
        )

        assert response.status_code == 200
        by_name = {r["filename"]: r for r in response.json()["results"]}
        assert by_name["a.pdf"]["status"] == "indexed"
        assert by_name["notes.txt"]["status"] == "failed"
        assert by_name["notes.txt"]["error"]
        assert by_name["notes.txt"]["chunk_count"] == 0
        assert by_name["گزارش.pdf"]["status"] == "indexed"

    def test_unsupported_extension_is_a_failed_result_not_an_http_error(
        self, client: TestClient
    ) -> None:
        response = upload(client, [("notes.txt", b"plain text")])

        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "failed"
        assert result["error"]

    def test_corrupt_file_of_a_supported_type_is_a_failed_result(
        self, client: TestClient
    ) -> None:
        response = upload(client, [("broken.pdf", b"not a real pdf")])

        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "failed"

    def test_a_request_where_every_file_fails_is_still_200(
        self, client: TestClient
    ) -> None:
        response = upload(
            client,
            [("a.txt", b"x"), ("b.txt", b"y")],
        )

        assert response.status_code == 200
        statuses = {r["status"] for r in response.json()["results"]}
        assert statuses == {"failed"}


class TestListDocuments:
    def test_empty_knowledge_base(self, client: TestClient) -> None:
        response = client.get("/documents")

        assert response.status_code == 200
        assert response.json() == {"documents": []}

    def test_lists_ingested_documents(self, client: TestClient) -> None:
        upload(client, [("report.pdf", build_pdf([ENGLISH_TEXT]))])
        upload(client, [("گزارش.docx", build_docx([PERSIAN_TEXT]))])

        response = client.get("/documents")

        assert response.status_code == 200
        filenames = {doc["filename"] for doc in response.json()["documents"]}
        assert filenames == {"report.pdf", "گزارش.docx"}
        for doc in response.json()["documents"]:
            assert doc["chunk_count"] >= 1


class TestDeleteDocument:
    def test_deletes_an_existing_document(self, client: TestClient) -> None:
        ingested = upload(client, [("report.pdf", build_pdf([ENGLISH_TEXT]))]).json()[
            "results"
        ][0]
        document_id = ingested["document_id"]

        response = client.delete(f"/documents/{document_id}")

        assert response.status_code == 200
        assert response.json() == {"document_id": document_id, "deleted": True}
        assert client.get("/documents").json()["documents"] == []

    def test_deleting_an_absent_document_id_is_not_an_error(
        self, client: TestClient
    ) -> None:
        response = client.delete("/documents/does-not-exist")

        assert response.status_code == 200
        assert response.json() == {"document_id": "does-not-exist", "deleted": False}


class TestReset:
    def test_reset_clears_the_knowledge_base(self, client: TestClient) -> None:
        upload(client, [("report.pdf", build_pdf([ENGLISH_TEXT]))])
        upload(client, [("گزارش.docx", build_docx([PERSIAN_TEXT]))])

        response = client.post("/reset")

        assert response.status_code == 200
        assert response.json()["message"]
        assert client.get("/documents").json()["documents"] == []

    def test_reset_on_an_already_empty_knowledge_base(self, client: TestClient) -> None:
        response = client.post("/reset")

        assert response.status_code == 200


class TestQuery:
    def test_successful_grounded_query_with_citations(
        self, client: TestClient, llm: StubLLM
    ) -> None:
        upload(client, [("report.pdf", build_pdf([ENGLISH_TEXT]))])
        llm.response = "Kubernetes costs rose significantly."

        response = client.post(
            "/query", json={"query": "How did Kubernetes cluster costs change?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["is_refusal"] is False
        assert body["answer"] == "Kubernetes costs rose significantly."
        assert len(body["sources"]) >= 1
        source = body["sources"][0]
        assert source["filename"] == "report.pdf"
        assert source["document_id"]
        assert source["excerpt"]

    def test_insufficient_context_refusal(
        self, client: TestClient, llm: StubLLM
    ) -> None:
        llm.response = "this must never be seen"

        response = client.post("/query", json={"query": "Anything at all?"})

        assert response.status_code == 200
        body = response.json()
        assert body["is_refusal"] is True
        assert body["sources"] == []
        assert body["answer"]
        assert llm.call_count == 0

    def test_blank_query_is_rejected(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "   "})

        assert response.status_code == 422

    def test_missing_query_field_is_rejected(self, client: TestClient) -> None:
        response = client.post("/query", json={})

        assert response.status_code == 422

    def test_persian_query_against_persian_document(
        self, client: TestClient, llm: StubLLM
    ) -> None:
        upload(client, [("گزارش.pdf", build_pdf([PERSIAN_TEXT]))])
        llm.response = "هزینه دوازده درصد افزایش یافت."

        response = client.post(
            "/query", json={"query": "هزینه خوشه کوبرنتیز چقدر تغییر کرد؟"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["is_refusal"] is False
        assert body["sources"][0]["filename"] == "گزارش.pdf"


class TestResponseSchemas:
    def test_ingestion_response_matches_schema_exactly(
        self, client: TestClient
    ) -> None:
        response = upload(client, [("a.pdf", build_pdf(["Some text."]))])
        result = response.json()["results"][0]
        assert set(result.keys()) == {
            "filename",
            "status",
            "document_id",
            "chunk_count",
            "error",
        }

    def test_document_list_response_matches_schema_exactly(
        self, client: TestClient
    ) -> None:
        upload(client, [("a.pdf", build_pdf(["Some text."]))])
        response = client.get("/documents")
        doc = response.json()["documents"][0]
        assert set(doc.keys()) == {
            "document_id",
            "filename",
            "file_type",
            "chunk_count",
        }

    def test_answer_response_matches_schema_exactly(
        self, client: TestClient, llm: StubLLM
    ) -> None:
        upload(client, [("a.pdf", build_pdf([ENGLISH_TEXT]))])
        llm.response = "answer"
        response = client.post("/query", json={"query": "Kubernetes costs"})
        body = response.json()
        assert set(body.keys()) == {"answer", "sources", "is_refusal"}
        if body["sources"]:
            assert set(body["sources"][0].keys()) == {
                "document_id",
                "filename",
                "file_type",
                "chunk_id",
                "excerpt",
            }
