"""Unit tests for `streamlit_app.ApiClient` — the UI's only interface to the network.

No Streamlit runtime is involved: `httpx.MockTransport` stands in for the FastAPI
application, so these tests exercise exactly what the UI layer is responsible for
(request shape, response decoding, error translation) without a running server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from streamlit_app import ApiClient, ApiError, _error_detail


def _client(handler) -> ApiClient:
    return ApiClient("http://testserver", transport=httpx.MockTransport(handler))


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


class TestIngestFiles:
    def test_posts_multipart_files_and_returns_results(self) -> None:
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return _json_response(
                200,
                {
                    "results": [
                        {
                            "filename": "a.pdf",
                            "status": "indexed",
                            "document_id": "d1",
                            "chunk_count": 2,
                            "error": None,
                        }
                    ]
                },
            )

        client = _client(handler)
        results = client.ingest_files([("a.pdf", b"%PDF-1.4 ...")])

        request = captured["request"]
        assert request.method == "POST"
        assert request.url.path == "/documents"
        assert b'name="files"; filename="a.pdf"' in request.content
        assert results == [
            {
                "filename": "a.pdf",
                "status": "indexed",
                "document_id": "d1",
                "chunk_count": 2,
                "error": None,
            }
        ]

    def test_a_failed_outcome_is_returned_not_raised(self) -> None:
        """A per-file ingestion failure is API response data (R-09), never an
        `ApiError` — the client must not treat a 200 with a `failed` outcome as an
        error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                200,
                {
                    "results": [
                        {
                            "filename": "notes.txt",
                            "status": "failed",
                            "document_id": None,
                            "chunk_count": 0,
                            "error": "Unsupported file type",
                        }
                    ]
                },
            )

        client = _client(handler)
        results = client.ingest_files([("notes.txt", b"irrelevant")])

        assert results[0]["status"] == "failed"
        assert results[0]["error"] == "Unsupported file type"


class TestListDocuments:
    def test_returns_the_documents_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert request.url.path == "/documents"
            return _json_response(
                200,
                {
                    "documents": [
                        {
                            "document_id": "fa1",
                            "filename": "گزارش.pdf",
                            "file_type": "pdf",
                            "chunk_count": 3,
                        }
                    ]
                },
            )

        client = _client(handler)
        documents = client.list_documents()

        assert documents[0]["filename"] == "گزارش.pdf"

    def test_empty_knowledge_base_returns_empty_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(200, {"documents": []})

        assert _client(handler).list_documents() == []


class TestDeleteDocument:
    def test_deletes_by_id(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "DELETE"
            assert request.url.path == "/documents/doc-1"
            return _json_response(200, {"document_id": "doc-1", "deleted": True})

        result = _client(handler).delete_document("doc-1")
        assert result == {"document_id": "doc-1", "deleted": True}


class TestResetKnowledgeBase:
    def test_posts_to_reset(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/reset"
            return _json_response(200, {"message": "Knowledge base has been reset."})

        result = _client(handler).reset_knowledge_base()
        assert result["message"]


class TestSubmitQuery:
    def test_posts_the_query_body_and_returns_the_answer(self) -> None:
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return _json_response(
                200,
                {
                    "answer": "12%.",
                    "sources": [
                        {
                            "document_id": "d1",
                            "filename": "report.pdf",
                            "file_type": "pdf",
                            "chunk_id": "0001",
                            "excerpt": "Costs rose 12%.",
                        }
                    ],
                    "is_refusal": False,
                },
            )

        client = _client(handler)
        answer = client.submit_query("What were the costs?")

        request = captured["request"]
        assert json.loads(request.content) == {"query": "What were the costs?"}
        assert answer["is_refusal"] is False
        assert answer["sources"][0]["excerpt"] == "Costs rose 12%."

    def test_refusal_is_returned_as_is(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                200,
                {
                    "answer": "I don't have enough information in the indexed documents "
                    "to answer that.",
                    "sources": [],
                    "is_refusal": True,
                },
            )

        answer = _client(handler).submit_query("Anything at all?")
        assert answer["is_refusal"] is True
        assert answer["sources"] == []

    def test_persian_query_round_trips(self) -> None:
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return _json_response(
                200, {"answer": "پاسخ", "sources": [], "is_refusal": True}
            )

        _client(handler).submit_query("هزینه چقدر بود؟")
        assert json.loads(captured["request"].content) == {"query": "هزینه چقدر بود؟"}


class TestErrorHandling:
    def test_network_failure_raises_a_friendly_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = _client(handler)
        with pytest.raises(ApiError) as exc_info:
            client.list_documents()

        message = str(exc_info.value)
        assert "reach" in message.lower()
        assert "Traceback" not in message
        assert "ConnectError" not in message

    def test_structured_error_response_surfaces_its_detail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(404, {"detail": "Document not found."})

        client = _client(handler)
        with pytest.raises(ApiError, match="Document not found."):
            client.delete_document("missing")

    def test_fastapi_validation_error_shape_surfaces_the_first_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                422,
                {
                    "detail": [
                        {"loc": ["body", "query"], "msg": "query must not be blank", "type": "value_error"}
                    ]
                },
            )

        client = _client(handler)
        with pytest.raises(ApiError, match="query must not be blank"):
            client.submit_query("   ")

    def test_non_json_error_body_falls_back_to_a_generic_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="<html>Internal Server Error</html>")

        client = _client(handler)
        with pytest.raises(ApiError) as exc_info:
            client.list_documents()

        assert "500" in str(exc_info.value)
        assert "<html>" not in str(exc_info.value)

    def test_error_detail_helper_ignores_an_empty_detail_list(self) -> None:
        response = _json_response(422, {"detail": []})
        assert "422" in _error_detail(response)


class TestModuleHasNoForbiddenImports:
    def test_streamlit_app_does_not_import_rag_storage_or_provider_modules(self) -> None:
        import ast
        from pathlib import Path

        source = Path("streamlit_app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        forbidden_prefixes = (
            "app.rag",
            "app.storage",
            "app.documents",
            "chromadb",
            "llama_index",
        )
        assert not any(m.startswith(forbidden_prefixes) for m in modules), modules

    def test_importing_the_module_has_no_side_effects(self) -> None:
        """Importing must not run `main()` or make any network/Streamlit call — the
        `if __name__ == "__main__"` guard is what makes this module testable at all."""
        import importlib

        import streamlit_app

        importlib.reload(streamlit_app)  # re-import must be side-effect free too
