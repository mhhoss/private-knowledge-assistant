"""Unit tests for `streamlit_app.ApiClient` — the UI's only interface to the network.

No Streamlit runtime is involved: `httpx.MockTransport` stands in for the FastAPI
application, so these tests exercise exactly what the UI layer is responsible for
(request shape, response decoding, error translation) without a running server.
"""

from __future__ import annotations

import json

import httpx
import pytest

from streamlit_app import (
    ApiClient,
    ApiError,
    _advance_batch,
    _answer_html,
    _batch_eta,
    _batch_is_complete,
    _cancel_remaining,
    _error_detail,
    _exclude_removed,
    _format_eta,
    _new_ingest_batch,
)


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

    def test_timeout_is_reported_as_slow_not_unreachable(self) -> None:
        """A long-running ingestion job that outlasts the client timeout is not the
        same failure as the API being down — the wording must not conflate them."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = _client(handler)
        with pytest.raises(ApiError) as exc_info:
            client.ingest_files([("big.pdf", b"%PDF-1.4 ...")])

        message = str(exc_info.value).lower()
        assert "longer than expected" in message
        assert "confirm it is running" not in message

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
                        {
                            "loc": ["body", "query"],
                            "msg": "query must not be blank",
                            "type": "value_error",
                        }
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


class TestSettingsAndProbe:
    def test_read_settings_returns_the_masked_configuration(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert request.url.path == "/settings"
            return _json_response(
                200,
                {
                    "llm": {
                        "model": "openai/gpt-4o-mini",
                        "host": "openrouter.ai",
                        "masked_key": "sk-or-v1••••••4f2a",
                        "is_local": False,
                    },
                    "embedding": {
                        "model": "bge-m3",
                        "host": "127.0.0.1:11434",
                        "masked_key": "ollama••••••llama",
                        "is_local": True,
                    },
                },
            )

        config = _client(handler).read_settings()
        assert config["embedding"]["is_local"] is True
        assert config["llm"]["masked_key"] == "sk-or-v1••••••4f2a"

    def test_probe_result_is_returned_as_data(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/settings/test"
            return _json_response(
                200,
                {
                    "llm": {"ok": True, "detail": None},
                    "embedding": {"ok": False, "detail": "Connection error."},
                },
            )

        result = _client(handler).test_providers()
        assert result["llm"]["ok"] is True
        assert result["embedding"]["ok"] is False
        assert result["embedding"]["detail"] == "Connection error."

    def test_update_llm_settings_posts_the_new_configuration(self) -> None:
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            assert request.url.path == "/settings/llm"
            return _json_response(
                200,
                {
                    "model": "gpt-x",
                    "host": "new-host",
                    "base_url": "http://new-host/v1",
                    "masked_key": "not set",
                    "is_local": False,
                },
            )

        result = _client(handler).update_llm_settings(
            api_key="k", base_url="http://new-host/v1", model="gpt-x"
        )

        assert json.loads(captured["request"].content) == {
            "api_key": "k",
            "base_url": "http://new-host/v1",
            "model": "gpt-x",
        }
        assert result["model"] == "gpt-x"

    def test_update_llm_settings_raises_on_a_failed_probe(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(502, {"detail": "provider unreachable"})

        with pytest.raises(ApiError, match="provider unreachable"):
            _client(handler).update_llm_settings(
                api_key="k", base_url="http://x/v1", model="x"
            )

    def test_update_embedding_settings_posts_the_new_configuration(self) -> None:
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            assert request.url.path == "/settings/embedding"
            return _json_response(
                200,
                {
                    "model": "bge-m3",
                    "host": "127.0.0.1:11434",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "masked_key": "not set",
                    "is_local": True,
                },
            )

        result = _client(handler).update_embedding_settings(
            api_key="", base_url="http://127.0.0.1:11434/v1", model="bge-m3"
        )

        assert json.loads(captured["request"].content) == {
            "api_key": "",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "bge-m3",
        }
        assert result["model"] == "bge-m3"

    def test_update_embedding_settings_raises_on_a_fingerprint_conflict(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                409, {"detail": "cannot switch without resetting the knowledge base"}
            )

        with pytest.raises(ApiError, match="resetting the knowledge base"):
            _client(handler).update_embedding_settings(
                api_key="k", base_url="http://x/v1", model="x"
            )


class TestAnswerRendering:
    """`_answer_html` turns `[n]` markers into styled references and escapes the rest.

    Range validity is the backend's guarantee (`rag/generator.py`), so this only has to
    render — but it must never emit unescaped user/model content.
    """

    def test_markers_become_reference_spans(self) -> None:
        rendered = _answer_html("Costs rose 12% [1].")
        assert '<span class="pka-ref">1</span>' in rendered
        assert "[1]" not in rendered

    def test_every_marker_in_a_multi_source_answer_is_rendered(self) -> None:
        rendered = _answer_html("First [1]. Second [2]. Third [3].")
        for number in ("1", "2", "3"):
            assert f'<span class="pka-ref">{number}</span>' in rendered

    def test_an_answer_without_markers_renders_as_plain_prose(self) -> None:
        rendered = _answer_html("Costs rose 12 percent.")
        assert rendered == "Costs rose 12 percent."
        assert "pka-ref" not in rendered

    def test_newlines_become_line_breaks(self) -> None:
        assert "<br>" in _answer_html("One line.\nAnother line.")

    def test_html_in_the_answer_is_escaped(self) -> None:
        rendered = _answer_html("<script>alert('x')</script> [1]")
        assert "<script>" not in rendered
        assert "&lt;script&gt;" in rendered
        assert '<span class="pka-ref">1</span>' in rendered

    def test_persian_answer_markers_are_rendered(self) -> None:
        rendered = _answer_html("هزینه دوازده درصد افزایش یافت [1].")
        assert '<span class="pka-ref">1</span>' in rendered
        assert "هزینه دوازده درصد افزایش یافت" in rendered

    def test_mixed_script_answer_keeps_both_scripts_and_markers(self) -> None:
        rendered = _answer_html("تیم از Kubernetes استفاده می‌کند [2].")
        assert "Kubernetes" in rendered
        assert "تیم از" in rendered
        assert '<span class="pka-ref">2</span>' in rendered

    def test_multi_digit_markers_are_supported(self) -> None:
        assert '<span class="pka-ref">12</span>' in _answer_html("Claim [12].")


class TestModuleHasNoForbiddenImports:
    def test_streamlit_app_does_not_import_rag_storage_or_provider_modules(
        self,
    ) -> None:
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


class TestFormatEta:
    def test_seconds_under_a_minute(self) -> None:
        assert _format_eta(42.4) == "~42s"

    def test_rounds_to_the_nearest_second(self) -> None:
        assert _format_eta(1.6) == "~2s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_eta(125) == "~2m 5s"

    def test_exact_minutes_drop_the_seconds(self) -> None:
        assert _format_eta(120) == "~2m"

    def test_never_negative(self) -> None:
        assert _format_eta(-5) == "~0s"


class TestExcludeRemoved:
    """The pre-start "remove from this batch" filter (requirement 3, first bullet)."""

    def test_keeps_everything_when_nothing_is_excluded(self) -> None:
        assert _exclude_removed(["a.pdf", "b.pdf"], set()) == ["a.pdf", "b.pdf"]

    def test_drops_excluded_names_only(self) -> None:
        assert _exclude_removed(["a.pdf", "b.pdf", "c.pdf"], {"b.pdf"}) == [
            "a.pdf",
            "c.pdf",
        ]


class TestIngestBatchStateMachine:
    """The batch progress state machine is plain data + pure functions, independent
    of Streamlit — real per-file progress, `X / N`, ETA, and cancellation are all
    testable here without a running app."""

    def test_new_batch_starts_at_zero_of_n_and_is_not_complete(self) -> None:
        batch = _new_ingest_batch([("a.pdf", b"1"), ("b.pdf", b"2")], started_at=0.0)
        assert batch["index"] == 0
        assert len(batch["files"]) == 2
        assert not _batch_is_complete(batch)

    def test_eta_is_estimating_before_any_file_completes(self) -> None:
        batch = _new_ingest_batch([("a.pdf", b"1"), ("b.pdf", b"2")], started_at=0.0)
        assert _batch_eta(batch) == "Estimating…"

    def test_eta_uses_real_completed_file_timing_once_available(self) -> None:
        """Not simulated: the ETA is `avg(real durations so far) * files remaining`."""
        batch = _new_ingest_batch(
            [("a.pdf", b"1"), ("b.pdf", b"2"), ("c.pdf", b"3")], started_at=0.0
        )
        _advance_batch(batch, {"filename": "a.pdf", "status": "indexed"}, 10.0)
        # One real 10s sample, two files remaining -> 20s.
        assert _batch_eta(batch) == _format_eta(20.0)

    def test_advance_batch_reaches_completion_with_correct_x_of_n(self) -> None:
        batch = _new_ingest_batch([("a.pdf", b"1"), ("b.pdf", b"2")], started_at=0.0)
        assert not _batch_is_complete(batch)

        _advance_batch(batch, {"filename": "a.pdf", "status": "indexed"}, 1.0)
        assert batch["index"] == 1
        assert not _batch_is_complete(batch)

        _advance_batch(batch, {"filename": "b.pdf", "status": "indexed"}, 1.0)
        assert batch["index"] == 2
        assert _batch_is_complete(batch)

    def test_advance_batch_never_alters_the_outcome_it_is_given(self) -> None:
        """No false success/failure states: the batch records exactly the outcome
        dict it receives, never substituting or inferring a status of its own."""
        batch = _new_ingest_batch([("a.pdf", b"1")], started_at=0.0)
        real_outcome = {
            "filename": "a.pdf",
            "status": "already_indexed",
            "document_id": "doc-1",
            "chunk_count": 0,
            "error": None,
        }
        _advance_batch(batch, real_outcome, 0.5)
        assert batch["results"] == [real_outcome]

    def test_mixed_batch_preserves_every_real_status(self) -> None:
        """Indexed, already_indexed, and failed outcomes all survive untouched in a
        single batch — exactly the existing per-file semantics, unchanged."""
        batch = _new_ingest_batch(
            [("a.pdf", b"1"), ("b.pdf", b"2"), ("c.pdf", b"3")], started_at=0.0
        )
        _advance_batch(batch, {"filename": "a.pdf", "status": "indexed"}, 1.0)
        _advance_batch(batch, {"filename": "b.pdf", "status": "already_indexed"}, 0.1)
        _advance_batch(
            batch, {"filename": "c.pdf", "status": "failed", "error": "bad file"}, 0.2
        )
        assert _batch_is_complete(batch)
        assert [r["status"] for r in batch["results"]] == [
            "indexed",
            "already_indexed",
            "failed",
        ]

    def test_cancel_remaining_only_marks_files_not_yet_started(self) -> None:
        """Safe handling of files already being processed: cancelling never touches
        `files[:index]` — whatever was already indexed, already-indexed, or failed
        stays exactly as recorded."""
        batch = _new_ingest_batch(
            [("a.pdf", b"1"), ("b.pdf", b"2"), ("c.pdf", b"3")], started_at=0.0
        )
        _advance_batch(batch, {"filename": "a.pdf", "status": "indexed"}, 1.0)

        _cancel_remaining(batch)

        assert _batch_is_complete(batch)
        statuses = {r["filename"]: r["status"] for r in batch["results"]}
        assert statuses["a.pdf"] == "indexed"  # untouched, already real
        assert statuses["b.pdf"] == "skipped"
        assert statuses["c.pdf"] == "skipped"

    def test_cancel_remaining_on_a_fresh_batch_skips_everything(self) -> None:
        batch = _new_ingest_batch([("a.pdf", b"1"), ("b.pdf", b"2")], started_at=0.0)
        _cancel_remaining(batch)
        assert _batch_is_complete(batch)
        assert {r["status"] for r in batch["results"]} == {"skipped"}
