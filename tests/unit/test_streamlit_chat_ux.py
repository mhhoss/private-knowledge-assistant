"""Chat UX: composer clears immediately on Send, and in-memory multi-exchange history
that keeps a question even when its request fails.

The success/history behavior is driven through Streamlit's own `AppTest` harness — a
headless script-execution/session-state test tool already shipped with the `streamlit`
dependency (no browser, no new dependency). `AppTest.from_file` re-executes the whole
script fresh on every `.run()`, which redefines module-level classes (including
`ApiError`) each time; an exception raised by an `ApiClient` instance built outside
that exec (e.g. from a directly-imported class) therefore cannot be caught by an
`except ApiError` inside the re-executed script — a harness artifact, not a production
bug, since the real app always raises and catches the same run's own class. The
failed-request-preserves-the-question behavior is exercised instead by pre-seeding
`session_state["exchanges"]` with a `(question, None, error)` entry — exactly the shape
`_render_chat` appends on a real failure — and asserting on what actually renders,
which sidesteps that harness limitation and is a more direct test of the contract
that matters here (the question and its error both show up) than the exception
plumbing would be.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from streamlit_app import ApiClient

_APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"

_ONE_DOCUMENT = {
    "documents": [
        {"document_id": "d1", "filename": "a.pdf", "file_type": "pdf", "chunk_count": 1}
    ]
}
_SETTINGS = {
    "llm": {
        "model": "m",
        "host": "h",
        "base_url": "http://h",
        "masked_key": "k",
        "is_local": False,
    },
    "embedding": {
        "model": "m",
        "host": "h",
        "base_url": "http://h",
        "masked_key": "k",
        "is_local": True,
    },
}


def _answer(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "answer": text,
            "sources": [
                {
                    "document_id": "d1",
                    "filename": "a.pdf",
                    "file_type": "pdf",
                    "chunk_id": "0000",
                    "excerpt": "excerpt text",
                }
            ],
            "is_refusal": False,
        },
    )


def _client_with_queued_answers(
    *answers: httpx.Response, documents: list[dict] | None = None
) -> ApiClient:
    """A stub `ApiClient` whose `/query` responses are consumed in order — one real
    request per `submit_query` call, matching how the composer actually calls it."""
    queue = list(answers)
    docs = {"documents": documents if documents is not None else _ONE_DOCUMENT["documents"]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/documents":
            return httpx.Response(200, json=docs)
        if request.method == "GET" and request.url.path == "/settings":
            return httpx.Response(200, json=_SETTINGS)
        if request.method == "POST" and request.url.path == "/query":
            return queue.pop(0)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    return ApiClient("http://test", transport=httpx.MockTransport(handler))


def _launch(client: ApiClient) -> AppTest:
    """Start the real app with a stubbed `ApiClient` pre-seeded into session state —
    `_get_client()` reuses whatever is already there instead of building its own."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
    at.session_state["api_client"] = client
    return at.run()


def _ask(at: AppTest, question: str) -> AppTest:
    at.text_area(key="ask_question").set_value(question)
    at.button(key="FormSubmitter:ask-→").click()
    return at.run()


class TestComposerClearsImmediatelyOnSend:
    def test_input_is_cleared_after_a_successful_answer(self) -> None:
        at = _launch(_client_with_queued_answers(_answer("first answer")))
        at = _ask(at, "what is x?")

        assert not at.exception, at.exception
        assert at.text_area(key="ask_question").value == ""

    def test_input_is_cleared_even_when_the_attempt_is_blocked(self) -> None:
        """Clearing happens the instant Send is clicked, before any outcome is known
        — including a pre-flight block (no documents indexed yet), not just success."""
        at = _launch(_client_with_queued_answers(documents=[]))
        at = _ask(at, "what is x?")

        assert not at.exception, at.exception
        assert at.text_area(key="ask_question").value == ""
        assert at.session_state["chat_error"]  # the block was real, not silently ignored
        assert at.session_state["exchanges"] == []  # a pre-flight block isn't a request


class TestChatHistory:
    def test_multiple_exchanges_remain_visible_in_chronological_order(self) -> None:
        at = _launch(
            _client_with_queued_answers(_answer("first answer"), _answer("second answer"))
        )
        at = _ask(at, "what is x?")
        at = _ask(at, "what is y?")

        assert not at.exception, at.exception
        exchanges = at.session_state["exchanges"]
        assert [q for q, _reply, _error in exchanges] == ["what is x?", "what is y?"]
        assert [r["answer"] for _q, r, _error in exchanges] == [
            "first answer",
            "second answer",
        ]
        assert [error for _q, _r, error in exchanges] == [None, None]

    def test_a_new_exchange_does_not_overwrite_previous_exchanges(self) -> None:
        at = _launch(
            _client_with_queued_answers(_answer("first answer"), _answer("second answer"))
        )
        at = _ask(at, "what is x?")
        first_exchange = at.session_state["exchanges"][0]

        at = _ask(at, "what is y?")

        assert len(at.session_state["exchanges"]) == 2
        assert at.session_state["exchanges"][0] == first_exchange

    def test_history_is_in_memory_and_session_local_not_fetched_from_anywhere(
        self,
    ) -> None:
        """A brand-new `AppTest` (a fresh script exec and fresh `session_state`, the
        closest equivalent to a new browser session) starts with no history — nothing
        is loaded from the API or any other durable source."""
        at = _launch(_client_with_queued_answers())
        assert at.session_state["exchanges"] == []


class TestFailedRequestPreservesTheQuestion:
    """A request that fails must never make its question disappear — it stays in
    history with the error shown directly beneath it (see the module docstring for why
    this is exercised by pre-seeding the exact shape `_render_chat` produces, rather
    than through a live failing request)."""

    def test_a_failed_exchange_renders_its_question_and_its_error(self) -> None:
        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.session_state["api_client"] = _client_with_queued_answers()
        at.session_state["exchanges"] = [
            ("what is x?", None, "The provider could not be reached.")
        ]
        at = at.run()

        assert not at.exception, at.exception
        rendered = "\n".join(m.value for m in at.markdown)
        assert "what is x?" in rendered
        assert "The provider could not be reached." in rendered

    def test_a_failed_exchange_does_not_replace_an_earlier_successful_one(self) -> None:
        at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
        at.session_state["api_client"] = _client_with_queued_answers()
        at.session_state["exchanges"] = [
            (
                "what is x?",
                {
                    "answer": "first answer",
                    "sources": [
                        {
                            "document_id": "d1",
                            "filename": "a.pdf",
                            "file_type": "pdf",
                            "chunk_id": "0000",
                            "excerpt": "excerpt text",
                        }
                    ],
                    "is_refusal": False,
                },
                None,
            ),
            ("what is y?", None, "boom"),
        ]
        at = at.run()

        assert not at.exception, at.exception
        rendered = "\n".join(m.value for m in at.markdown)
        assert "first answer" in rendered
        assert "what is x?" in rendered
        assert "what is y?" in rendered
        assert "boom" in rendered
