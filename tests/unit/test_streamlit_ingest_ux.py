"""Add-sources ingestion progress: the polling dialog reruns without error and
without an app-wide rerun on every tick.

Driven through Streamlit's own `AppTest` harness, the same approach
`test_streamlit_chat_ux.py` uses (see that module's docstring for why: no browser,
no new dependency). `_render_ingest_progress` is wrapped in `st.fragment(run_every=...)`
so its recurring poll reruns only the dialog's own content instead of the whole page
(chat, sources list, models panel) — this regression-tests that a plain `st.rerun()`
scoped to the fragment was never reintroduced, since `scope="fragment"` raises
`StreamlitAPIException` unless the current run is already a fragment rerun (it is not,
on the run that follows a full-page interaction like clicking "Add to knowledge base"),
which is exactly the crash a naive `st.rerun(scope="fragment")` fix would reintroduce.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from streamlit_app import ApiClient

_APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"

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

_RUNNING_JOB = {
    "job_id": "job1",
    "status": "running",
    "total": 1,
    "completed": 0,
    "current_filename": "a.pdf",
    "eta_seconds": None,
    "files": [{"filename": "a.pdf", "status": "processing"}],
}

_COMPLETED_JOB = {
    "job_id": "job1",
    "status": "completed",
    "total": 1,
    "completed": 1,
    "current_filename": None,
    "eta_seconds": None,
    "files": [
        {
            "filename": "a.pdf",
            "status": "indexed",
            "document_id": "d1",
            "chunk_count": 3,
        }
    ],
}


def _client_polling_job(poll_responses: list[dict]) -> ApiClient:
    """A stub `ApiClient` whose job-status endpoint returns `poll_responses` in
    order, one per call, repeating the last response once exhausted."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/documents":
            return httpx.Response(200, json={"documents": []})
        if request.method == "GET" and request.url.path == "/settings":
            return httpx.Response(200, json=_SETTINGS)
        if request.method == "GET" and request.url.path == "/documents/jobs/job1":
            index = min(calls["n"], len(poll_responses) - 1)
            calls["n"] += 1
            return httpx.Response(200, json=poll_responses[index])
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    return ApiClient("http://test", transport=httpx.MockTransport(handler))


def _launch_with_running_job(client: ApiClient) -> AppTest:
    """Start the app as if `_start_ingest_batch` already ran: a job id is already
    in session state, so the dialog reopens straight into the progress fragment."""
    at = AppTest.from_file(str(_APP_PATH), default_timeout=60)
    at.session_state["api_client"] = client
    at.session_state["ingest_job_id"] = "job1"
    at.session_state["ingest_started_at"] = 0.0
    return at.run()


class TestIngestProgressPolling:
    def test_a_running_job_renders_without_error_across_several_polls(self) -> None:
        client = _client_polling_job([_RUNNING_JOB, _RUNNING_JOB, _RUNNING_JOB])
        at = _launch_with_running_job(client)
        assert not at.exception, at.exception

        at = at.run()
        assert not at.exception, at.exception
        assert "ingest_job_id" in at.session_state

    def test_job_completion_clears_state_and_reaches_the_sources_list(self) -> None:
        client = _client_polling_job([_RUNNING_JOB, _COMPLETED_JOB])
        at = _launch_with_running_job(client)
        at = at.run()  # second poll observes completion

        assert not at.exception, at.exception
        assert "ingest_job_id" not in at.session_state
        assert at.session_state["upload_outcomes"] == _COMPLETED_JOB["files"]
