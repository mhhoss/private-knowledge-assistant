"""Streamlit UI: a thin HTTP client of the FastAPI application (ADR-1).

No RAG logic, storage, or provider clients are imported here. Every action calls the
API over HTTP through `ApiClient` and renders exactly what the API returns — API
response data and citation metadata are displayed as-is, never invented or transformed.
`ApiClient` is deliberately Streamlit-free so it can be unit-tested without a running UI
or a running API (see `tests/unit/test_streamlit_client.py`); everything below it is
presentation only.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import httpx
import streamlit as st

from app.config import get_settings

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Ingestion of a large document can take minutes at a slow (e.g. local CPU) embedding
# rate (see ARCHITECTURE.md ADR-13) — a short timeout here would report a false
# failure in the UI while the API keeps indexing in the background regardless.
_INGEST_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class ApiError(Exception):
    """A user-facing error from the API or the network layer — never a raw traceback."""


class ApiClient:
    """Thin wrapper over the FastAPI application's HTTP contract (`app/schemas/api.py`).

    Returns plain `dict`s decoded straight from the API's JSON responses — the schema
    shapes are preserved exactly, never renamed or restructured, so the caller displays
    the same field names and values the API sent.
    """

    def __init__(
        self, base_url: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, timeout=_TIMEOUT, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def ingest_files(self, files: list[tuple[str, bytes]]) -> list[dict]:
        """Upload files (R-01) and return their per-file outcomes (R-09)."""
        response = self._request(
            "POST",
            "/documents",
            files=[("files", (name, content)) for name, content in files],
            timeout=_INGEST_TIMEOUT,
        )
        return response.json()["results"]

    def list_documents(self) -> list[dict]:
        """List indexed documents (R-06)."""
        return self._request("GET", "/documents").json()["documents"]

    def delete_document(self, document_id: str) -> dict:
        """Delete one document by id (R-07)."""
        return self._request("DELETE", f"/documents/{document_id}").json()

    def reset_knowledge_base(self) -> dict:
        """Clear the entire knowledge base (R-07)."""
        return self._request("POST", "/reset").json()

    def submit_query(self, query: str) -> dict:
        """Ask a question (R-04) and return the grounded answer or refusal (R-05)."""
        return self._request("POST", "/query", json={"query": query}).json()

    def _request(
        self,
        method: str,
        path: str,
        *,
        files: Sequence[tuple[str, tuple[str, bytes]]] | None = None,
        json: object | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> httpx.Response:
        try:
            if timeout is None:
                response = self._client.request(method, path, files=files, json=json)
            else:
                response = self._client.request(
                    method, path, files=files, json=json, timeout=timeout
                )
        except httpx.TimeoutException as error:
            # Distinct from a connectivity failure: the API was reached and is very
            # likely still indexing in the background (see `_INGEST_TIMEOUT`'s note) —
            # telling the user it's unreachable would be actively misleading.
            raise ApiError(
                "This is taking longer than expected. The upload may still be "
                "indexing in the background — check Indexed documents shortly."
            ) from error
        except httpx.RequestError as error:
            raise ApiError(
                "Could not reach the Private Knowledge Assistant API. "
                "Please confirm it is running and try again."
            ) from error
        if response.status_code >= 400:
            raise ApiError(_error_detail(response))
        return response


def _error_detail(response: httpx.Response) -> str:
    """Translate an error response into one plain sentence, never a raw body/traceback.

    Handles both this API's own `ErrorResponse` shape (`{"detail": "..."}`) and
    FastAPI's built-in validation-error shape (`{"detail": [{"msg": ..., ...}]}`) —
    the two distinct shapes documented as unresolved in ARCHITECTURE.md.
    """
    try:
        body = response.json()
    except ValueError:
        return f"The API returned an unexpected error (HTTP {response.status_code})."

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        message = first.get("msg") if isinstance(first, dict) else None
        if message:
            return str(message)
    return f"The API returned an unexpected error (HTTP {response.status_code})."


# --- presentation (everything below calls only `ApiClient` and `st`) ---

_STATUS_BADGES = {
    "indexed": ("pka-badge-success", "Indexed"),
    "already_indexed": ("pka-badge-info", "Already indexed"),
    "failed": ("pka-badge-danger", "Failed"),
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --pka-bg: #f6f6f8;
            --pka-surface: #ffffff;
            --pka-border: #e6e6ea;
            --pka-text: #1c1c22;
            --pka-muted: #6b6b76;
            --pka-accent: #4f46e5;
            --pka-accent-soft: #eef0ff;
            --pka-success: #15803d;
            --pka-success-soft: #ecfdf3;
            --pka-danger: #b91c1c;
            --pka-danger-soft: #fef2f2;
            --pka-info: #1d4ed8;
            --pka-info-soft: #eff6ff;
        }

        html, body, [class*="css"] { unicode-bidi: plaintext; }

        .stApp { background: var(--pka-bg); }
        [data-testid="stSidebar"] { background: var(--pka-surface); border-right: 1px solid var(--pka-border); }
        [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
        .block-container { padding-top: 2.5rem; max-width: 760px; }
        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

        .pka-eyebrow {
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--pka-accent); margin-bottom: 0.35rem;
        }
        .pka-title { font-size: 1.9rem; font-weight: 700; color: var(--pka-text); line-height: 1.2; }
        .pka-subtitle { color: var(--pka-muted); font-size: 0.95rem; margin: 0.4rem 0 1.75rem 0; }

        .pka-sidebar-title {
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
            color: var(--pka-muted); margin: 0.25rem 0 0.6rem 0;
        }

        .pka-empty {
            color: var(--pka-muted); font-size: 0.85rem; padding: 0.6rem 0; text-align: center;
            border: 1px dashed var(--pka-border); border-radius: 10px; background: var(--pka-bg);
        }

        .pka-doc-row { padding: 0.5rem 0; border-bottom: 1px solid var(--pka-border); }
        .pka-doc-name { font-size: 0.88rem; font-weight: 500; color: var(--pka-text); word-break: break-word; }
        .pka-doc-meta { font-size: 0.74rem; color: var(--pka-muted); margin-top: 0.1rem; }

        .pka-ingest-row {
            display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
            padding: 0.35rem 0;
        }
        .pka-filename { font-size: 0.84rem; color: var(--pka-text); word-break: break-word; }
        .pka-ingest-error { font-size: 0.76rem; color: var(--pka-danger); margin: -0.1rem 0 0.35rem 0; }

        .pka-badge-success, .pka-badge-info, .pka-badge-danger {
            font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 999px;
            white-space: nowrap;
        }
        .pka-badge-success { background: var(--pka-success-soft); color: var(--pka-success); }
        .pka-badge-info { background: var(--pka-info-soft); color: var(--pka-info); }
        .pka-badge-danger { background: var(--pka-danger-soft); color: var(--pka-danger); }

        .pka-answer {
            background: var(--pka-surface); border: 1px solid var(--pka-border); border-radius: 14px;
            padding: 1.25rem 1.4rem; font-size: 1.02rem; line-height: 1.65; color: var(--pka-text);
            margin-top: 1rem;
        }
        .pka-refusal {
            display: flex; gap: 0.6rem; align-items: flex-start;
            background: var(--pka-info-soft); border: 1px solid #c7d7fb; border-radius: 14px;
            padding: 1rem 1.2rem; margin-top: 1rem; color: #1e3a8a; font-size: 0.95rem; line-height: 1.55;
        }
        .pka-refusal-icon { font-weight: 700; }

        .pka-citation {
            background: var(--pka-surface); border: 1px solid var(--pka-border); border-radius: 10px;
            padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
        }
        .pka-citation-file { font-size: 0.8rem; font-weight: 600; color: var(--pka-accent); margin-bottom: 0.3rem; }
        .pka-citation-excerpt { font-size: 0.86rem; color: var(--pka-muted); line-height: 1.5; }

        div[data-testid="stForm"] { border: none; padding: 0; }
        .stButton button, .stFormSubmitButton button { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient(get_settings().api_base_url)
    return st.session_state["api_client"]


def _render_ingestion_results(results: list[dict]) -> None:
    for outcome in results:
        badge_class, label = _STATUS_BADGES[outcome["status"]]
        st.markdown(
            f'<div class="pka-ingest-row">'
            f'<span class="pka-filename" dir="auto">{html.escape(outcome["filename"])}</span>'
            f'<span class="{badge_class}">{label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if outcome["status"] == "failed" and outcome.get("error"):
            st.markdown(
                f'<div class="pka-ingest-error" dir="auto">{html.escape(outcome["error"])}</div>',
                unsafe_allow_html=True,
            )


def _render_document_row(client: ApiClient, doc: dict) -> None:
    chunk_count = doc["chunk_count"]
    unit = "chunk" if chunk_count == 1 else "chunks"
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="pka-doc-row">'
            f'<div class="pka-doc-name" dir="auto">{html.escape(doc["filename"])}</div>'
            f'<div class="pka-doc-meta">{doc["file_type"].upper()} · {chunk_count} {unit}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button(
            "🗑", key=f"delete-{doc['document_id']}", help=f"Delete {doc['filename']}"
        ):
            try:
                client.delete_document(doc["document_id"])
            except ApiError as error:
                st.error(str(error))
            else:
                st.rerun()


def _render_reset_control(client: ApiClient) -> None:
    if not st.session_state.get("confirm_reset"):
        if st.button("Reset knowledge base", use_container_width=True):
            st.session_state["confirm_reset"] = True
            st.rerun()
        return

    st.warning(
        "This permanently deletes every indexed document. This cannot be undone."
    )
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.session_state["confirm_reset"] = False
            st.rerun()
    with confirm_col:
        if st.button("Confirm reset", type="primary", use_container_width=True):
            try:
                client.reset_knowledge_base()
            except ApiError as error:
                st.error(str(error))
            else:
                st.session_state["ingestion_results"] = None
            st.session_state["confirm_reset"] = False
            st.rerun()


def _render_sidebar(client: ApiClient) -> None:
    st.markdown(
        '<div class="pka-sidebar-title">Upload documents</div>', unsafe_allow_html=True
    )
    uploader_key = f"uploader_{st.session_state.get('uploader_generation', 0)}"
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=uploader_key,
    )
    st.caption("Large documents can take several minutes to index.")
    if st.button(
        "Upload", type="primary", use_container_width=True, disabled=not uploaded
    ):
        with st.spinner(
            "Indexing... this can take several minutes for large documents."
        ):
            files = [(f.name, f.getvalue()) for f in uploaded]
            try:
                results = client.ingest_files(files)
            except ApiError as error:
                st.session_state["ingestion_results"] = None
                st.session_state["ingestion_error"] = str(error)
            else:
                st.session_state["ingestion_results"] = results
                st.session_state["ingestion_error"] = None
        st.session_state["uploader_generation"] = (
            st.session_state.get("uploader_generation", 0) + 1
        )
        st.rerun()

    if st.session_state.get("ingestion_error"):
        st.error(st.session_state["ingestion_error"])
    if st.session_state.get("ingestion_results"):
        _render_ingestion_results(st.session_state["ingestion_results"])

    st.divider()

    st.markdown(
        '<div class="pka-sidebar-title">Indexed documents</div>', unsafe_allow_html=True
    )
    try:
        documents = client.list_documents()
    except ApiError as error:
        st.error(str(error))
        documents = []

    if not documents:
        st.markdown(
            '<div class="pka-empty">No documents indexed yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        for doc in documents:
            _render_document_row(client, doc)

    st.divider()
    _render_reset_control(client)


def _render_citation(source: dict) -> None:
    excerpt = html.escape(source["excerpt"]).replace("\n", "<br>")
    st.markdown(
        f'<div class="pka-citation">'
        f'<div class="pka-citation-file" dir="auto">{html.escape(source["filename"])}</div>'
        f'<div class="pka-citation-excerpt" dir="auto">{excerpt}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_answer(answer: dict) -> None:
    if answer["is_refusal"]:
        st.markdown(
            f'<div class="pka-refusal"><span class="pka-refusal-icon">i</span>'
            f'<span dir="auto">{html.escape(answer["answer"])}</span></div>',
            unsafe_allow_html=True,
        )
        return

    body = html.escape(answer["answer"]).replace("\n", "<br>")
    st.markdown(
        f'<div class="pka-answer" dir="auto">{body}</div>', unsafe_allow_html=True
    )

    sources = answer.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                _render_citation(source)


def _render_query_panel(client: ApiClient) -> None:
    st.markdown(
        '<div class="pka-eyebrow">Private Knowledge Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pka-title">Ask your documents</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="pka-subtitle">Answers are grounded only in what you\'ve indexed — '
        "in English, Persian, or both.</div>",
        unsafe_allow_html=True,
    )

    with st.form("query-form"):
        query_text = st.text_area(
            "Question",
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            height=100,
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not query_text.strip():
            st.session_state["query_error"] = "Please enter a question."
            st.session_state["last_answer"] = None
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = client.submit_query(query_text)
                except ApiError as error:
                    st.session_state["query_error"] = str(error)
                    st.session_state["last_answer"] = None
                else:
                    st.session_state["query_error"] = None
                    st.session_state["last_answer"] = answer

    if st.session_state.get("query_error"):
        st.error(st.session_state["query_error"])

    answer = st.session_state.get("last_answer")
    if answer:
        _render_answer(answer)


def main() -> None:
    st.set_page_config(
        page_title="Private Knowledge Assistant", page_icon="📚", layout="wide"
    )
    _inject_styles()

    client = _get_client()

    with st.sidebar:
        _render_sidebar(client)

    _render_query_panel(client)


if __name__ == "__main__":
    main()
