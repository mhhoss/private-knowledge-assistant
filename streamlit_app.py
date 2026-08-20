"""Streamlit UI: a thin HTTP client of the FastAPI application (ADR-1).

No RAG logic, storage, or provider clients are imported here. Every action calls the
API over HTTP through `ApiClient` and renders exactly what the API returns — API
response data, provider configuration, and citation metadata are displayed as-is, never
invented or transformed. `ApiClient` is deliberately Streamlit-free so it can be
unit-tested without a running UI or a running API (see
`tests/unit/test_streamlit_client.py`); everything below it is presentation only.

Citation markers (`[n]`) are rendered, not validated: `rag/generator.py` guarantees every
surviving marker resolves to `sources[n - 1]`, so this module only splits the answer text
and attaches the popover.
"""

from __future__ import annotations

import html
import re
import time
from collections.abc import Sequence

import httpx
import streamlit as st
import streamlit.components.v1 as components

from app.config import get_settings

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Ingestion itself now runs as a background job (ADR-17): `POST /documents` and
# `GET .../jobs/{id}` both return almost immediately regardless of how long embedding
# takes, so they need no timeout longer than any other call — unlike the old
# synchronous upload this constant used to cover.
# A provider probe makes a real call and retries internally before giving up.
_PROBE_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


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

    def start_ingestion(self, files: list[tuple[str, bytes]]) -> dict:
        """Start a background ingestion job for these files (R-01, ADR-17).

        Returns the job's initial (`queued`) state immediately; poll
        `get_ingestion_job` for progress and final per-file outcomes (R-09).
        """
        return self._request(
            "POST",
            "/documents",
            files=[("files", (name, content)) for name, content in files],
        ).json()

    def get_ingestion_job(self, job_id: str) -> dict:
        """Poll one ingestion job's progress and, once finished, its results (ADR-17)."""
        return self._request("GET", f"/documents/jobs/{job_id}").json()

    def cancel_ingestion_job(self, job_id: str) -> dict:
        """Request cancellation of a job's not-yet-started files (ADR-17).

        A file already being embedded still finishes normally; only files whose turn
        has not yet come are affected.
        """
        return self._request("DELETE", f"/documents/jobs/{job_id}").json()

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

    def read_settings(self) -> dict:
        """Read the provider configuration in effect, credentials already masked."""
        return self._request("GET", "/settings").json()

    def test_providers(self) -> dict:
        """Probe both configured providers; outcomes come back as data, not errors."""
        return self._request("POST", "/settings/test", timeout=_PROBE_TIMEOUT).json()

    def update_llm_settings(self, *, api_key: str, base_url: str, model: str) -> dict:
        """Replace the LLM provider at runtime (process-local; never written to `.env`).

        Raises `ApiError` if the new provider fails a live probe — the previous
        provider is left in effect on the API side in that case.
        """
        return self._request(
            "POST",
            "/settings/llm",
            json={"api_key": api_key, "base_url": base_url, "model": model},
            timeout=_PROBE_TIMEOUT,
        ).json()

    def update_embedding_settings(
        self, *, api_key: str, base_url: str, model: str
    ) -> dict:
        """Replace the embedding provider at runtime (process-local; never written to
        `.env`).

        Raises `ApiError` on a failed probe, or on an HTTP 409 if the change would
        conflict with an already-indexed knowledge base (ADR-8) — the previous
        provider is left in effect on the API side in either case.
        """
        return self._request(
            "POST",
            "/settings/embedding",
            json={"api_key": api_key, "base_url": base_url, "model": model},
            timeout=_PROBE_TIMEOUT,
        ).json()

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
            # likely still working — telling the user it is unreachable would be
            # actively misleading.
            raise ApiError(
                "This is taking longer than expected. The work may still be running "
                "in the background — check Sources shortly."
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
#
# A three-pane workspace — Sources | Chat | Models — as white cards on a light canvas,
# each with its own header and collapse control. Sources and Models can be folded to a
# narrow rail so the conversation moves toward the centre. There is no sidebar.

# Splits an answer into text and `[n]` markers in one pass, keeping the markers so they
# can be rendered as interactive references.
_CITATION_SPLIT = re.compile(r"(\[\d+\])")

_STATUS_LABEL = {
    "indexed": ("pka-ok", "Indexed"),
    "already_indexed": ("pka-dim", "Already indexed"),
    "failed": ("pka-bad", "Failed"),
    # Client-side only: a file the user removed from the batch before it was ever sent
    # to the API. Never produced by the backend, never a real `IngestStatus` value —
    # purely a label for this dialog's own progress list.
    "skipped": ("pka-dim", "Removed"),
}

_FONT_URL = "app/static/fonts/Vazirmatn-Variable.woff2"

_CSS = """
<style>
/* Vazirmatn covers Latin and Persian in one family with matched metrics, so a mixed
   line shares a baseline and weight. Served from this machine, never a font CDN. */
@font-face {
    font-family: 'Vazirmatn';
    src: url('__FONT__') format('woff2-variations');
    font-weight: 100 900;
    font-display: swap;
}

:root {
    --ink: #1c1c1e;
    --ink-soft: #5d6066;
    --ink-faint: #8b8f96;
    --canvas: #f2f3f5;
    --card: #ffffff;
    --line: #e4e6e9;
    --line-soft: #eef0f2;
    --blue: #3d78b5;
    --blue-hover: #336aa2;
    --blue-wash: #eaf2fa;
    --blue-ink: #2d6193;
    --bad: #c0392f;
    --bad-wash: #fdf0ee;
    --r-card: 14px;
    --r-ctl: 10px;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Vazirmatn', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
/* Direction follows each element's own content, everywhere. */
html, body, [class*="css"] { unicode-bidi: plaintext; }

.stApp { background: var(--canvas); }
/* The frame never moves: the page itself does not scroll, so the masthead and the three
   panel cards stay exactly where they are. Scrolling happens inside each panel instead
   (see the column rules below), which keeps the layout stable and jitter-free. */
html, body { overflow: hidden; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    height: 100vh; overflow: hidden;
}
.block-container {
    padding: 1.1rem 1.1rem 0.6rem 1.1rem; max-width: 1700px;
    height: 100vh; overflow: hidden;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* --- masthead --- */
.pka-brand {
    display: flex; align-items: center; gap: 0.6rem; padding: 0 0.25rem 0.9rem 0.25rem;
}
.pka-brand-dot {
    width: 22px; height: 22px; border-radius: 7px; background: var(--blue);
    display: inline-block; flex: none;
}
.pka-brand-name {
    font-size: 1.02rem; font-weight: 600; color: var(--ink); letter-spacing: -0.01em;
}
.pka-brand-sub { font-size: 0.8rem; color: var(--ink-faint); margin-inline-start: auto; }

/* --- panes as cards --- */
/* A fixed-height card whose *content* scrolls, not the card: the frame is immovable and
   only what is inside a panel moves when the user scrolls it. */
div[data-testid="stColumn"]:has(.pka-card) {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--r-card); padding: 0.95rem 1.05rem 1.1rem 1.05rem;
    height: calc(100vh - 88px); overflow-y: auto; overflow-x: hidden;
    scrollbar-width: thin;
}
div[data-testid="stColumn"]:has(.pka-rail) {
    background: var(--card); border: 1px solid var(--line);
    border-radius: var(--r-card); padding: 0.95rem 0.35rem;
    height: calc(100vh - 88px); overflow: hidden;
}
/* Panel titles sit in the middle column of a symmetric 1fr/auto/1fr header row, so
   centring the text here centres it in the panel, not just in its own column. */
.pka-head {
    font-size: 0.95rem; font-weight: 600; color: var(--ink); letter-spacing: -0.005em;
    text-align: center;
}
.pka-head-note {
    font-size: 0.76rem; font-weight: 400; color: var(--ink-faint);
    margin-inline-start: 0.45rem;
}
.pka-rule {
    height: 1px; background: var(--line); margin: 0.7rem -1.05rem 1rem -1.05rem;
}
.pka-rail-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--ink-faint);
    writing-mode: vertical-rl; margin: 1rem auto 0 auto;
}

/* --- sources --- */
.pka-blank {
    text-align: center; padding: 2.2rem 0.5rem; color: var(--ink-faint);
}
.pka-blank-mark { font-size: 1.4rem; opacity: 0.5; }
.pka-blank-title {
    font-size: 0.88rem; color: var(--ink-soft); margin-top: 0.6rem; font-weight: 500;
}
.pka-blank-text { font-size: 0.79rem; line-height: 1.6; margin-top: 0.35rem; }

.pka-row { padding: 0.45rem 0.4rem; margin: 0 -0.4rem; border-radius: 8px; }
div[data-testid="stHorizontalBlock"]:has(.pka-row):hover .pka-row {
    background: var(--canvas);
}
.pka-row-name {
    font-size: 0.845rem; font-weight: 500; color: var(--ink); line-height: 1.45;
    word-break: break-word;
}
.pka-row-meta {
    font-size: 0.715rem; color: var(--ink-faint); margin-top: 0.1rem;
    letter-spacing: 0.03em;
}
.pka-ok { color: var(--blue-ink); font-weight: 600; }
.pka-dim { color: var(--ink-faint); font-weight: 600; }
.pka-bad { color: var(--bad); font-weight: 600; }
.pka-row-error {
    font-size: 0.74rem; color: var(--bad); line-height: 1.5; margin: 0 0 0.45rem 0.4rem;
}
.pka-scope {
    font-size: 0.735rem; color: var(--ink-soft); line-height: 1.55;
    background: var(--canvas); border-radius: 8px; padding: 0.5rem 0.65rem;
    margin-top: 0.9rem;
}

/* The row's destructive action is a bare red glyph — no chrome, no label, so it adds no
   visual weight to the row it sits in. */
div[data-testid="stHorizontalBlock"]:has(button[title^="Remove"]) button {
    background: transparent; border: none; color: var(--bad);
    font-size: 0.95rem; font-weight: 400; line-height: 1;
    padding: 0; min-width: 22px; width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center;
    margin-inline-start: auto;
}
div[data-testid="stHorizontalBlock"]:has(button[title^="Remove"]) button:hover {
    color: var(--bad); background: var(--bad-wash); border-radius: 6px;
}
div[data-testid="stHorizontalBlock"]:has(button[title^="Remove"]) button:focus-visible {
    outline: 2px solid var(--bad); outline-offset: 2px;
}

/* --- chat --- */
.pka-hero { text-align: center; padding: 3.2rem 1rem 0 1rem; }
.pka-hero-mark { font-size: 1.9rem; opacity: 0.5; }
.pka-hero-title {
    font-size: 1.35rem; font-weight: 600; color: var(--ink); margin-top: 0.9rem;
    letter-spacing: -0.01em;
}
.pka-hero-sub {
    font-size: 0.87rem; color: var(--ink-faint); margin-top: 0.45rem; line-height: 1.6;
}
.pka-asked {
    font-size: 0.87rem; color: var(--ink-soft); line-height: 1.55;
    background: var(--canvas); border-radius: 12px; padding: 0.7rem 0.9rem;
    margin-bottom: 1.3rem;
}
.pka-reply { font-size: 0.99rem; line-height: 1.78; color: var(--ink); }
.pka-declined {
    background: var(--canvas); border-radius: 12px; padding: 0.9rem 1rem;
}
.pka-declined-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 0.3rem;
}
.pka-declined-text {
    color: var(--ink-soft); font-size: 0.93rem; line-height: 1.65;
}
/* The chat panel is a flex column: the title + history live in their own scrollable
   region (`.pka-history` marks its container below) above the composer, which is a
   plain flex sibling rendered *after* that region closes — structurally outside the
   scrolling area, so it can never end up below the scrollable content or need
   scrolling to reach. Scoped to the chat column only (via `.pka-composer`, only ever
   present there) so Sources/Models keep scrolling as a whole card, unchanged. */
div[data-testid="stColumn"]:has(.pka-composer) {
    display: flex; flex-direction: column; overflow: hidden;
}
/* Every Streamlit layout block in this chain defaults to `min-height: auto`, which
   lets a flex item grow past its allotted space to fit its content instead of
   shrinking into it — the actual reason the composer used to be pushed off-panel.
   Resetting it at each level (this is Streamlit's own single shared block wrapping
   both the history region and the composer) is what makes the next rule's `flex: 1 1
   auto` on the history region actually take effect within the card's fixed height. */
div[data-testid="stColumn"]:has(.pka-composer) > div[data-testid="stVerticalBlock"] {
    min-height: 0;
}
/* The direct wrapper Streamlit puts around the history container: its own default
   (`flex: 0 1 auto`, sized to content only) is what was silently preventing the
   scrollable region from ever being height-constrained — `flex: 1 1 auto` here is
   what actually makes it fill the remaining space above the composer. */
div[data-testid="stColumn"]:has(.pka-composer)
    div[data-testid="stLayoutWrapper"]:has(div[data-testid="stVerticalBlock"] .pka-history) {
    flex: 1 1 auto; min-height: 0; overflow: hidden;
}
div[data-testid="stColumn"]:has(.pka-composer)
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .pka-history) {
    flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden;
    scrollbar-width: thin;
}
/* Keeps a short hero/history bottom-anchored within that scrollable region. */
.pka-grow { min-height: 12vh; }

/* --- inline citations --- */
.pka-ref {
    display: inline-block; font-size: 0.69rem; font-weight: 600;
    color: var(--blue-ink); background: var(--blue-wash); border-radius: 4px;
    padding: 0.02rem 0.3rem; margin: 0 0.1rem; vertical-align: 0.12em; line-height: 1.5;
}
.pka-evidence {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--ink-faint); margin: 1.4rem 0 0.5rem 0;
}
[data-testid="stPopover"] button {
    font-size: 0.755rem; font-weight: 500; color: var(--ink-soft);
    background: var(--card); border: 1px solid var(--line); border-radius: 999px;
    padding: 0.1rem 0.7rem;
}
[data-testid="stPopover"] button:hover {
    color: var(--blue-ink); border-color: var(--blue); background: var(--blue-wash);
}
[data-testid="stPopover"] button:focus-visible {
    outline: 2px solid var(--blue); outline-offset: 2px;
}
.pka-cite-name {
    font-size: 0.8rem; font-weight: 600; color: var(--ink); margin-bottom: 0.1rem;
}
.pka-cite-meta {
    font-size: 0.69rem; color: var(--ink-faint); letter-spacing: 0.03em;
    padding-bottom: 0.5rem; margin-bottom: 0.55rem; border-bottom: 1px solid var(--line);
}
.pka-cite-body { font-size: 0.85rem; color: var(--ink-soft); line-height: 1.65; }

/* --- models --- */
.pka-field-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--ink-faint); margin: 0 0 0.5rem 0;
}
.pka-model {
    font-size: 0.87rem; font-weight: 600; color: var(--ink); line-height: 1.4;
    word-break: break-word;
}
.pka-kv {
    display: flex; gap: 0.5rem; font-size: 0.745rem; line-height: 1.65;
    margin-top: 0.3rem;
}
.pka-kv-key {
    color: var(--ink-faint); flex: none; min-width: 2.6rem; letter-spacing: 0.03em;
}
.pka-kv-val { color: var(--ink-soft); word-break: break-all; font-variant-numeric: tabular-nums; }
.pka-chip {
    display: inline-block; font-size: 0.69rem; font-weight: 600; border-radius: 999px;
    padding: 0.1rem 0.55rem; letter-spacing: 0.02em;
}
.pka-chip-local { background: var(--blue-wash); color: var(--blue-ink); }
.pka-chip-cloud { background: #fdf3e7; color: #8a5a1b; }
.pka-note {
    font-size: 0.74rem; color: var(--ink-soft); line-height: 1.6;
    background: var(--canvas); border-radius: 8px; padding: 0.55rem 0.65rem;
    margin-top: 0.7rem;
}
.pka-probe-ok { font-size: 0.76rem; color: var(--blue-ink); font-weight: 600; }
.pka-probe-bad { font-size: 0.76rem; color: var(--bad); font-weight: 600; }
.pka-probe-detail {
    font-size: 0.72rem; color: var(--ink-faint); line-height: 1.5; margin-top: 0.2rem;
    word-break: break-word;
}

/* --- messages --- */
.pka-alert {
    background: var(--bad-wash); border: 1px solid #f2ddd9;
    border-left: 3px solid var(--bad); border-radius: var(--r-ctl);
    padding: 0.65rem 0.85rem; color: var(--bad); font-size: 0.815rem;
    line-height: 1.55; margin-bottom: 0.9rem;
}
.pka-busy {
    font-size: 0.78rem; color: var(--blue-ink); font-weight: 500; margin-top: 0.4rem;
}

/* --- controls --- */
div[data-testid="stForm"] {
    border: none; border-top: 1px solid var(--line);
    padding: 0.95rem 0 0 0; margin-top: 1.2rem; background: var(--card);
}
.stTextArea textarea {
    border-radius: 12px !important; background: var(--canvas) !important;
    border-color: var(--line) !important;
    font-size: 0.93rem !important; line-height: 1.6 !important;
    /* Direction and alignment follow the typed script, with no toggle. */
    unicode-bidi: plaintext; text-align: start;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px var(--blue-wash) !important;
}
/* The composer grows with its content — upwards, because the form sits at the foot of
   the pane. `field-sizing` covers browsers that have it; the script below covers the
   rest. Both stop at the same ceiling, after which the textarea scrolls. */
div[data-testid="stForm"] .stTextArea textarea {
    field-sizing: content;
    min-height: 44px !important; max-height: 220px !important;
    overflow-y: auto; resize: none;
    padding: 0.6rem 0.2rem !important;
    /* Bare field: the bar has no box of its own to sit inside. */
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
div[data-testid="stForm"] .stTextArea textarea:focus {
    border: none !important; box-shadow: none !important;
}
/* The composer row *is* the chat box: one bordered container holding the field and the
   send button, so the button reads as sitting inside it. The focus ring moves to the
   container, which is why the field above drops its own border. */
/* Streamlit wraps the textarea in its own bordered BaseWeb shell; with the row below
   drawing the frame, that shell would read as a second box inside the first. */
div[data-testid="stForm"] .stTextArea div[data-baseweb="textarea"],
div[data-testid="stForm"] .stTextArea div[data-baseweb="base-input"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important; border-radius: 0 !important;
}
/* "Press Ctrl+Enter to submit" contradicts this composer: Enter already sends. */
div[data-testid="stForm"] [data-testid="InputInstructions"] {
    display: none !important;
}
/* One flat, full-width bar — no enclosing box, no fill, no radius, no shadow. The only
   separator is the form's own top rule, so the field and the send button read as sitting
   directly on the panel. */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
    align-items: center; gap: 0.4rem;
    background: transparent; border: none; border-radius: 0;
    padding: 0; margin: 0;
}
/* A fixed-size flex child, never inside the scrollable history region above it (see
   the chat-column rules earlier) — this is what actually keeps it permanently in
   place, rather than relying on `position: sticky` within Streamlit's generated
   markup. */
div[data-testid="stColumn"]:has(.pka-composer) div[data-testid="stForm"] {
    flex: 0 0 auto;
    background: var(--card); margin-top: 0;
    padding: 0.7rem 0 0.2rem 0;
}
/* The send button is a circle inside the capsule — scoped to the chat composer via
   its `.pka-composer` marker, since a plain `stForm` selector would also catch the
   Models panel's ordinary "Save" buttons, which are forms too. Recent Streamlit
   builds name form-submit buttons by test id rather than `kind`, so both spellings
   are pinned. */
div[data-testid="stColumn"]:has(.pka-composer) .stFormSubmitButton button,
div[data-testid="stColumn"]:has(.pka-composer) button[data-testid*="FormSubmit"] {
    height: 40px; min-width: 40px; width: 40px; padding: 0;
    border-radius: 50%; font-size: 1.45rem; line-height: 1;
    background: var(--blue); border: 1px solid var(--blue); color: #fff;
}
/* Idle until there is something to send: `:placeholder-shown` is true only while the
   field is empty, so the button greys out and colours up as the user types — no script
   and no rerun involved. The row is the common ancestor of field and button. */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(
    textarea:placeholder-shown
) .stFormSubmitButton button,
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(
    textarea:placeholder-shown
) button[data-testid*="FormSubmit"] {
    background: var(--line); border-color: var(--line); color: var(--ink-faint);
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px; background: var(--canvas); border-color: var(--line);
}
/* The dialog's scrim: a neutral dark wash so the page recedes behind it. Streamlit
   derives its own from the theme background, which tints it cream. */
div[data-testid="stDialog"] {
    background-color: rgba(28, 28, 30, 0.45) !important;
    backdrop-filter: none;
}
.stButton button, .stFormSubmitButton button {
    border-radius: var(--r-ctl); border: 1px solid var(--line); font-weight: 500;
    box-shadow: none; color: var(--ink);
    transition: color 130ms ease, background-color 130ms ease,
        border-color 130ms ease, opacity 130ms ease;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    border-color: var(--blue); color: var(--blue-ink); background: var(--blue-wash);
}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    background: var(--blue); border-color: var(--blue); color: #fff; font-weight: 600;
}
.stButton button[kind="primary"]:hover,
.stFormSubmitButton button[kind="primary"]:hover {
    background: var(--blue-hover); border-color: var(--blue-hover); color: #fff;
}
.stButton button:focus-visible, .stFormSubmitButton button:focus-visible {
    outline: 2px solid var(--blue); outline-offset: 2px;
}
/* The send button inverts on hover — white with a blue outline and glyph — matching
   every other control here, rather than darkening like a generic primary button.
   Same `.pka-composer` scoping as the circle button itself. */
div[data-testid="stColumn"]:has(.pka-composer) .stFormSubmitButton button:hover,
div[data-testid="stColumn"]:has(.pka-composer) button[data-testid*="FormSubmit"]:hover {
    background: var(--card); border-color: var(--blue); color: var(--blue-ink);
}
/* With nothing typed there is nothing to invite: the idle circle stays quiet on hover. */
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(
    textarea:placeholder-shown
) .stFormSubmitButton button:hover,
div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]:has(
    textarea:placeholder-shown
) button[data-testid*="FormSubmit"]:hover {
    background: var(--line); border-color: var(--line); color: var(--ink-soft);
}
/* Pane collapse/expand controls read as quiet glyphs, not buttons. */
div[data-testid="stHorizontalBlock"]:has(button[title*="panel"]) button,
div[data-testid="stColumn"]:has(.pka-rail) button {
    background: transparent; border: none; color: var(--ink-faint);
    font-size: 0.95rem; font-weight: 400;
    /* Fixed box so the glyph is the same target open or folded — the folded rail is a
       much narrower column, and an auto-width button would shrink with it. */
    width: 30px; min-width: 30px; height: 30px; padding: 0;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 8px; margin-inline: auto;
}
div[data-testid="stHorizontalBlock"]:has(button[title*="panel"]) button:hover,
div[data-testid="stColumn"]:has(.pka-rail) button:hover {
    color: var(--blue-ink); background: var(--blue-wash);
}
div[data-testid="stColumn"]:has(.pka-rail) .stButton {
    display: flex; justify-content: center;
}
[data-testid="stExpander"] { border-color: var(--line); box-shadow: none; }

/* Narrow screens: Streamlit stacks the columns; keep every pane reachable. */
@media (max-width: 1000px) {
    .block-container { padding: 0.8rem 0.6rem; }
    /* Stacked panels can't each own a viewport-height scroller, so the page scrolls
       normally again here and every panel grows to its content. */
    html, body { overflow: auto; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    .block-container {
        height: auto; overflow: visible;
    }
    div[data-testid="stColumn"]:has(.pka-card),
    div[data-testid="stColumn"]:has(.pka-rail) {
        height: auto; min-height: 0; overflow: visible; margin-bottom: 0.7rem;
    }
    div[data-testid="stColumn"]:has(.pka-composer) {
        display: block; overflow: visible;
    }
    div[data-testid="stColumn"]:has(.pka-composer)
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .pka-history) {
        overflow: visible;
    }
    .pka-grow { min-height: 0; }
    .pka-rail-label { writing-mode: horizontal-tb; margin: 0.4rem 0 0 0; }
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(_CSS.replace("__FONT__", _FONT_URL), unsafe_allow_html=True)


def _get_client() -> ApiClient:
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient(get_settings().api_base_url)
    return st.session_state["api_client"]


def _load_documents(client: ApiClient) -> tuple[list[dict], str | None]:
    """Fetch the document list once per render; every pane reads the same result."""
    try:
        return client.list_documents(), None
    except ApiError as error:
        return [], str(error)


def _alert(message: str) -> None:
    st.markdown(
        f'<div class="pka-alert" dir="auto">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def _pane_header(
    title: str,
    *,
    note: str = "",
    collapse: tuple[str, str] | None = None,
    control_side: str = "right",
) -> None:
    """Card header: the title centred in the panel, plus an optional collapse glyph.

    The glyph sits in one of two equal-width outer columns, so the title's middle column
    is itself centred in the panel — that is what keeps the title centred rather than
    merely left-aligned beside the control. `control_side` puts the glyph on the pane's
    outer edge, so the two side panes fold towards their own screen edge symmetrically.
    """
    heading = (
        f'<div class="pka-head">{html.escape(title)}'
        f'<span class="pka-head-note">{html.escape(note)}</span></div>'
    )
    if collapse is None:
        st.markdown(heading, unsafe_allow_html=True)
    else:
        state_key, glyph = collapse
        left, name, right = st.columns([1, 5, 1], vertical_alignment="center")
        control = left if control_side == "left" else right
        with name:
            st.markdown(heading, unsafe_allow_html=True)
        with control:
            if st.button(glyph, key=f"fold-{state_key}", help=f"Hide {title} panel"):
                st.session_state[state_key] = False
                st.rerun()
    st.markdown('<div class="pka-rule"></div>', unsafe_allow_html=True)


def _render_rail(title: str, state_key: str, glyph: str) -> None:
    """A folded pane: just enough to say what it is and to bring it back."""
    st.markdown('<div class="pka-rail"></div>', unsafe_allow_html=True)
    if st.button(glyph, key=f"unfold-{state_key}", help=f"Show {title} panel"):
        st.session_state[state_key] = True
        st.rerun()
    st.markdown(
        f'<div class="pka-rail-label">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


# --- sources pane ---


def _enable_folder_picking() -> None:
    """Let the file dialog select a whole folder, not just files.

    Presentation-only: it sets the directory attributes the browser already understands
    on Streamlit's own file input. Non-PDF/DOCX files a folder happens to contain are
    still filtered out by the uploader's own type check, and dropping a folder onto the
    dropzone already worked. Silently a no-op if the input isn't found.
    """
    components.html(
        """
        <script>
        (function () {
            try {
                var doc = window.parent.document;
                function bind() {
                    doc.querySelectorAll(
                        '[data-testid="stFileUploaderDropzone"] input[type="file"]'
                    ).forEach(function (input) {
                        input.setAttribute("webkitdirectory", "");
                        input.setAttribute("directory", "");
                    });
                }
                bind();
                new MutationObserver(bind).observe(doc.body, {
                    childList: true,
                    subtree: true,
                });
            } catch (error) {
                /* Missing elements or a sandboxed frame: file picking still works. */
            }
        })();
        </script>
        """,
        height=0,
    )


def _format_eta(seconds: float) -> str:
    """Render a duration as a short, rounded-down estimate — never false precision."""
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"~{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"~{minutes}m {rest}s" if rest else f"~{minutes}m"


def _exclude_removed(names: list[str], excluded: set[str]) -> list[str]:
    """Names still queued for indexing: every picked file not removed before starting."""
    return [name for name in names if name not in excluded]


# How often the UI re-polls a running ingestion job (ADR-17). The job itself runs
# entirely server-side now — this only paces how often the browser asks for an update,
# it does not affect ingestion speed.
_INGEST_POLL_SECONDS = 1.5


def _start_ingest_batch(client: ApiClient, files: list) -> None:
    """Start one real background ingestion job for these files (ADR-17) and remember
    its id so `_render_ingest_progress` can poll it across reruns."""
    job = client.start_ingestion([(f.name, f.getvalue()) for f in files])
    st.session_state["ingest_job_id"] = job["job_id"]
    st.session_state["ingest_started_at"] = time.monotonic()


def _render_ingest_progress(client: ApiClient) -> None:
    """Poll the running job's real, server-reported progress and rerun.

    The job itself advances entirely server-side (ADR-17); this only reflects its
    current state — nothing here simulates progress or invents an ETA before real
    completed-file timing exists to base one on.
    """
    job_id = st.session_state["ingest_job_id"]
    try:
        job = client.get_ingestion_job(job_id)
    except ApiError as error:
        st.session_state["sources_error"] = str(error)
        del st.session_state["ingest_job_id"]
        return

    finished_statuses = {"indexed", "already_indexed", "failed", "skipped"}
    results = [f for f in job["files"] if f["status"] in finished_statuses]
    if results:
        _render_upload_outcomes(results)

    if job["status"] == "completed":
        _finish_ingest_job(job)
        st.rerun()
        return

    total = job["total"]
    completed = job["completed"]
    current_name = job["current_filename"] or ""
    st.progress(completed / total if total else 0.0)
    st.markdown(
        f'<div class="pka-row-meta">{completed} / {total} files &middot; '
        f'<span dir="auto">{html.escape(current_name)}</span> &middot; Indexing…</div>',
        unsafe_allow_html=True,
    )
    eta = job["eta_seconds"]
    st.markdown(
        '<div class="pka-row-meta">Estimated time remaining: '
        f'{_format_eta(eta) if eta is not None else "Estimating…"}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Cancel remaining", use_container_width=True):
        try:
            client.cancel_ingestion_job(job_id)
        except ApiError as error:
            st.session_state["sources_error"] = str(error)

    time.sleep(_INGEST_POLL_SECONDS)
    st.rerun()


def _finish_ingest_job(job: dict) -> None:
    st.session_state["upload_outcomes"] = job["files"]
    started_at = st.session_state.pop("ingest_started_at", None)
    st.session_state["upload_elapsed"] = (
        time.monotonic() - started_at if started_at is not None else None
    )
    st.session_state["sources_error"] = None
    st.session_state["upload_excluded"] = set()
    del st.session_state["ingest_job_id"]


@st.dialog("Add sources")
def _add_sources_dialog(client: ApiClient) -> None:
    if "ingest_job_id" in st.session_state:
        _render_ingest_progress(client)
        return

    st.markdown(
        '<div class="pka-blank-text">PDF or DOCX, as many at once as you like — pick '
        "several files, or drag a folder onto the box. Each file is indexed on its own, "
        "so one bad file never blocks the rest.</div>",
        unsafe_allow_html=True,
    )
    picked = st.file_uploader(
        "Choose files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.checkbox("Browse for a folder instead of files"):
        _enable_folder_picking()
        st.markdown(
            '<div class="pka-blank-text">The browse dialog now selects a folder; '
            "only its PDF and DOCX files are taken.</div>",
            unsafe_allow_html=True,
        )

    excluded: set[str] = st.session_state.setdefault("upload_excluded", set())
    to_index: list = []
    if picked:
        excluded &= {f.name for f in picked}  # drop stale names if the picker changed
        for f in picked:
            if f.name in excluded:
                continue
            to_index.append(f)
            row, action = st.columns([9, 1], vertical_alignment="center")
            with row:
                st.markdown(
                    f'<div class="pka-row"><div class="pka-row-name" dir="auto">'
                    f"{html.escape(f.name)}</div></div>",
                    unsafe_allow_html=True,
                )
            with action:
                if st.button(
                    "✕", key=f"exclude-{f.name}", help=f"Remove {f.name} from this batch"
                ):
                    excluded.add(f.name)
                    st.rerun()
        st.markdown(
            f'<div class="pka-row-meta">{len(to_index)} of {len(picked)} '
            f'{"file" if len(picked) == 1 else "files"} ready</div>',
            unsafe_allow_html=True,
        )

    if st.button(
        "Add to knowledge base",
        type="primary",
        use_container_width=True,
        disabled=not to_index,
    ):
        _start_ingest_batch(client, to_index)
        st.rerun()

    # Reset lives here rather than in the panel: it is the documented recovery for an
    # embedding-model change refused by ADR-8's fingerprint check ("reset the knowledge
    # base first"), so the UI must keep offering it somewhere. Tucked behind this dialog
    # and a confirmation, since it deletes every source permanently. Hidden while a
    # batch is queued so the knowledge base can't be reset out from under it.
    st.markdown('<div class="pka-rule"></div>', unsafe_allow_html=True)
    _render_reset(client)


def _render_upload_outcomes(results: list[dict]) -> None:
    for outcome in results:
        state_class, label = _STATUS_LABEL[outcome["status"]]
        st.markdown(
            f'<div class="pka-row"><div class="pka-row-name" dir="auto">'
            f'{html.escape(outcome["filename"])}</div>'
            f'<div class="pka-row-meta {state_class}">{label}</div></div>',
            unsafe_allow_html=True,
        )
        if outcome["status"] == "failed" and outcome.get("error"):
            st.markdown(
                f'<div class="pka-row-error" dir="auto">'
                f'{html.escape(outcome["error"])}</div>',
                unsafe_allow_html=True,
            )


def _render_source_row(client: ApiClient, doc: dict) -> None:
    detail, action = st.columns([9, 1], vertical_alignment="center")
    with detail:
        st.markdown(
            f'<div class="pka-row"><div class="pka-row-name" dir="auto">'
            f'{html.escape(doc["filename"])}</div>'
            f'<div class="pka-row-meta">{html.escape(doc["file_type"].upper())}'
            f' · <span class="pka-ok">Indexed</span></div></div>',
            unsafe_allow_html=True,
        )
    with action:
        # A bare glyph, not a labelled button: the `help` text is what names the action
        # for a screen reader and on hover, so the row stays visually quiet.
        if st.button(
            "✕",
            key=f"remove-{doc['document_id']}",
            help=f"Remove {doc['filename']}",
        ):
            try:
                client.delete_document(doc["document_id"])
            except ApiError as error:
                st.session_state["sources_error"] = str(error)
            else:
                st.session_state["upload_outcomes"] = None
            st.rerun()


def _render_reset(client: ApiClient) -> None:
    if not st.session_state.get("confirm_reset"):
        if st.button("Reset knowledge base", use_container_width=True):
            st.session_state["confirm_reset"] = True
            st.rerun()
        return

    st.markdown(
        '<div class="pka-blank-text">This removes every source permanently.</div>',
        unsafe_allow_html=True,
    )
    keep, wipe = st.columns(2)
    with keep:
        if st.button("Keep", use_container_width=True):
            st.session_state["confirm_reset"] = False
            st.rerun()
    with wipe:
        if st.button("Reset", type="primary", use_container_width=True):
            try:
                client.reset_knowledge_base()
            except ApiError as error:
                st.session_state["sources_error"] = str(error)
            else:
                st.session_state["upload_outcomes"] = None
                st.session_state["exchanges"] = []
            st.session_state["confirm_reset"] = False
            st.rerun()


def _render_sources(
    client: ApiClient, documents: list[dict], fetch_error: str | None
) -> None:
    st.markdown('<div class="pka-card"></div>', unsafe_allow_html=True)
    count = len(documents)
    note = "" if fetch_error else f"{count}" if count else ""
    _pane_header("Sources", note=note, collapse=("sources_open", "«"))

    if st.button("＋  Add sources", use_container_width=True):
        _add_sources_dialog(client)
    elif "ingest_job_id" in st.session_state:
        # A background job is still running (see `_render_ingest_progress`, which
        # polls it): reopen the same dialog so it stays visible without another click.
        _add_sources_dialog(client)

    if st.session_state.get("sources_error"):
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        _alert(st.session_state["sources_error"])
    if st.session_state.get("upload_outcomes"):
        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        elapsed = st.session_state.get("upload_elapsed")
        if elapsed is not None:
            st.markdown(
                f'<div class="pka-row-meta">Finished in {_format_eta(elapsed)}</div>',
                unsafe_allow_html=True,
            )
        _render_upload_outcomes(st.session_state["upload_outcomes"])

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if fetch_error:
        _alert(fetch_error)
        return

    if not documents:
        st.markdown(
            '<div class="pka-blank"><div class="pka-blank-mark">▤</div>'
            '<div class="pka-blank-title">Your sources will appear here</div>'
            '<div class="pka-blank-text">Add a PDF or DOCX, then ask questions '
            "answered only from it.</div></div>",
            unsafe_allow_html=True,
        )
        return

    for doc in documents:
        _render_source_row(client, doc)

    # Stated plainly because it is the truth: retrieval has no per-source filter, so
    # every question searches the whole knowledge base. No selection controls are
    # offered, because none would change the outcome.
    st.markdown(
        f'<div class="pka-scope">Every question searches all {count} '
        f'{"source" if count == 1 else "sources"}.</div>',
        unsafe_allow_html=True,
    )


# --- models pane ---


def _kv(key: str, value: str) -> str:
    return (
        f'<div class="pka-kv"><span class="pka-kv-key">{html.escape(key)}</span>'
        f'<span class="pka-kv-val">{html.escape(value)}</span></div>'
    )


def _render_provider_editor(
    client: ApiClient, *, slot: str, label: str, provider: dict, locality: bool = False
) -> None:
    """One provider's current values, plus a form to replace them at runtime.

    A blank API key field means "keep the credential already in effect" — the masked
    value shown is display-only, and the real secret never has to round-trip through
    the browser just to be preserved (R-08's masking rule extended to this form).
    """
    chip = ""
    if locality:
        local = provider["is_local"]
        cls = "pka-chip-local" if local else "pka-chip-cloud"
        chip = f'<span class="pka-chip {cls}">{"Local" if local else "Cloud"}</span>'
    st.markdown(
        f'<div class="pka-field-label">{html.escape(label)}{chip}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_kv("key", provider["masked_key"]), unsafe_allow_html=True)

    with st.form(f"{slot}-settings-form", clear_on_submit=False):
        model = st.text_input("Model", value=provider["model"])
        base_url = st.text_input("Base URL", value=provider["base_url"])
        api_key = st.text_input(
            "API key",
            value="",
            type="password",
            placeholder="Leave blank to keep the current key",
        )
        # Deliberately not `type="primary"`: that hands the button Streamlit's own theme
        # accent, which is neither this UI's palette nor the "＋ Add sources" button this
        # one is meant to be indistinguishable from. As an ordinary full-width button it
        # inherits exactly that button's fill, border, radius, type and hover.
        saved = st.form_submit_button("Save", use_container_width=True)

    error_key, saved_key = f"{slot}_settings_error", f"{slot}_settings_saved"
    if saved:
        try:
            if slot == "llm":
                client.update_llm_settings(
                    api_key=api_key or "",
                    base_url=base_url or "",
                    model=model or "",
                )
            else:
                client.update_embedding_settings(
                    api_key=api_key or "",
                    base_url=base_url or "",
                    model=model or "",
                )
        except ApiError as error:
            st.session_state[error_key] = str(error)
            st.session_state[saved_key] = False
        else:
            st.session_state[error_key] = None
            st.session_state[saved_key] = True
        st.rerun()

    if st.session_state.get(error_key):
        _alert(st.session_state[error_key])
    elif st.session_state.get(saved_key):
        st.markdown(
            '<div class="pka-note">Saved — in effect now. Runtime changes are not '
            "written to <code>.env</code>; restarting the app reverts to it."
            "</div>",
            unsafe_allow_html=True,
        )


def _render_probe_result(result: dict) -> None:
    for label, key in (("Language model", "llm"), ("Embeddings", "embedding")):
        check = result.get(key) or {}
        if check.get("ok"):
            st.markdown(
                f'<div class="pka-probe-ok">✓ {label} reachable</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="pka-probe-bad">✕ {label} unreachable</div>'
                f'<div class="pka-probe-detail" dir="auto">'
                f'{html.escape(str(check.get("detail") or ""))}</div>',
                unsafe_allow_html=True,
            )


def _render_models(client: ApiClient) -> None:
    st.markdown('<div class="pka-card"></div>', unsafe_allow_html=True)
    _pane_header("Models", collapse=("models_open", "»"), control_side="left")

    try:
        config = client.read_settings()
    except ApiError as error:
        _alert(str(error))
        return

    _render_provider_editor(client, slot="llm", label="Language model", provider=config["llm"])

    st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)
    _render_provider_editor(
        client,
        slot="embedding",
        label="Embeddings",
        provider=config["embedding"],
        locality=True,
    )

    if config["embedding"]["is_local"]:
        st.markdown(
            '<div class="pka-note">Documents are embedded on this machine and never '
            "leave it. A hosted embedding provider instead — faster for large "
            "libraries, but document text is sent to that provider — can be set "
            "above, or via <code>EMBEDDING_BASE_URL</code>/<code>EMBEDDING_API_KEY</code>"
            "/<code>EMBEDDING_MODEL</code> in <code>.env</code> to persist across "
            "restarts.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pka-note">Document text is sent to this hosted provider to be '
            "embedded. Switch back to a local server above, or set "
            "<code>EMBEDDING_BASE_URL</code> in <code>.env</code> to make that the "
            "default across restarts.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    probing = st.empty()
    if st.button("Test connection", use_container_width=True):
        probing.markdown(
            '<div class="pka-busy">Contacting providers…</div>',
            unsafe_allow_html=True,
        )
        try:
            st.session_state["probe"] = client.test_providers()
        except ApiError as error:
            st.session_state["probe"] = None
            st.session_state["probe_error"] = str(error)
        else:
            st.session_state["probe_error"] = None
        st.rerun()

    if st.session_state.get("probe_error"):
        _alert(st.session_state["probe_error"])
    elif st.session_state.get("probe"):
        _render_probe_result(st.session_state["probe"])

    st.markdown(
        '<div class="pka-note">Changing the embedding model above is refused (HTTP '
        "409) while sources are indexed under a different one — reset the knowledge "
        "base and re-add your sources afterwards if you want to switch anyway."
        "</div>",
        unsafe_allow_html=True,
    )


# --- chat pane ---


def _render_citation(number: int, source: dict) -> None:
    with st.popover(f"[{number}] {source['filename']}"):
        excerpt = html.escape(source["excerpt"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="pka-cite-name" dir="auto">'
            f'{html.escape(source["filename"])}</div>'
            f'<div class="pka-cite-meta">{html.escape(source["file_type"].upper())}'
            f" · excerpt {html.escape(str(source['chunk_id']))}</div>"
            f'<div class="pka-cite-body" dir="auto">{excerpt}</div>',
            unsafe_allow_html=True,
        )


def _answer_html(answer: str) -> str:
    """Render answer text with `[n]` markers as styled inline references.

    Markers are already guaranteed resolvable by `rag/generator.py`, so this only
    styles them; an answer containing none renders as plain prose (the fallback).
    """
    rendered = []
    for part in _CITATION_SPLIT.split(answer):
        if _CITATION_SPLIT.fullmatch(part):
            rendered.append(f'<span class="pka-ref">{html.escape(part[1:-1])}</span>')
        else:
            rendered.append(html.escape(part).replace("\n", "<br>"))
    return "".join(rendered)


def _render_exchange(question: str, reply: dict | None, error: str | None = None) -> None:
    st.markdown(
        f'<div class="pka-asked" dir="auto">{html.escape(question)}</div>',
        unsafe_allow_html=True,
    )

    if error:
        # A failed request never erases the question that caused it — it stays in
        # history with the error attached directly beneath it, not as a separate
        # banner disconnected from which question actually failed.
        _alert(error)
        return

    assert reply is not None
    if reply["is_refusal"]:
        st.markdown(
            '<div class="pka-declined">'
            '<div class="pka-declined-label">Not in your sources</div>'
            f'<div class="pka-declined-text" dir="auto">'
            f'{html.escape(reply["answer"])}</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="pka-reply" dir="auto">{_answer_html(reply["answer"])}</div>',
        unsafe_allow_html=True,
    )

    sources = reply.get("sources") or []
    if not sources:
        return

    st.markdown('<div class="pka-evidence">Sources</div>', unsafe_allow_html=True)
    for row_start in range(0, len(sources), 3):
        row = sources[row_start : row_start + 3]
        for column, offset in zip(st.columns(3), range(len(row))):
            with column:
                _render_citation(row_start + offset + 1, row[offset])


def _inject_composer_behavior() -> None:
    """Enter submits a non-empty question; Shift+Enter inserts a newline.

    Also grows the textarea with its content, up to the ceiling the stylesheet sets, for
    browsers without `field-sizing`. Presentation-only client-side behavior — no state,
    no business logic. Defensive by construction: if the expected elements aren't found
    (a Streamlit DOM change, a stripped/sandboxed iframe), it silently does nothing
    rather than breaking the form.
    """
    components.html(
        """
        <script>
        (function () {
            try {
                var doc = window.parent.document;
                function autoGrow(area) {
                    area.style.height = "auto";
                    var ceiling = 220;
                    area.style.height = Math.min(area.scrollHeight, ceiling) + "px";
                }
                function bind() {
                    var area = doc.querySelector('[data-testid="stForm"] textarea');
                    var form = doc.querySelector('[data-testid="stForm"]');
                    if (!area || !form || area.dataset.pkaBound) return;
                    area.dataset.pkaBound = "1";
                    autoGrow(area);
                    area.addEventListener("input", function () { autoGrow(area); });
                    area.addEventListener("keydown", function (event) {
                        // isComposing: never submit mid-IME (Persian/CJK input).
                        if (event.key !== "Enter" || event.shiftKey ||
                            event.isComposing) return;
                        event.preventDefault();
                        if (!area.value.trim()) return;
                        var send = form.querySelector(
                            'button[data-testid*="FormSubmit"], ' +
                            'button[kind="primary"], button[type="submit"]'
                        );
                        if (send) send.click();
                    });
                }
                bind();
                new MutationObserver(bind).observe(doc.body, {
                    childList: true,
                    subtree: true,
                });
            } catch (error) {
                /* Cross-origin or missing elements: no-op, native behavior stands. */
            }
        })();
        </script>
        """,
        height=0,
    )


def _render_chat(
    client: ApiClient, documents: list[dict], fetch_error: str | None
) -> None:
    st.markdown('<div class="pka-card"></div>', unsafe_allow_html=True)

    # Everything above the composer — title, hero/history — lives in its own container
    # so only *this* region scrolls (`.pka-history` below marks it for the CSS); the
    # composer, rendered as a sibling after this `with` block ends, is never inside it.
    with st.container():
        st.markdown('<div class="pka-history"></div>', unsafe_allow_html=True)
        _pane_header("Chat", note="grounded in your sources")

        # In-memory only (per ADR-1's UI-is-a-thin-client boundary): a plain list in
        # `st.session_state`, not a session/conversation model — no persistence, no
        # naming, no effect on retrieval or generation. Lost on refresh/restart by
        # design. Each entry is `(question, reply, error)`, exactly one of `reply`/
        # `error` set — a failed request is still a real entry, never dropped.
        exchanges: list[tuple[str, dict | None, str | None]] = (
            st.session_state.setdefault("exchanges", [])
        )

        # Only pre-flight validation (no question typed, no sources yet, API
        # unreachable) uses this banner — a failed *request* is shown inline with its
        # question in history instead (see `_render_exchange`).
        if st.session_state.get("chat_error"):
            _alert(st.session_state["chat_error"])

        if exchanges:
            for question_asked, reply, error in exchanges:
                _render_exchange(question_asked, reply, error)
            st.markdown('<div class="pka-grow"></div>', unsafe_allow_html=True)
        else:
            if fetch_error:
                body = (
                    '<div class="pka-hero-title">The service is unreachable</div>'
                    '<div class="pka-hero-sub">Start the API, then reload this page.</div>'
                )
            elif not documents:
                body = (
                    '<div class="pka-hero-title">Add a source to begin</div>'
                    '<div class="pka-hero-sub">Answers come only from your own documents, '
                    "with the supporting passage attached.</div>"
                )
            else:
                count = len(documents)
                body = (
                    '<div class="pka-hero-title">Ask your sources</div>'
                    f'<div class="pka-hero-sub">{count} '
                    f'{"source" if count == 1 else "sources"} ready — in English, '
                    "Persian, or both.</div>"
                )
            st.markdown(
                f'<div class="pka-hero"><div class="pka-hero-mark"></div>{body}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="pka-grow"></div>', unsafe_allow_html=True)

    # The composer is always typeable — a half-written question survives adding the
    # first source. Whether it can be answered is decided on submit, not by disabling.
    pending = st.empty()
    # Marks this column for the chat-composer-only CSS below (the capsule, the circular
    # send button) so it never bleeds into the Models panel's own, ordinary forms.
    st.markdown('<div class="pka-composer"></div>', unsafe_allow_html=True)
    # A widget's own session-state key cannot be reassigned after that widget has
    # already run in the same script pass — only *before* it is next instantiated. So
    # a successful submit sets this flag instead of clearing `ask_question` directly;
    # it is applied here, on the following rerun, before the widget below is created.
    if st.session_state.pop("_clear_ask_question", False):
        st.session_state["ask_question"] = ""

    with st.form("ask", clear_on_submit=False):
        field, send = st.columns([12, 1], vertical_alignment="bottom")
        with field:
            question = st.text_area(
                "Your question",
                key="ask_question",
                placeholder="Ask something…",
                label_visibility="collapsed",
                height=68,
            )
        with send:
            submitted = st.form_submit_button(
                "→", type="primary", help="", use_container_width=True
            )
    _inject_composer_behavior()

    if not submitted:
        return

    # Clears the composer the instant Send is clicked, regardless of what happens next
    # (applied on the next rerun, before the widget is instantiated again, per the
    # flag's check above) — a real chat app never leaves the last thing you typed
    # sitting in the box once you've sent it.
    st.session_state["_clear_ask_question"] = True

    if not question.strip():
        st.session_state["chat_error"] = "Please type a question first."
        st.rerun()

    if fetch_error:
        st.session_state["chat_error"] = fetch_error
        st.rerun()

    if not documents:
        st.session_state["chat_error"] = (
            "Add a source first — answers come only from your own documents."
        )
        st.rerun()

    pending.markdown(
        '<div class="pka-busy">Searching your sources…</div>', unsafe_allow_html=True
    )
    st.session_state["chat_error"] = None
    try:
        result = client.submit_query(question)
    except ApiError as error:
        # The question is never lost to a provider/API failure: it is recorded in
        # history with its error attached, not silently dropped.
        exchanges.append((question, None, str(error)))
    else:
        exchanges.append((question, result, None))
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Private Knowledge Assistant",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.session_state.setdefault("sources_open", True)
    st.session_state.setdefault("models_open", True)

    client = _get_client()
    documents, fetch_error = _load_documents(client)

    st.markdown(
        '<div class="pka-brand"><span class="pka-brand-dot"></span>'
        '<span class="pka-brand-name">Private Knowledge Assistant</span>'
        '<span class="pka-brand-sub">Local-first · grounded answers with citations'
        "</span></div>",
        unsafe_allow_html=True,
    )

    sources_open = st.session_state["sources_open"]
    models_open = st.session_state["models_open"]
    left, middle, right = st.columns(
        [1.15 if sources_open else 0.16, 2.3, 1.15 if models_open else 0.16],
        gap="small",
    )

    with left:
        if sources_open:
            _render_sources(client, documents, fetch_error)
        else:
            _render_rail("Sources", "sources_open", "»")
    with middle:
        _render_chat(client, documents, fetch_error)
    with right:
        if models_open:
            _render_models(client)
        else:
            _render_rail("Models", "models_open", "«")


if __name__ == "__main__":
    main()
