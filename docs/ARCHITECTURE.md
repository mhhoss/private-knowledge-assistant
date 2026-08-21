# Architecture

Boundaries, responsibilities, invariants, and the decisions behind them.
Behavioral requirements live in [REQUIREMENTS.md](REQUIREMENTS.md).

## Layers

Dependencies point downward only. A module may import from its own layer or below,
never above.

```text
UI              streamlit_app.py                  (HTTP client of the API)
Transport       app/main.py  app/api/  app/schemas/
Orchestration   app/rag/engine.py
Domain          app/documents/  app/rag/{indexer,retriever,generator}.py
Infrastructure  app/storage/vector_store.py  app/config.py  app/observability.py
```

Consequences:

- Domain and orchestration code contains no FastAPI, Streamlit, or HTTP types.
- `storage/` is the only module that imports Chroma. Chroma types never surface in
  `rag/` or `api/` signatures.
- Provider SDK/LlamaIndex integration wiring is confined to `config.py` (client
  construction) and the modules that consume the clients — never spread across layers.
- Routes contain no chunking, retrieval, or generation logic.

## Structure

Target layout. The packages exist; individual modules are created as they are
implemented, at these paths and nowhere else.

```text
app/
├── main.py              FastAPI app construction and wiring. No business logic.
├── config.py            Typed settings (pydantic-settings) + provider client construction.
├── observability.py     Structured (JSON) logging setup: format and destination only —
│                        never decides what gets logged (ADR-15).
├── api/routes.py        Thin HTTP layer: validate, delegate, return schemas.
├── schemas/api.py       Request/response contracts. Independent of FastAPI and of
│                        the domain/storage types they mirror — `api/routes.py`
│                        translates between the two.
├── documents/
│   ├── loader.py        File intake, type dispatch, document_id assignment.
│   ├── parser.py        PDF (poppler `pdftotext`, via subprocess) and DOCX
│   │                    (python-docx) text extraction.
│   └── processor.py     Normalization, chunking, metadata propagation, and rejecting
│                        malformed extraction before it reaches the indexer (ADR-18).
├── rag/
│   ├── indexer.py       One document's chunks → embeddings → vector store, with
│   │                    failure compensation. Never loops over files.
│   ├── retriever.py     Query → relevant chunks. No generation.
│   ├── generator.py     Context + query → answer + sources, or a refusal. No store,
│   │                    filesystem, or retrieval access — takes its own `ContextChunk`
│   │                    type, never `retriever.RetrievedChunk`.
│   ├── engine.py        Orchestrates both flows: `ingest_file(s)` (per-file loading,
│   │                    chunking, indexing, and outcome reporting) and `answer_query`
│   │                    (retrieve → decide → generate → cite). The only module that
│   │                    calls both `rag/retriever.py` and `rag/generator.py`.
│   └── jobs.py          Background ingestion job tracking (ADR-17): `JobStore`,
│                        `IngestionJob`/`FileProgress`, and `run_ingestion_job`, the
│                        function `api/routes.py` hands to `BackgroundTasks`. Delegates
│                        every file to `engine.ingest_file`; never reimplements
│                        ingestion logic.
└── storage/vector_store.py   All Chroma persistence and queries.

tests/unit/               Deterministic components.
tests/integration/        Ingestion and query flows.
docs/                     This documentation.
data/                     Uploaded originals (git-ignored, private).
chroma_db/                Persistent vector store (git-ignored).
```

`streamlit_app.py` is UI only: input, display, and API calls.

## Data flow

**Ingestion** — `file → loader → parser → processor → indexer → vector store`,
once per file, independently ([ADR-7](#decisions)).

**Query** — `query → api → engine → normalize → retriever → [cutoff check] → generator → answer + sources`

Both paths validate the collection's embedding fingerprint before touching vectors
([ADR-8](#decisions)).

Chunk metadata, required at every stage:

```text
document_id   stable, content-derived identifier
filename      original filename, shown in citations
file_type     pdf | docx
chunk_id      unique within the document
```

## Invariants

Violating any of these is a bug, not a style preference.

1. Source metadata is never dropped or rewritten after `processor.py` assigns it.
2. `generator.py` receives context as an argument and never fetches it itself.
3. The engine is the only component that sequences retrieval and generation.
4. No layer above `storage/` knows the vector store is Chroma.
5. `config.py` is the only source of credentials, model names, and provider URLs.
6. Deleting a document removes its chunks from the store, not just its listing.
7. Document text and query text pass through the same normalization function. Divergence
   silently degrades retrieval instead of failing.
8. A document is either fully indexed or absent. Partial chunk sets are never left behind.
9. Vectors are never read or written without confirming the collection's embedding
   fingerprint matches the current configuration.
10. No stage branches on detected document or query language. Language-specific handling
    exists only in normalization and in the refusal-message choice.

## Decisions

Deliberate choices with real alternatives, recorded so they are not silently reversed.

**ADR-1 — The Streamlit UI talks to the API over HTTP, not by importing the engine.**
Both are viable; the API path keeps a single code path into application logic, exercises
the API during normal development, and prevents UI-side shortcuts around the engine.
Cost: two processes to run, and an `API_BASE_URL` setting. Reversible — switching to
direct imports touches only `streamlit_app.py`.

**ADR-2 — The document list is derived from vector-store metadata; there is no separate manifest.**
One source of truth, so a listing can never disagree with what is actually
indexed. Cost: listing requires a metadata scan of the collection.

**ADR-3 — `document_id` is derived from file content, not from the upload event.**
Re-uploading the same file is idempotent and detectable instead of silently duplicating
chunks. Filenames are not identifiers; two different files may share a name.

**ADR-4 — Insufficient context is enforced at two levels.** The retriever drops chunks
below `RETRIEVAL_MIN_SCORE`; if nothing survives, no LLM call is made. This is enforced
in `rag/generator.py` itself, not in `rag/engine.py`: `engine.answer_query` always calls
`generate()` with whatever `retrieve()` returned, including an empty list, rather than
special-casing it — duplicating the check in the orchestrator would be exactly the kind
of "internal logic" `engine.py` is documented not to duplicate. When context does survive,
the prompt still instructs the model to refuse — via a fixed sentinel token the model
must emit verbatim, replaced before it reaches the user — rather than fill gaps. The
retrieval-level check is what makes refusal deterministic and cheap; the prompt-level
instruction covers weak-but-passing context. Both paths converge on the same two
canned messages (English/Persian, chosen by testing the query for Arabic-script
characters), so a refusal reads identically regardless of which layer produced it, and
the model's own wording is never shown to the user for a refusal.

**ADR-5 — LLM and embedding providers are configured independently.** Some
OpenAI-compatible gateways serve chat completions but no embeddings, which would make
R-08 unsatisfiable with a single credential set. Embedding settings fall back to the LLM
settings when omitted, so the common single-provider case stays a three-variable setup.

**ADR-6 — Chroma is accessed through LlamaIndex's vector-store integration, wrapped by`storage/vector_store.py`.**
The wrapper exists so that swapping the store or the
framework is a single-module change; it is a thin boundary, not an abstraction layer with
its own model.

**ADR-7 — Ingestion is per-file, with compensation instead of a transaction.** Chroma has
no multi-write transaction, so atomicity is achieved at the only granularity that matters:
each file is indexed in a single attempt, and any failure triggers a delete-by-`document_id`
compensation before the failure is recorded. Successful files are never rolled back because
of an unrelated failure — a 40-file upload where one PDF is corrupt is 39 useful documents,
not a wasted operation. Consequence: the ingestion API returns a per-file outcome list
rather than a single status, and callers must read it. Split of duties: `rag/indexer.py`
owns the single-document unit of work and its compensation, `rag/engine.py` owns the loop
over files and converts load/parse failures into outcomes — an empty chunk list is a
caller error there, never a silently skipped file.

**ADR-8 — The collection records the embedding model that built it; mismatch is refused.**
The fingerprint is the embedding model name, stored in Chroma collection metadata and
checked before any read or write. On mismatch the operation fails with instructions to
re-index; recovery is a knowledge-base reset. Rejected alternatives: per-model collections
(unnecessary complexity for a single-user tool, and the user must still re-embed to query
old documents) and trusting the store's own dimension check (it catches only the subset of
mismatches that change dimensionality, and reports it as an opaque error). The base URL is
deliberately *not* in the fingerprint — the same model behind a different gateway is still
the same model, and including it would cause false rejections on routine gateway changes.
An empty collection adopts the current fingerprint, so reset-then-reindex is friction-free.

**ADR-9 — One language-neutral text path; no per-language branching.** Persian and English
are handled by the same pipeline (R-10), which constrains four places:

- *Normalization* is the only language-aware code. It applies NFKC (Persian PDFs commonly
  extract as Arabic presentation forms), unifies Arabic-vs-Persian codepoints that render
  identically but break matching — yeh, kaf, alef maksura, Arabic-Indic digits — strips
  tashkeel and tatweel, and removes bidi and zero-width control characters. ZWNJ (U+200C)
  is preserved: it is a letter-level semantic separator in Persian, not formatting. No
  stemming, stopword removal, or digit transliteration — there is no lexical index to serve
  them, and they would damage citation fidelity.
- *Chunking* is character-based, not token-based, and its sentence boundaries include
  Persian punctuation (`؟` `؛` `…`). Token-based sizing would silently produce very
  different chunk lengths per script, since Persian is far more token-dense in
  English-tuned BPE vocabularies. This is why the project splits text itself rather than
  configuring LlamaIndex's English/token-oriented `SentenceSplitter`.
- *Retrieval* uses one global threshold and one collection. Cross-lingual matches score
  lower than same-language ones, which makes `RETRIEVAL_MIN_SCORE` the tuning point (see
  open questions), not a reason for per-language thresholds.
- *Generation* instructs the model to answer in the language of the question and never to
  translate retrieved context unless asked. The deterministic no-context refusal (ADR-4)
  cannot call the model, so it selects a Persian or English message by testing the query
  for Arabic-script characters. That test is the only language detection in the system.

Language is deliberately not stored in chunk metadata: nothing routes on it, and a stored
guess would be wrong for mixed-language documents.

**ADR-10 — Settings, the embedding client, the LLM client, and the `VectorStore` are
each built once at startup, in `main.py`'s `lifespan`; `EmbeddingMismatchError`
therefore fails application startup, not a request.** *(Amended by ADR-14: the
LLM/embedding half of this is no longer "reused for every request" — see there. The
`VectorStore` half is unchanged and is not superseded.)* `app/main.py` stores the
`VectorStore` on `app.state`; `api/routes.py`'s `_get_vector_store` only reads
`request.app.state`, never constructs anything (invariant 5 stays satisfied —
`config.py` is still the only place credentials are read and clients are built). This is
safe because the app is local and single-user: there is no per-request identity or
tenancy that would demand a fresh client, so one long-lived store per process is
strictly simpler than reopening a Chroma client on every call. Opening the `VectorStore`
in `lifespan` is also where the ADR-8 fingerprint check now runs: a mismatch raises
`EmbeddingMismatchError` out of `lifespan`, which fails startup — the process never
begins serving requests against an index it cannot safely read or write, rather than
every request paying for a re-check that can only ever have one answer for the lifetime
of the process. Recovery from a *startup-time* mismatch is out-of-process either way:
fix `EMBEDDING_MODEL` back or clear `chroma_db/`, then restart — `POST /reset` cannot
help, since the process that would serve it never finishes starting up. (A mismatch
discovered later, from a runtime provider update, is a different situation — ADR-14
covers it: the store is never reopened, so this paragraph's "out-of-process" recovery
doesn't apply there.)

Superseded interim design: an earlier revision, written before `main.py` existed,
resolved this by opening a fresh `VectorStore` per request and translating a mismatch to
an HTTP 409 in the route dependency. That was the correct choice for the reasons given
there — no lifecycle owner existed yet, and a fresh-per-request store made a 409
possible without one — but it re-opened a Chroma client on every request and offered no
real recovery advantage over failing fast, since `POST /reset` already required the same
successfully-opened store and so 409'd identically either way. Once `main.py` existed to
own a startup hook, fail-fast became simpler and no worse for recovery, so it replaced
the per-request check rather than living alongside it.

**ADR-11 — `build_llm` overrides llama-index's `OpenAI.metadata` to tolerate non-catalog
model names, the same way `build_embedding_model` already uses `model_name=` instead of
`model=`.** Discovered during real-provider evaluation (2026-08-17): `OpenAI.metadata`
is read on every `chat()` call (via `to_payload()`, for telemetry) and computes
`context_window` by looking `model` up in a fixed table of official OpenAI model names,
raising `ValueError` for anything else — including every OpenRouter model id
(`openai/gpt-4o-mini` included) and any other non-OpenAI model served through an
OpenAI-compatible gateway. Unlike the embedding client, the chat client has no
constructor-level escape hatch, and the lookup is lazy (triggered by the first real
`chat()` call, not by construction), so it passed existing tests — which only construct
the client — silently, and only surfaced against a real gateway. Without a fix, R-08
("any OpenAI-compatible provider, no code change") is false for every provider except
literal, unprefixed OpenAI model names. The fix is a small subclass, local to
`build_llm`, that overrides `metadata` to fall back to a generous fixed context window
(128k) when the model isn't in llama-index's catalog, rather than raising; an actual
context-window mismatch still surfaces as a real provider error on the request itself,
not a silent truncation. Confined entirely to `config.py` (invariant 5): this is
provider-client construction, not a new architectural layer.

**ADR-12 — PDF extraction shells out to poppler's `pdftotext -layout` instead of using
the `pypdf` library.** Chosen 2026-08-18 after a six-way benchmark (`pypdf`, `PyMuPDF`,
`pypdfium2`, `pdfplumber`, `pdfminer.six`, `pdfmux`, and `poppler`) against a diverse
real corpus — LibreOffice- and Chrome-generated Persian/English/mixed PDFs, two
different Arabic-script fonts — measured against source-document ground truth and real
`bge-m3` retrieval (see open question 6's resolution note and the evaluation record for
the full comparison). `pypdf` recovered as little as ~11% of the true text on realistic
Persian PDFs; `poppler` averaged 81% fidelity with correct RTL/logical order on every
document tested, ahead of `PyMuPDF` (77%, AGPL-3.0/commercial-licensed, one bad case on
dense inline-bidi lines) and clearly ahead of the rest. `pdftotext` takes a file path,
not bytes, so `_extract_pdf` writes `content` to a `NamedTemporaryFile(delete=False)`
and unlinks it in a `finally` block — `delete=False` because an open, still-locked
handle can't reliably be reopened by a child process on every platform this runs on, not
because cleanup is optional. Missing-binary (`FileNotFoundError`), a non-zero exit, and
a hang (`subprocess.TimeoutExpired`, capped at 30s) all convert to the same
`ParsingError` `_extract_docx` already raises for a bad DOCX, so `loader.py` and
everything above it needed no change — `extract_text`'s `bytes -> str` contract and
`ParsingError` semantics are identical to before. Cost: PDF extraction now depends on
the `poppler-utils` system package rather than a pure-Python library (README.md's Setup
section has the per-platform install step); `pypdf` was dropped from `pyproject.toml`
since nothing imports it anymore. `tests/pdf_fixtures.py`'s synthetic byte-level PDF
builder was rewritten alongside this change — the old fixture reproduced `pypdf`-specific
decoding quirks that don't reflect how `pdftotext` (or any real PDF renderer) actually
processes RTL text; the new one reproduces the real mechanism (bidi-shaped visual-order
glyph runs, verified directly against real LibreOffice/Chrome PDFs), not a
library-specific one.

**ADR-13 — `embed_batch_size` and `timeout` on the embedding client are configurable
(`EMBEDDING_BATCH_SIZE`, `EMBEDDING_TIMEOUT_SECONDS`), defaulting to values safe for a
slow backend.** A scale evaluation (2026-08-18, see Performance below) reproduced a
concrete ingestion failure: llama-index's `OpenAIEmbedding` defaults to
`embed_batch_size=100` and `timeout=60.0`s; against a real CPU-served `BAAI/bge-m3`
measured at ~2.5-2.8s/chunk, one default-sized batch request takes ~250-280s — 4-5x the
timeout — so any document producing more than ~20-24 chunks reproducibly failed
ingestion outright (confirmed directly: a 626-chunk document failed after 709.8s of
retries). `build_embedding_model` now passes both as constructor kwargs, sourced from
`Settings` (invariant 5 holds: still the only place these are read); defaults are
`embedding_batch_size=10`, `embedding_timeout_seconds=120.0`, chosen to keep every
request comfortably under the timeout at the measured rate while leaving margin for
slower conditions or a larger `CHUNK_SIZE`. A fast/hosted embedding backend can raise
`EMBEDDING_BATCH_SIZE` for higher throughput without reintroducing the failure. This is
a client-construction tuning change, not a retrieval/chunking/Chroma/generation change —
confined to `config.py` per invariant 5, same as ADR-11.

**ADR-14 — The LLM and embedding clients may be replaced at runtime, via
`POST /settings/llm`/`POST /settings/embedding`; the `VectorStore` may not (amends
ADR-10).** R-08 originally meant "via environment variables, no code change"; a user
who wants to try a different key, model, or gateway had to edit `.env` and restart both
processes. `config.py` gains one new type, `ProviderRegistry` — a small, plain
dataclass holding the `Settings`/`LLM`/`BaseEmbedding` currently in effect, with two
methods (`replace_llm`, `replace_embedding`) that only ever reassign its own fields.
`app/main.py`'s `lifespan` builds one and stores it as `app.state.registry`;
`api/routes.py`'s `_get_settings`/`_get_embed_model`/`_get_llm` now read through it
instead of reading `app.state` directly, so every other route is unaffected by this
change. The registry constructs nothing itself — `build_llm`/`build_embedding_model`
remain the only client constructors (invariant 5 intact) — it is deliberately not a
provider abstraction, plugin system, or dependency-injection framework: a request-scoped
mutable holder, and nothing else. The two new routes follow one fixed sequence, never
skipped: build a candidate client from the requested values -> make one real call
against it (`probe_llm`/`probe_embedding`) -> only on success, hand it to the registry.
A failed probe raises before the registry is touched, so the previously active client
keeps serving every request exactly as before — there is no partial or torn state. A
request already holding the old client via dependency injection (FastAPI resolves
`Depends(_get_llm)` once per request) simply finishes with it; the registry is never
mutated in place mid-response, only reassigned between requests, so this needs no lock.
The embedding route adds one more step ADR-8 requires: after a successful probe, it
calls `VectorStore.adopt_embedding_fingerprint` — a no-op if the fingerprint already
matches, a same-instance metadata update if the collection is empty, and an
`EmbeddingMismatchError` (translated to HTTP 409) if chunks already exist under a
different model. That call never resets or deletes data itself; recovery is still the
same explicit `POST /reset` ADR-8 already prescribes. The `VectorStore` instance itself
is never replaced or reopened by any of this — the registry holds providers, not
storage, and `CHROMA_PATH`/`CHROMA_COLLECTION` remain environment-only, unchanged from
ADR-10. All of this is process-local: nothing here writes to `.env`, so a runtime change
is lost on restart and `.env` is authoritative again — a deliberate simplicity choice
(ADR-10's "no per-request identity or tenancy" reasoning extends naturally to "no
durable multi-user config store" either) that keeps `config.py` the single place
credentials are ever read, rather than adding a second, file-writing path to the same
data. `mask_secret` and a new `_sanitize_provider_error` helper apply to this path
exactly as they already did to `GET /settings`, so a probe failure's error message can
never leak a raw key back to the caller (R-08).

**ADR-15 — Structured (JSON) logging via a new `app/observability.py`, with one strict
invariant: a log record is metadata only, never document/query/answer text or
credentials.** Before this, `app/` had no logging at all — no record of what was
ingested, when, with what outcome, no query latency or retrieval scores, no visibility
into a provider failure once the HTTP response describing it was gone. `observability.py`
owns exactly two things: `configure_logging(level)`, which attaches one JSON-line
`StreamHandler` to the root logger (idempotent — a second call is a no-op, so re-entrant
startup in tests never duplicates handlers), and `log_event(logger, level, message,
**fields)`, a thin wrapper around the standard library's own `extra=` mechanism so every
call site produces the same `{message, ...fields}` shape instead of re-inventing it.
Every other module keeps using `logging.getLogger(__name__)` the normal way — this
module decides *how* logs are formatted and where they go, never *what* gets logged;
that stays with each module, same separation `config.py` already has from the code that
calls it. Four call sites were added, each named directly from a gap in the prior
foundation audit: `app/main.py` logs one `"startup complete"` event with the masked,
already-`describe_providers`-shaped provider summary (never a raw key); `app/rag/
engine.py::ingest_file` logs one `"document ingested"` event per file — status, chunk
count, and, on failure, the same error string R-09 already deems safe to show the user,
at `WARNING` for a failure and `INFO` otherwise; `app/rag/engine.py::answer_query` logs
one `"query answered"` event — elapsed time, retrieved-chunk count, min/max retrieval
score, refusal flag, and source count, but never the query or answer text, both of which
may hold sensitive user content; `app/api/routes.py`'s `/query` handler logs a
`"provider request failed"` event on the existing 502 path. `LOG_LEVEL` (default `INFO`)
is a new `Settings` field, read once like every other setting (invariant 5 unaffected —
this adds no new credential or secret). Deliberately excluded: request tracing/span IDs,
metrics export, log shipping, and a logging call in `streamlit_app.py` — the UI is a
thin HTTP client with no business logic to observe (ADR-1), and the other omissions have
no evidenced need yet, matching `STRATEGY.md`'s "avoid overengineering" — plain
`logging` + one JSON formatter is deliberately the entire mechanism, not a first piece
of a larger observability framework.

**ADR-16 — One pinned Python version (3.14), enforced by CI, resolves open question 1.**
`pyproject.toml`'s `requires-python` previously said `>=3.11`, a floor that was never
actually verified — every test in this repository has only ever run on 3.14, the
version `.python-version` already pinned. The two disagreeing forever was strictly worse
than either alone: a contributor on 3.11 would hit failures nothing here ever caught.
Resolved by narrowing `requires-python` to `==3.14.*`, matching `.python-version`
exactly, rather than the reverse (widening `.python-version`'s guarantee to a range
nothing had tested) — the smaller, evidence-backed claim. `uv` already installs
whichever Python version a project asks for if it isn't present, so a single pinned
version costs nothing extra on Windows or Linux (`STRATEGY.md`'s "Windows deployment
simple" / "no platform-specific hacks" both hold unchanged).

New minimal CI (`.github/workflows/ci.yml`) runs on every push and pull request against
this same pinned version: `ruff check`, `pyright`, then `pytest` — the same three checks
this project's own workflow already required of every change by convention, now
enforced automatically instead of only by habit. One deliberate scoping decision: `ruff`
and `pyright` are run against `app/ streamlit_app.py tests/` specifically, not the whole
repository — `eval/run_benchmark.py` has pre-existing, unrelated typing/lint issues from
its numpy/ONNX-heavy benchmarking code (never part of the shipped application, and out
of scope for this change to fix), and scoping the CI commands is far smaller than either
fixing that debt now or changing `ruff`/`pyright`'s project-wide configuration to carve
it out. Not added, deliberately: a multi-version test matrix (there is exactly one
supported version, so a matrix would test versions nothing else in this project
supports), dependency caching beyond `astral-sh/setup-uv`'s own built-in cache, and any
deployment/release step — this workflow only answers "does the existing test suite,
lint, and type-check still pass," matching `STRATEGY.md`'s standing instruction not to
add infrastructure ahead of an evidenced need.

**ADR-17 — `POST /documents` starts a background ingestion job instead of embedding
inline; a new `rag/jobs.py` tracks progress.** Diagnosed 2026-08-20 on a real
deployment (CPU-only, no GPU, local Ollama-served `bge-m3`): because the API runs as a
single uvicorn worker and `ingest_file`'s embedding calls are synchronous, one in-flight
upload blocked the *entire* process — including `GET /documents`, which touches no
embedding model at all. A small (~500KB, ~20-chunk) file could appear to take ~10
minutes wall-clock not because embedding itself is that slow (documented at
~2.5–2.8s/chunk, so ~1 minute expected), but because the whole API — and therefore the
UI — was unresponsive for the duration, compounding with the OpenAI client's own
retry/backoff on any request that brushed the timeout while starved of CPU by a
concurrent call. Fix: `POST /documents` now reads the uploaded files, creates an
`IngestionJob` via `JobStore.create`, hands `run_ingestion_job` to FastAPI's
`BackgroundTasks` (which Starlette runs via `anyio`'s worker-thread pool, off the
request-handling event loop — verified directly: a regression test drives a slow-stub
upload from one thread and asserts `GET /documents` still returns in well under the
embedding delay from another), and returns `202` with the job's initial state
immediately. `GET /documents/jobs/{job_id}` polls progress and, once finished, the same
`indexed`/`already_indexed`/`failed` per-file outcomes the old synchronous response
carried — ADR-7's per-file compensation and ADR-3's dedup are called exactly as before,
now from `run_ingestion_job` instead of `engine.ingest_files` directly; only *when* the
caller learns the outcome changed. `DELETE /documents/jobs/{job_id}` requests
cancellation of not-yet-started files only — a file already mid-embedding always
finishes normally, the same rule the UI's pre-existing "Cancel remaining" affordance
already followed.

Deliberately not Celery/Redis/a message queue: this is a single-user, local-first tool
(STRATEGY.md), and the actual requirement is "don't block the one process," not
distributed task execution. `JobStore` is an in-memory `dict[str, IngestionJob]` behind
a `threading.Lock`; a module-level `threading.Lock` in `jobs.py` is held for a job's
entire file loop so at most one job ever embeds at a time — the embedding backend
already serializes requests internally (one local model instance), so this doesn't cost
throughput, it just keeps this process's own behavior deterministic rather than opening
concurrent calls into a backend never asked to handle them. Accepted trade-off: the job
store is not persisted, so a job is gone after a restart. This is safe because nothing
is ever written to Chroma until a file's chunks are fully embedded (`indexer._write`
inserts all of a document's nodes in one call) — a restart mid-job loses that job's
*tracking*, never leaves partial vectors behind; a client sees `404` on the vanished job
id and can safely re-upload (ADR-3's content-derived id makes any already-finished file
report `already_indexed` again, cheaply). `streamlit_app.py`'s ingest UI keeps its
existing visual design (progress bar, current file, final per-file outcomes, "Cancel
remaining") but now polls the job endpoint instead of driving one HTTP call per file
itself; `_INGEST_TIMEOUT`'s long client timeout is no longer needed since neither
`POST /documents` nor a status poll ever blocks on embedding anymore.

**ADR-18 — `documents/processor.py` rejects a document whose extracted text is
dominated by Unicode control/private-use/surrogate/unassigned characters, before any
chunk reaches the indexer.** *(Threshold amended by ADR-19: 0.7% below was found to
reject an entire real document family that is indexable once embedding batch/timeout
is sized correctly; the mechanism and reasoning below are otherwise unchanged.)*
Diagnosed 2026-08-20 against a real file
(`15_abyari_ch3.pdf`, an Amuzeh-custom-encoded-font Persian PDF): its `pdftotext`
extraction is not merely low-fidelity (the known, already-documented ADR-12
territory) but embeds C0 control codepoints (`\x06`, `\x0f`, `\x18`, ...) in place
of real glyphs — a broken font/CMap mapping, not degraded-but-real text — and
sending such text to the embedding backend was observed to make individual
embedding requests take far longer than the ~2.5-2.8s/chunk baseline ADR-13
measured, the practical experience of which is a slow-to-hanging ingestion rather
than a clean failure. `process_document` now computes, per chunk (after
normalization, so legitimate invisible/bidi/mark characters are already gone), the
share of characters in Unicode categories `Cc`/`Co`/`Cs`/`Cn` — categories that
never occur in real prose in any meaningful density — and raises
`PathologicalTextError` if any chunk at least 100 characters long exceeds 0.7%.
`rag/engine.py::ingest_file` catches it the same way it already catches
`ParsingError`, converting it into a normal `failed` `IngestOutcome` (R-09) with no
chunk ever reaching `index_document` — this is a chunking-stage rejection, not a new
indexer or Chroma behavior, so ADR-7's per-file compensation is never invoked
because there is nothing to compensate. The 0.7% threshold was calibrated against a
real corpus of ~35 extracted PDFs (English, Persian, mixed, clean and broken-font):
clean documents — including ones with an occasional single mis-mapped glyph, e.g.
one stray `\x07` standing in for a citation character in an otherwise pristine
academic PDF — measured at most ~0.32% per chunk; every document sharing
`15_abyari_ch3.pdf`'s broken font measured at least ~1.6%. The threshold sits at
that gap's geometric midpoint, with over 2x margin on both sides, and is
deliberately not text length, page count, or language (invariant 10) — a genuinely
long, clean PDF in either language is unaffected; the signal is specific to
malformed extraction. Documents sharing the same broken font as `15_abyari_ch3.pdf`
are rejected too, which is correct: they carry the same risk, not merely
coincidental similarity. Recovery is the same as any other `failed` outcome:
re-export the PDF with a different tool (or a different font) and re-upload.

**ADR-19 — ADR-18's rejection threshold is raised from 0.7% to 15%, and
`EMBEDDING_BATCH_SIZE`/`EMBEDDING_TIMEOUT_SECONDS` are re-tuned from 10/120 to
5/300, so ADR-18's whole broken-font document family indexes successfully instead
of being rejected outright.** Investigated 2026-08-20 after ADR-18 was found to
reject the majority (14 of 19) of a real Persian PDF corpus sharing one broken
custom font ("Amuzeh"), not just the one file (`15_abyari_ch3.pdf`) that originally
motivated it. Two questions needed real measurement, not assumption: whether
`16_apartmani3_ch2.pdf` — one of the rejected files, specifically raised as a
possible false positive — actually belongs to the same family, and whether that
family's corruption is severe enough to justify outright rejection versus merely
needing more time. Both were answered directly against the real files and the real
embedding backend (CPU-served `BAAI/bge-m3` via Ollama), not inferred:
- **Same family, confirmed two ways.** `pdffonts` shows `16_apartmani3_ch2.pdf`
  embeds the identical custom `Amuzeh`/`Amuzeh-Bold` Type1C font as
  `15_abyari_ch3.pdf`, same `Producer: Apogee Pilot Series3 v1.0`, same 2004
  creation era. Separately, `_non_text_ratio` measured 15 and 16 as
  statistically indistinguishable (15: 1.6%-3.9% per chunk; 16: 1.6%-5.0%) — no
  ratio threshold could separate one from the other.
- **The corruption is real but not fatal.** Direct embedding-latency measurement
  (5+ samples per case, single and batched requests) found corrupted-family
  chunks take **~3-4x longer** to embed than clean chunks of the same length
  (clean: ~6.3-6.6s/chunk sustained; corrupted family: ~22-24s/chunk sustained) —
  a real, consistent, non-overlapping slowdown, not noise. At the *previous*
  defaults (`EMBEDDING_BATCH_SIZE=10`, `EMBEDDING_TIMEOUT_SECONDS=120`), a batch of
  10 corrupted chunks (~220-240s) reproducibly exceeds the 120s timeout — this is
  the exact mechanism ADR-18 was built to avoid, confirmed directly rather than
  assumed, and it explains why the family was originally deemed unrecoverable.
  But 3-4x slower is not "hanging": a whole real document from this family (a
  scale-model corpus test) completed successfully once batch/timeout were sized
  for the measured rate.

Given both findings, outright rejection was treating a *tunable* problem
(request-size-vs-timeout math) as an *unrecoverable* one (content that can never be
embedded safely). The fix addresses each: `EMBEDDING_BATCH_SIZE` drops to 5 and
`EMBEDDING_TIMEOUT_SECONDS` rises to 300, keeping a full batch at a deliberately
conservative 30s/chunk (150s) at 2x margin under the timeout even for the slower
corrupted-family case — the same "at least 2x margin" methodology ADR-13 already
established, re-measured rather than assumed stale. ADR-18's `_PATHOLOGICAL_RATIO_THRESHOLD`
rises from 0.007 to 0.15 — more than 2x above the corrupted family's measured
ceiling (6.4% across the full 19-file real corpus), so the whole family now passes,
while a document whose corruption is far more severe (no realistic batch/timeout
tuning would make embedding it worthwhile) still fails fast before any embedding
call, exactly as ADR-18 intended. This is a recalibration of two existing,
independent tuning knobs (ADR-13's and ADR-18's), not a new mechanism, an
architectural change, or a loosening of ADR-18's purpose: content that is
*genuinely* mostly noise is still rejected outright; content that is merely slower
than average is now given the time it measurably needs instead.

**ADR-20 — generation is pinned to `temperature=0.0`, the system prompt asks for
explicit step-by-step reasoning on rule/calculation questions and for conflicts
between sources to be surfaced rather than resolved, and `RETRIEVAL_TOP_K` drops
from 5 to 3.** All four are groundedness measures on the generation side, made
2026-08-20 while working the real Persian corpus of ADR-18/ADR-19. `temperature=0.0`
in `build_llm` makes the same question over the same retrieved context produce the
same answer: sampling variance was never a feature here, and a fixed temperature is
what makes the refusal sentinel (ADR-4) and the eval corpus reproducible at all —
open question 5's earlier verification ran at framework defaults, which is why it
is recorded there as a caveat rather than a result. The two added prompt rules keep
the model inside the context it was given: showing the arithmetic makes an
ungrounded step visible in the answer instead of hidden inside a number, and an
explicit conflict report is the honest output when two indexed documents disagree —
silently picking one, or averaging them, would present a choice the context does
not support as if it were a fact. Both rules are worded domain-agnostically
(invariant: no domain logic in the foundation) and add no code, no state, and no
new failure mode: they are prompt text, and every existing guard — the sentinel
refusal, `_drop_unresolvable_citations`, the `sources` contract — is unchanged.
`RETRIEVAL_TOP_K=3` narrows the context window each answer must stay grounded in;
it was *not* re-swept, so the honest status is an operational default, not a
measured one. Open question 2's sweep found no correctness difference across
3/5/8 on the `eval/` corpus, so 3 is within the range that was measured to be
safe there, but `RETRIEVAL_MIN_SCORE`'s own threshold was measured at `top_k=5` —
re-measure both together before treating either as final.

## Performance

Scale evaluation (2026-08-18) against the current production setup: poppler PDF
extraction, python-docx, real `BAAI/bge-m3` (local CPU-served Ollama), Chroma, the
current retrieval/generation pipeline. Full methodology, all measured tables, and the
sweep results are not reproduced here in detail — the findings below are what changed
production behavior or configuration; treat this as a summary, not the full record.

- **Chroma read path (retrieval) scales cleanly.** Pure-vector insertion (synthetic
  1024-dim vectors, isolating Chroma from embedding cost) up to 400,000 vectors showed
  flat query latency (3.9-7.4ms at k=5) across the entire range — no cliff, no
  measurable degradation as the collection grows. Insert throughput degraded gradually
  (1,205/s at 1k -> 262/s at 400k, ~4.6x) — real, but not a nonlinear blowup. Disk usage
  converges to ~5.0KB/vector from ~5,000 vectors onward.
- **Embedding throughput is the dominant ingestion cost, by orders of magnitude**, and
  is a hardware/deployment characteristic, not an architectural one — R-08's
  OpenAI-compatible interface already supports swapping to a faster (GPU-served or
  hosted) backend with no code change. Measured on this evaluation's CPU-only setup:
  ~2.5-2.8s/chunk steady-state for realistic (~750-char) English/Persian/mixed chunks,
  versus microseconds for parsing/chunking and single-digit milliseconds for Chroma
  reads/writes at the same scale.
- **The batch_size/timeout failure above was the one concrete blocker found and is
  fixed by ADR-13.** No other production code changed as a result of this evaluation.
- **Retrieval latency is dominated by the single query-embedding call, not corpus
  size or Chroma**: ~500-520ms steady-state, statistically identical at 10 chunks and
  at 400,000 vectors.
- **`RETRIEVAL_MIN_SCORE=0.60` remained correct for ranking at the scales tested**
  (needle-document scores and top-1 correctness were unaffected by a growing unrelated
  haystack, up to a few hundred synthetic chunks — real-embedding throughput made
  testing meaningfully larger scale impractical in-session), **but the count of
  weakly-related chunks clearing the threshold for off-topic queries grew (1 -> 2 -> 4
  of the top 5 retrieved) as the haystack grew**, shifting more weight onto the LLM's
  own prompt-level refusal (ADR-4's second line of defense) rather than the cheap
  deterministic retrieval-level cutoff. Not yet acted on: the effect was measured only
  at a few hundred chunks, and no threshold change is justified without measuring it at
  a realistic target scale.
- **`CHUNK_SIZE`/`CHUNK_OVERLAP`/`RETRIEVAL_TOP_K` sweep (512/1024/1500 x 3/5/8, real
  corpus) found no correctness or latency difference** — every combination retrieved
  the correct document at top-1. Total indexing time was roughly invariant to
  `CHUNK_SIZE` (fixed total token volume regardless of how it's split); smaller chunks
  mean more, smaller vector-store entries, not more embedding work. No change to the
  current defaults is justified by this.
- **Not yet justified**: any change to Chroma, chunking, or the retrieval/generation
  architecture. **Speculative, not yet needed**: parallelizing `ingest_files`'
  currently-sequential per-document loop — only worth considering if a fast embedding
  backend is in place and ingestion wall-clock time is still the bottleneck afterward;
  the CPU-bound embedding cost measured here dominates by orders of magnitude over any
  sequential-loop overhead, so this isn't where the time currently goes.

## Testing

Unit-test the deterministic parts: parsing, normalization, chunking, metadata
propagation, configuration resolution and fallbacks, schema validation, retrieval
filtering (fixture chunks, stubbed embeddings).

Integration-test the two critical flows: `document → parse → chunk → index` and
`query → retrieve → generate → sources`, including the insufficient-context refusal,
per-file ingestion outcomes, and embedding-fingerprint rejection.

Integration-test the transport layer (`api/routes.py`, `main.py`) through a real
`TestClient`: request validation, response-schema shape, and the startup-time embedding
fingerprint check (ADR-10) — the same stubbed LLM/embedding clients, injected via
`app.dependency_overrides` or a monkeypatched `main.build_llm`/`build_embedding_model`,
stand in for real providers here too. The runtime provider-update routes (ADR-14) get
the same treatment via a `ProviderRegistry`-backed fixture: a successful update, a
failed probe leaving the previous client and settings untouched, a raw key never
appearing in a probe-failure response, and an embedding fingerprint conflict returning
409 without touching the vector store.

Unit-test `streamlit_app.py`'s `ApiClient` against `httpx.MockTransport` — request
shape, response decoding, and error-message translation — never a running UI or a
running API; there is no browser-level test in this project.

Test logging (ADR-15) via `caplog`, at both ends: the formatter/`log_event` mechanism in
isolation (`tests/unit/test_observability.py`), and each real call site in
`app/rag/engine.py`/`app/main.py`/`app/api/routes.py` — asserting the event was emitted
with the right fields, and, for events built from real inputs (a query, an ingested
file), that the logged fields never contain the query, answer, or document text itself.

Rules: no live provider calls in tests — stub the LLM and embedding clients; tests use a
temporary Chroma path, never `chroma_db/`; assert observable behavior, not internal call
sequences. Every text-handling test covers English, Persian, and mixed content — a
suite that passes only on ASCII does not demonstrate R-10.

## Extending

| Change | Where |
| --- | --- |
| New file format | `documents/parser.py` + dispatch in `loader.py` |
| Different vector store | `storage/vector_store.py` only |
| Different chunking strategy | `documents/processor.py` |
| Normalization rules (incl. script handling) | `documents/processor.py`, one function |
| Prompt or refusal wording | `rag/generator.py` |
| New endpoint | `schemas/api.py` + `api/routes.py`, delegating to existing logic |
| New setting | `config.py` + `.env.example` + README table |

Do not add frameworks, infrastructure, or abstraction layers without a requirement in
[REQUIREMENTS.md](REQUIREMENTS.md) that needs them.

## Open questions

Unresolved by the repository; decide before the affected work begins.

1. ~~**Python target.** `.python-version` pins 3.14 while `pyproject.toml` requires
   ≥ 3.11. Which is authoritative for CI and deployment?~~ **Resolved 2026-08-20:**
   `pyproject.toml`'s `requires-python` is corrected to `==3.14.*`, matching
   `.python-version` exactly — the single supported version is 3.14, the one every
   test in this repository has actually been run against so far (a floor of `>=3.11`
   was never verified on 3.11/3.12/3.13 and offered no evidenced benefit). `uv`
   installs this exact version itself if it isn't already present, so pinning one
   version adds no deployment friction on either Windows or Linux (ADR-16). CI
   (`.github/workflows/ci.yml`, new) enforces this same version.
2. **`RETRIEVAL_MIN_SCORE` value and metric.** ~~The default is a placeholder~~
   **Measured 2026-08-17** against a real `BAAI/bge-m3` deployment (served locally via
   Ollama's OpenAI-compatible endpoint) and a real `openai/gpt-4o-mini` LLM (via
   OpenRouter), on a small corpus (6 documents: English/Persian/mixed, DOCX and
   LibreOffice-generated PDF) with 10 queries (same-language, cross-lingual with no
   shared surface terms, and off-topic/absent-from-corpus). At `top_k=5`, cosine-similarity
   top-1 scores for genuinely relevant queries were `>= 0.667`; top-1 scores for
   off-topic queries were `<= 0.567` — a clean gap, with `0.60` chosen as the midpoint
   and applied as the new default (was `0.35`, which let every off-topic query's chunks
   through and relied entirely on the LLM's own prompt-level refusal rather than the
   cheap deterministic retrieval-level cutoff this setting exists for). At `0.35`, answers
   also cited most of the collection regardless of relevance, since `generate()` cites
   every chunk it's given (by design, see `rag/generator.py`) — raising the cutoff to
   `0.60` narrowed citations back to genuinely relevant documents without losing any
   correct retrieval in this sample. ~~Caveat: n=10 queries against a 9-chunk corpus is
   small; re-measure with more real documents before trusting this as a final production
   value, and re-measure entirely if `EMBEDDING_MODEL` changes.~~ **Re-measured
   2026-08-19** against the versioned `eval/` corpus (6 documents, 23 queries,
   `eval/results/bge-m3.json`): the larger, more representative sample shifted both
   clusters down and the gap narrower than the earlier n=10 measurement suggested — top-1
   scores for genuinely answerable queries were `>= 0.487`, and for out-of-corpus queries
   `<= 0.462`. At the old default of `0.60`, several correct cross-lingual answers
   (e.g. `q11`, `q15`, `q12`, scores 0.487–0.549) would have been wrongly rejected by the
   retrieval-level cutoff and refused, despite top-1 accuracy being 1.0 without the
   cutoff — the threshold, not the retrieval, was the failure point. Default corrected to
   `0.47` (the midpoint of the new gap). Two alternative models (`multilingual-e5-base`,
   `multilingual-e5-small`) were also run against the same corpus for comparison: both
   had *negative* separability (out-of-corpus scores exceeding answerable scores) and
   lower top-1 accuracy (0.89 and 0.95 vs. `bge-m3`'s 1.0), so `bge-m3` remains the
   recommended default model — see `eval/results/*.json`. Caveat: the gap is still narrow
   (0.024) and the corpus is still small (6 documents); re-measure if `EMBEDDING_MODEL`
   changes, and re-measure again once real multi-document, multi-topic corpora are
   available. `CHUNK_SIZE` /
   `CHUNK_OVERLAP` were swept at 256/512/1024 (with proportional overlap) against the
   same corpus and queries: retrieval correctness (correct document ranked first) was
   unchanged across all three, so the existing defaults (1024/128) were kept — this
   corpus's documents are short enough (1–2 chunks each at 1024) that the sweep mostly
   changed citation granularity, not correctness; it does not rule out `CHUNK_SIZE`
   mattering more on longer real documents. `RETRIEVAL_TOP_K` was not swept
   independently; `top_k=5` combined with the corrected `RETRIEVAL_MIN_SCORE` produced
   reasonable results in this sample — the default was later narrowed to 3 for
   groundedness (ADR-20) without re-measuring this threshold against it. The
   collection is created with cosine distance so that scores are comparable to a
   fixed threshold at all.
3. **Upload limits.** No maximum file size, page count, or concurrent-upload behavior is
   specified.
4. **Cross-lingual semantic retrieval quality is untested against a real embedding
   model.** ~~Whether a given `EMBEDDING_MODEL` clears `RETRIEVAL_MIN_SCORE` for such
   queries can only be verified against real documents and a real provider once one is
   configured.~~ **Verified 2026-08-17** against real `BAAI/bge-m3`: a purely-Persian
   query with no shared surface tokens correctly retrieved an English-only DOCX and an
   English-only PDF as top-1 (scores 0.78 and 0.74), and a purely-English query correctly
   retrieved a Persian-only DOCX as top-1 (score 0.74) — see open question 2 for the
   full setup. One cross-lingual case failed (an English query against what should have
   been a Persian PDF): the cause was not retrieval or the embedding model but the PDF
   extraction bug in open question 6 below — the target document's extracted text was
   too corrupted to embed meaningfully, so it never surfaced. R-10's cross-lingual claim
   holds for `bge-m3` on DOCX and on PDFs that extract cleanly.
5. **Sentinel-based model refusal detection is untested against a real LLM.**
   ~~Needs verification against the configured provider once one is available.~~
   **Verified 2026-08-17** against real `openai/gpt-4o-mini` via OpenRouter (framework
   defaults, no `temperature=0` override — since pinned to 0.0 by ADR-20, so this
   result predates the current setting): 3/3 out-of-corpus queries triggered the
   sentinel and were converted to the canned refusal correctly, with no instances of the
   raw `[[INSUFFICIENT_CONTEXT]]` token leaking to the user, wrapped in commentary, or
   translated, across all 10 evaluation queries (English and Persian). Sample size is
   small (n=10, one model, one provider) — not proof the token can never leak or get
   translated, but no evidence of the specific failure modes this question raised for
   this model/provider pair.
6. ~~**RTL text order from PDF extraction, for producers that emit visual-order
   glyphs.**~~ **Resolved 2026-08-18 — see ADR-12.** `parser.py` extracted real Persian
   PDFs via `pypdf` as either mirror-reversed or, on more realistic real-world PDFs
   (LibreOffice/Chrome-generated, variable-font Arabic script), substantially corrupted
   (~11.5% true vocabulary recovered in the worst case, with PostScript glyph names like
   `/dotabovear` leaking into the extracted text) — a groundedness risk, since it
   presented as "no answer in the documents" rather than "text extraction actually
   failed" (see the now-resolved open question 4 note above). A six-way extractor
   benchmark against a diverse real corpus (LibreOffice- and Chrome-generated PDFs,
   two different Persian fonts, English/Persian/mixed content, real bge-m3 retrieval)
   found poppler's `pdftotext -layout` the strongest candidate — 81% average character
   fidelity and correct RTL/logical order on every document tested, chosen over
   `PyMuPDF` (77% avg, one bad case on dense inline-bidi lines, AGPL-3.0/commercial
   licensing), `pypdfium2`, `pdfplumber`, `pdfminer.six`, and `pdfmux` (all clearly
   worse). `parser.py` now shells out to `pdftotext -layout` via `subprocess`, still
   returning the same page-joined plain-text `str` it always did — everything above
   `_extract_pdf` is unaffected. Re-verified after implementation, through the real
   `extract_text()` call (not a reimplementation): 81.2% all-doc / 76.6% Persian-only
   character fidelity, identical to the benchmark, and all previously-wrong retrieval
   top-1 results now correct.
7. **FastAPI's built-in validation-error body is a different shape from
   `ErrorResponse`.** A malformed request (e.g. missing `query` field) gets FastAPI's own
   `{"detail": [{"loc": ..., "msg": ..., "type": ...}]}` structure — `detail` as a list of
   objects, not the string `ErrorResponse.detail` expects. Whether `api/routes.py` should
   normalize this to a single shape, or document two distinct error shapes for the API,
   is undecided.

