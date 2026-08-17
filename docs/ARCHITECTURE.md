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
Infrastructure  app/storage/vector_store.py  app/config.py
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
├── api/routes.py        Thin HTTP layer: validate, delegate, return schemas.
├── schemas/api.py       Request/response contracts. Independent of FastAPI and of
│                        the domain/storage types they mirror — `api/routes.py`
│                        translates between the two.
├── documents/
│   ├── loader.py        File intake, type dispatch, document_id assignment.
│   ├── parser.py        PDF (pypdf) and DOCX (python-docx) text extraction.
│   └── processor.py     Normalization, chunking, metadata propagation.
├── rag/
│   ├── indexer.py       One document's chunks → embeddings → vector store, with
│   │                    failure compensation. Never loops over files.
│   ├── retriever.py     Query → relevant chunks. No generation.
│   ├── generator.py     Context + query → answer + sources, or a refusal. No store,
│   │                    filesystem, or retrieval access — takes its own `ContextChunk`
│   │                    type, never `retriever.RetrievedChunk`.
│   └── engine.py        Orchestrates both flows: `ingest_file(s)` (per-file loading,
│                        chunking, indexing, and outcome reporting) and `answer_query`
│                        (retrieve → decide → generate → cite). The only module that
│                        calls both `rag/retriever.py` and `rag/generator.py`.
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
each built once at startup, in `main.py`'s `lifespan`, and reused for every request;
`EmbeddingMismatchError` therefore fails application startup, not a request.**
`app/main.py` stores each on `app.state`; `api/routes.py`'s dependencies (`_get_settings`
etc.) only read `request.app.state`, never construct anything (invariant 5 stays
satisfied — `config.py` is still the only place credentials are read, just read once).
This is safe because the app is local and single-user: there is no per-request identity
or tenancy that would demand a fresh client, so one long-lived client per process is
strictly simpler than reopening a Chroma client and rebuilding provider clients on every
call. Opening the `VectorStore` in `lifespan` is also where the ADR-8 fingerprint check
now runs: a mismatch raises `EmbeddingMismatchError` out of `lifespan`, which fails
startup — the process never begins serving requests against an index it cannot safely
read or write, rather than every request paying for a re-check that can only ever have
one answer for the lifetime of the process. Recovery is out-of-process either way: fix
`EMBEDDING_MODEL` back or clear `chroma_db/`, then restart — `POST /reset` cannot help,
since the process that would serve it never finishes starting up.

Superseded interim design: an earlier revision, written before `main.py` existed,
resolved this by opening a fresh `VectorStore` per request and translating a mismatch to
an HTTP 409 in the route dependency. That was the correct choice for the reasons given
there — no lifecycle owner existed yet, and a fresh-per-request store made a 409
possible without one — but it re-opened a Chroma client on every request and offered no
real recovery advantage over failing fast, since `POST /reset` already required the same
successfully-opened store and so 409'd identically either way. Once `main.py` existed to
own a startup hook, fail-fast became simpler and no worse for recovery, so it replaced
the per-request check rather than living alongside it.

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
stand in for real providers here too.

Unit-test `streamlit_app.py`'s `ApiClient` against `httpx.MockTransport` — request
shape, response decoding, and error-message translation — never a running UI or a
running API; there is no browser-level test in this project.

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

1. **Python target.** `.python-version` pins 3.14 while `pyproject.toml` requires
   ≥ 3.11. Which is authoritative for CI and deployment?
2. **`RETRIEVAL_MIN_SCORE` value and metric.** The default is a placeholder; the correct
   cutoff depends on the embedding model and needs empirical tuning against real documents,
   in both languages and cross-lingually (ADR-9). Same for `CHUNK_SIZE` / `CHUNK_OVERLAP` /
   `RETRIEVAL_TOP_K`. The collection is created with cosine distance so that scores are
   comparable to a fixed threshold at all.
3. **Upload limits.** No maximum file size, page count, or concurrent-upload behavior is
   specified.
4. **Cross-lingual semantic retrieval quality is untested against a real embedding
   model.** `tests/integration/test_retriever.py` confirms a Persian query retrieves an
   English-only document when the two share a literal token (e.g. "Kubernetes"), but the
   test embedding is a bag-of-tokens hash, not a semantic model — it cannot demonstrate
   that a *purely* Persian query retrieves a *purely* English document with no shared
   surface form, which is the actual cross-lingual claim in R-10. Whether a given
   `EMBEDDING_MODEL` clears `RETRIEVAL_MIN_SCORE` for such queries can only be verified
   against real documents and a real provider once one is configured.
5. **Sentinel-based model refusal detection is untested against a real LLM.**
   `rag/generator.py` asks the model to emit a fixed token (`[[INSUFFICIENT_CONTEXT]]`)
   verbatim when it cannot answer, and treats that token appearing anywhere in the
   response as a refusal. Real models can wrap instructed tokens in extra commentary,
   translate them, or occasionally ignore the instruction under a high `temperature`
   (the default `OpenAI` LLM client is currently built with framework defaults — no
   `temperature=0` override yet). A substring match is deliberately lenient to reduce
   false negatives, at the cost of a theoretical false positive if a real answer happens
   to contain that exact bracketed string. Needs verification against the configured
   provider once one is available; if unreliable, the alternative is structured/function-
   call output, which is a larger change and not justified without evidence it's needed.
6. **RTL text order from PDF extraction, for producers that emit visual-order glyphs.**
   `parser.py` uses `pypdf`'s "layout" extraction mode rather than its default "plain"
   mode — plain mode's bidi heuristic was found, empirically, to silently drop entire
   runs of text on lines mixing right-to-left and left-to-right content (see
   `tests/unit/test_parser.py::TestPdfRtl`), which is a groundedness risk, not just a
   display one. Layout mode never drops content, but it also does not attempt bidi
   correction: a PDF whose producer wrote right-to-left glyphs in visual rather than
   logical stream order still extracts mirror-reversed, and normalization cannot repair
   word or character order. Not resolvable generically without per-document layout
   analysis, which is out of scope here; affected real documents would need re-export
   from a bidi-correct source or a different extraction library.
7. **FastAPI's built-in validation-error body is a different shape from
   `ErrorResponse`.** A malformed request (e.g. missing `query` field) gets FastAPI's own
   `{"detail": [{"loc": ..., "msg": ..., "type": ...}]}` structure — `detail` as a list of
   objects, not the string `ErrorResponse.detail` expects. Whether `api/routes.py` should
   normalize this to a single shape, or document two distinct error shapes for the API,
   is undecided.

