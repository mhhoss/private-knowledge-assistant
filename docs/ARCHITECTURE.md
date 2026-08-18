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
│   ├── parser.py        PDF (poppler `pdftotext`, via subprocess) and DOCX
│   │                    (python-docx) text extraction.
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
   correct retrieval in this sample. Caveat: n=10 queries against a 9-chunk corpus is
   small; re-measure with more real documents before trusting this as a final production
   value, and re-measure entirely if `EMBEDDING_MODEL` changes. `CHUNK_SIZE` /
   `CHUNK_OVERLAP` were swept at 256/512/1024 (with proportional overlap) against the
   same corpus and queries: retrieval correctness (correct document ranked first) was
   unchanged across all three, so the existing defaults (1024/128) were kept — this
   corpus's documents are short enough (1–2 chunks each at 1024) that the sweep mostly
   changed citation granularity, not correctness; it does not rule out `CHUNK_SIZE`
   mattering more on longer real documents. `RETRIEVAL_TOP_K` was not swept
   independently; `top_k=5` combined with the corrected `RETRIEVAL_MIN_SCORE` produced
   reasonable results in this sample. The collection is created with cosine distance so
   that scores are comparable to a fixed threshold at all.
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
   defaults, no `temperature=0` override): 3/3 out-of-corpus queries triggered the
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

