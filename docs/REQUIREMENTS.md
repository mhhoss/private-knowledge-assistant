# Requirements

Source of truth for **intended behavior**. The code is the source of truth for what
currently exists (see [README](../README.md#status)).

## Problem

A user has private documents and no trustworthy way to query them. General assistants
cannot see the documents; pasting them into a hosted chat leaks them and gives answers
with no verifiable provenance.

This project answers questions strictly from the user's own documents and shows where
each answer came from.

## Scope

Single-user, local, v1.

| ID | Requirement |
| --- | --- |
| R-01 | Upload one or more PDF/DOCX files |
| R-02 | Parse, normalize, chunk, embed, and persist uploaded documents automatically |
| R-03 | Persist the index across restarts (local Chroma) |
| R-04 | Answer natural-language questions using only retrieved document context |
| R-05 | Return citations with filename and the supporting excerpt for every answer |
| R-06 | List indexed documents |
| R-07 | Delete an individual document, or reset the entire knowledge base |
| R-08 | Switch LLM/embedding provider — via environment variables (persists across restarts), or from the UI at runtime (process-local only) — with no code change either way |
| R-09 | Process each uploaded file independently and report per-file success or failure |
| R-10 | Support English and Persian documents and queries, including mixed-language text |
| R-11 | Refuse to use an index built with an incompatible embedding configuration |

R-08 covers OpenAI-compatible providers and gateways only. A runtime (UI) provider
change is validated with one real call before it takes effect, is never written to
`.env`, and — for the embedding provider specifically — is refused (R-11) rather than
applied if it would conflict with an already-indexed knowledge base; see ARCHITECTURE.md
ADR-14.

### Ingestion outcomes (R-09)

Each file in an upload is an independent unit of work with one outcome:

| Outcome | Meaning |
| --- | --- |
| `indexed` | Chunks embedded and persisted |
| `already_indexed` | Identical content is already indexed; nothing was re-embedded |
| `failed` | A reason the user can act on (unreadable file, no extractable text, provider error) |

Rules: one file's failure never rolls back or blocks another file's success; a failed file
leaves no chunks behind; a well-formed upload request succeeds even if every file fails —
per-file failures are response data, not HTTP errors. Identity is content, not filename
([ADR-3](ARCHITECTURE.md#decisions)): the same bytes uploaded under a new name report
`already_indexed` and keep the originally indexed filename.

### Languages (R-10)

English and Persian are both first-class in v1: documents in either language, queries in
either language, Persian documents containing English terms, and queries in one language
against documents in the other. No stage of the pipeline may assume monolingual content.
Mechanism: [ADR-9](ARCHITECTURE.md#decisions).

The configured embedding model must support both languages; this is a deployment
constraint, not something code can enforce.

### Embedding compatibility (R-11)

Vectors produced by different embedding models are not comparable. Changing the embedding
configuration against a non-empty index must be detected and refused with an actionable
message, never silently queried. Recovery in v1 is a full re-index.
Mechanism: [ADR-8](ARCHITECTURE.md#decisions).

## Groundedness rules

Non-negotiable; these define correctness for R-04/R-05.

1. Retrieved document chunks are the only knowledge source for answers.
2. Generation must never bypass retrieval.
3. If retrieved context is insufficient, the system says so explicitly instead of
   answering. See [ADR-4](ARCHITECTURE.md#decisions) for how this is enforced.
4. Every answer that is not a refusal carries its sources.
5. Unsupported information is never presented as fact.

## Privacy rules

1. Documents and vector data are local by default.
2. Uploaded documents, vector data, and secrets are never committed.
3. Document content is never written to logs.

## User stories

| ID | Story | Requirements |
| --- | --- | --- |
| US-01 | Upload one or more PDF/DOCX files as my private knowledge base | R-01 |
| US-02 | Have uploads processed and indexed automatically, and be told exactly which files failed and why | R-02, R-03, R-09 |
| US-03 | Ask questions and get answers based only on my documents | R-04 |
| US-04 | See which document and which part each answer came from | R-05 |
| US-05 | See which documents are currently indexed | R-06 |
| US-06 | Delete individual documents or reset the knowledge base | R-07 |
| US-07 | Switch LLM/embedding providers — via environment variables or from the UI — and be stopped rather than given garbage results when the change invalidates the index | R-08, R-11 |
| US-08 | Use Persian documents and ask questions in Persian, with the same behavior as English | R-10 |

## Out of scope (v1)

Authentication/authorization · multi-user or multi-tenant support · cloud vector
databases · fine-tuning · multi-agent workflows · distributed infrastructure ·
billing · OCR for scanned or image-only PDFs (a common Persian document format —
such files are reported as `failed`, not silently indexed empty) · languages other
than English and Persian · non-PDF/DOCX formats · answer streaming ·
per-embedding-model collection namespacing.

Out of scope means *not built and not designed for*. Adding any of these requires a
concrete requirement first.

## Definition of done

1. PDF/DOCX upload and indexing works end to end.
2. The index survives a restart.
3. Queries return grounded answers with sources.
4. Insufficient context produces an explicit refusal, not a guess.
5. Providers are switchable through configuration alone.
6. Core logic is modular and covered by the tests described in
   [ARCHITECTURE.md](ARCHITECTURE.md#testing).
7. README setup and run instructions are complete and accurate.
8. Secrets and private runtime data are excluded from Git.
9. A multi-file upload containing one bad file indexes the rest and reports that file's
   failure, with no leftover chunks from it.
10. Persian, English, and mixed-language documents and queries all work end to end.
11. Changing the embedding model against a non-empty index is refused with an actionable
    message.

## Implementation order

Each step is independently verifiable:

1. `config.py` + `.env` contract
2. `storage/vector_store.py` (persistence, listing, deletion, embedding fingerprint)
3. `documents/processor.py` (normalize → chunk → metadata), then `parser.py` + `loader.py`
4. `rag/indexer.py` (per-file ingestion outcomes)
5. `rag/retriever.py`
6. `rag/generator.py` + `rag/engine.py`
7. `schemas/api.py` + `api/routes.py` + `main.py`
8. `streamlit_app.py`

## Trade-off priority

**Correctness > Groundedness > Reliability > Maintainability > Simplicity > Convenience**

Complexity requires a concrete requirement, not an anticipated one.
