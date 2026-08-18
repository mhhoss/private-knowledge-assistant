# Private Knowledge Assistant

A local-first RAG application: upload private PDF/DOCX documents, index them into a
persistent local vector store, and ask questions answered **only** from those documents,
with citations.

Everything — documents, embeddings, index — stays on the local machine. Only the
configured LLM/embedding provider is remote. English and Persian documents and queries
are both supported.

- **What it does and does not do:** [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
- **How it is built and why:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Status

**Feature-complete for v1.** Unit/integration tested end to end, including real
PDF/DOCX fixtures and English/Persian/mixed-language content throughout: `app/config.py`,
`app/storage/vector_store.py`, `app/documents/loader.py`, `app/documents/parser.py`,
`app/documents/processor.py`, `app/rag/indexer.py`, `app/rag/retriever.py`,
`app/rag/generator.py`, `app/rag/engine.py`, `app/schemas/api.py`, `app/api/routes.py`,
`app/main.py`, `streamlit_app.py`.

`app/main.py` builds settings, the embedding client, the LLM client, and the
`VectorStore` once at startup (ADR-10) and wires `app/api/routes.py`'s endpoints
(`POST`/`GET /documents`, `DELETE /documents/{document_id}`, `POST /reset`,
`POST /query`) onto them. `streamlit_app.py` is a thin HTTP client of that API (ADR-1):
upload and manage documents, ask questions, and see grounded answers or refusals with
their citations. Both run commands below start.

Implementation order and the definition of done are in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Stack

Python ≥ 3.11 · `uv` · FastAPI · Streamlit · LlamaIndex · Chroma (persistent) ·
poppler (`pdftotext`) · python-docx · pydantic-settings

## Setup

PDF extraction shells out to poppler's `pdftotext`, a system package — not something
`uv sync` installs. Install it first:

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install poppler-utils

# Linux (Fedora/RHEL)
sudo dnf install poppler-utils

# macOS (Homebrew)
brew install poppler

# Windows
# Download a poppler build (e.g. https://github.com/oschwartz10612/poppler-windows),
# extract it, and add its `bin/` folder (containing pdftotext.exe) to PATH.
```

Verify with `pdftotext -v`. Without it, PDF uploads fail with a `ParsingError`
mentioning `poppler-utils`; DOCX uploads are unaffected.

```bash
uv sync
cp .env.example .env   # then fill in credentials
```

## Run

The API and the UI are two processes; the UI is an HTTP client of the API
(see [ADR-1](docs/ARCHITECTURE.md#decisions)).

```bash
uv run uvicorn app.main:app --reload          # API on http://127.0.0.1:8000
uv run streamlit run streamlit_app.py         # UI  on http://127.0.0.1:8501
```

## Test

```bash
uv run pytest
uv run pytest tests/unit -q
```

## Configuration

All configuration is environment-driven and loaded through `app/config.py`; no
provider details are hardcoded. `.env.example` is the authoritative list of variables.

The provider must be OpenAI-compatible. LLM and embedding credentials are configured
separately because some gateways (e.g. OpenRouter) serve chat completions but not
embeddings; embedding settings fall back to the LLM settings when omitted.

| Variable | Purpose |
| --- | --- |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Chat/completion provider |
| `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` | Embedding provider (optional; defaults to the LLM values) |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | Vector store location and collection |
| `DATA_DIR` | Uploaded originals |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SCORE` | Retrieval breadth and the groundedness cutoff |
| `API_BASE_URL` | Where the Streamlit UI reaches the API |

`CHUNK_SIZE` and `CHUNK_OVERLAP` are character counts, not tokens. The embedding model
must support both English and Persian, and changing it invalidates an existing index —
the app will refuse to use a mismatched one until the knowledge base is reset.

`RETRIEVAL_MIN_SCORE`'s default (0.60) was measured against `BAAI/bge-m3`, not chosen
arbitrarily — see [ARCHITECTURE.md open question 2](docs/ARCHITECTURE.md#open-questions)
for the evaluation. Re-measure if the embedding model changes.

## Privacy

`data/` (uploaded originals), `chroma_db/` (vector store), and `.env` are local runtime
data and are Git-ignored. Never commit them, and never log document contents.
