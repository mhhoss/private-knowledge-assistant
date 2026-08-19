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

## Quick start: Ollama (BGE-M3) + OpenRouter

This is the setup this project has actually been run and evaluated against (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s Performance section) — a local,
free embedding backend and a hosted LLM.

1. Install [Ollama](https://ollama.com) and pull the embedding model:

   ```bash
   ollama pull bge-m3
   ollama serve   # if not already running as a service
   ```

2. Get an API key from [OpenRouter](https://openrouter.ai/keys) for the LLM.

3. Fill in `.env`:

   ```bash
   LLM_API_KEY=sk-or-v1-...
   LLM_BASE_URL=https://openrouter.ai/api/v1
   LLM_MODEL=openai/gpt-4o-mini

   # Ollama serves embeddings through its OpenAI-compatible endpoint; OpenRouter does
   # not serve embeddings, so this must be set separately (see ADR-5).
   EMBEDDING_API_KEY=ollama
   EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
   EMBEDDING_MODEL=bge-m3
   ```

4. Follow **Run** below to start both processes, then open
   http://127.0.0.1:8501, upload a document, and ask a question.

## Run

The API and the UI are two separate processes; the UI is an HTTP client of the API
(see [ADR-1](docs/ARCHITECTURE.md#decisions)) and expects the API to already be running.
Start each in its own terminal, from the project root:

**Terminal 1 — API:**

```bash
uv run uvicorn app.main:app --reload          # http://127.0.0.1:8000
```

Wait for it to finish starting (it builds the embedding/LLM clients and opens the
vector store) before starting the UI.

**Terminal 2 — UI:**

```bash
uv run streamlit run streamlit_app.py         # http://127.0.0.1:8501
```

Streamlit opens the UI in your browser automatically; if not, visit the URL above.

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
| `EMBEDDING_BATCH_SIZE`, `EMBEDDING_TIMEOUT_SECONDS` | Chunks per embedding request, and its timeout — tuned for slow backends by default |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | Vector store location and collection |
| `DATA_DIR` | Uploaded originals |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SCORE` | Retrieval breadth and the groundedness cutoff |
| `API_BASE_URL` | Where the Streamlit UI reaches the API |

`CHUNK_SIZE` and `CHUNK_OVERLAP` are character counts, not tokens. The embedding model
must support both English and Persian, and changing it invalidates an existing index —
the app will refuse to use a mismatched one until the knowledge base is reset.

`RETRIEVAL_MIN_SCORE`'s default (0.47) was measured against `BAAI/bge-m3` on the
versioned `eval/` corpus (23 queries, 6 documents), not chosen arbitrarily — see
[ARCHITECTURE.md open question 2](docs/ARCHITECTURE.md#open-questions) for the
evaluation. Re-measure if the embedding model changes.

`EMBEDDING_BATCH_SIZE` (default 10) and `EMBEDDING_TIMEOUT_SECONDS` (default 120) default
to values safe for slow backends; raise `EMBEDDING_BATCH_SIZE` for a fast/hosted provider
to improve ingestion throughput. See the Performance / Scale section below.

### Choosing an embedding backend

Embedding throughput is the ingestion bottleneck (see Performance below), and it depends
almost entirely on where the embedding model runs:

- **CPU** (e.g. local Ollama on a laptop/desktop with no GPU) — what this project has
  been measured against: ~2.5–2.8s per ~750-character chunk. A large document (hundreds
  of chunks) can take several minutes to index. Keep the low default
  `EMBEDDING_BATCH_SIZE` here so requests stay well under `EMBEDDING_TIMEOUT_SECONDS`.
- **GPU** (Ollama or another server with CUDA/ROCm/Metal) — the same model served on a
  GPU is typically an order of magnitude or more faster per chunk than CPU; raise
  `EMBEDDING_BATCH_SIZE` (e.g. 50–100) once you've confirmed request latency stays
  comfortably under the timeout.
- **Hosted API** (e.g. OpenAI's own embedding endpoint) — fastest and most consistent,
  at the cost of sending document text to that provider (no longer local-only for that
  data). Raise `EMBEDDING_BATCH_SIZE` toward the provider's own batch limit.

Whichever backend you choose, `EMBEDDING_MODEL` must support both English and Persian,
and changing it later requires resetting the knowledge base (see above).

## Performance / Scale

| Metric                    | Measured result                  |
| ------------------------- | -------------------------------- |
| Chroma scale tested       | 400K vectors                     |
| Chroma query latency      | ~4–7 ms                          |
| End-to-end retrieval      | ~500 ms                          |
| Storage                   | ~5 KB / vector                   |
| Main ingestion bottleneck | Embedding throughput             |
| PDF extraction            | Poppler `pdftotext -layout`      |
| Languages evaluated       | English / Persian / Mixed        |

A scale evaluation (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full results)
found that Chroma query latency stayed flat through 400,000 vectors. The main ingestion
bottleneck is embedding throughput, while large-corpus ingestion time depends heavily
on the embedding backend and hardware (CPU vs. GPU vs. hosted API). Local CPU-served
embedding can be orders of magnitude slower than a properly served GPU or hosted backend.

**Measured-environment caveat:** every number above comes from one evaluation run on one
machine, one corpus, and one real embedding/LLM pair (local CPU-served `BAAI/bge-m3` via
Ollama, `openai/gpt-4o-mini` via OpenRouter — see ARCHITECTURE.md's Performance section
and open questions for the full setup and sample sizes). Treat these as directional, not
guaranteed, for a different machine, corpus, or provider — re-measure before relying on
them for capacity planning.

## Privacy

`data/` (uploaded originals), `chroma_db/` (vector store), and `.env` are local runtime
data and are Git-ignored. Never commit them, and never log document contents.
