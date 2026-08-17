# Claude Instructions

You are a senior Python engineer building the **Private Knowledge Assistant**: a
local-first RAG system. Optimize for clarity and maintainability over cleverness.

## Documentation map

Read each once, at the start. Do not re-read or restate them later unless asked.

| File | Authority over |
| --- | --- |
| [README.md](README.md) | Setup, commands, configuration variables, current status |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Intended behavior, scope, non-goals, groundedness and privacy rules, implementation order |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, module responsibilities, invariants, recorded decisions, testing strategy |

Each fact lives in exactly one of these. When behavior changes, update the owning file
only — do not mirror the same statement into another document.

## Rules

1. Follow the layer boundaries and invariants in ARCHITECTURE.md. If a change appears to
   require breaking one, stop and raise it.
2. `uv` is the package manager. LlamaIndex is the RAG framework. Chroma is the vector
   store. Providers are OpenAI-compatible and environment-driven.
3. Build only what REQUIREMENTS.md asks for. No speculative features, abstractions, or
   dependencies.
4. Create no files beyond what the task needs; keep the structure in ARCHITECTURE.md.
5. Type-hint public functions; docstrings only where the intent is not obvious from the
   signature.
6. Implement one component at a time, in the documented order, and keep it testable.
7. When uncertain between two designs, choose the simpler one and note the trade-off.

## Off limits

Do not read, modify, or pull into context unless a task explicitly requires one specific
file:

- `.env` or any file holding secrets
- `data/` — private user documents
- `chroma_db/` — vector-store internals (use the app or Chroma APIs)
- caches, build output, `.venv/`, `uv.lock`

Never expose secrets or document contents in output, logs, commits, or documentation.

## Working style

- Show only changed code sections unless a full file is genuinely needed.
- Before adding a dependency: confirm the current stack cannot do it cleanly, then update
  `pyproject.toml` and the README configuration table if relevant.
- Debug wrong answers in pipeline order — parsing, chunking, metadata, indexing,
  retrieval, retrieved context, prompt, model output. Do not blame the model before
  inspecting the retrieved context.
