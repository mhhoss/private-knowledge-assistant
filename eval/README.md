# Embedding evaluation set

A small, versioned, reproducible corpus + query set used to compare candidate
embedding models on Persian/English/mixed retrieval. Built 2026-08-19. This
supersedes the earlier ad-hoc evaluation described in `docs/ARCHITECTURE.md`
open questions 2 and 4, whose corpus and queries were never persisted.

## Corpus (`eval/corpus/`)

Six real documents sourced from Wikipedia (CC BY-SA 4.0) via the MediaWiki
API's plain-text extract endpoint (`action=query&prop=extracts&explaintext=1`),
truncated to the introduction plus three sections per article for a small,
predictable size. Exact source URL, revision ID, and retrieval timestamp for
each document are recorded in `eval/corpus/manifest.json`. No model or vendor
files were downloaded — only this article text (~579 KB across six API
responses).

The user's own private/user-authored document found during earlier corpus
scoping (`~/Downloads/...Market Validation...docx`) was deliberately **not**
included, per instruction.

| Document ID | Language | Format | Source article | Topic |
| --- | --- | --- | --- | --- |
| `doc-en-pdf-solar-energy` | English | PDF | Solar energy | renewable energy, physics |
| `doc-en-docx-compound-interest` | English | DOCX | Compound interest | personal finance |
| `doc-fa-pdf-coffee` | Persian | PDF | قهوه (Coffee) | food/agriculture, etymology, history |
| `doc-fa-docx-bicycle` | Persian | DOCX | دوچرخه (Bicycle) | history, technology, health |
| `doc-mixed-pdf-ai-fa` | Mixed (Persian-dominant, English AI/CS terms and names inline) | PDF | هوش مصنوعی (Artificial intelligence) | technology |
| `doc-mixed-docx-persian-language-en` | Mixed (English-dominant, embedded Persian script/terms) | DOCX | Persian language | linguistics, history |

Each document keeps its source's natural structure: a title, an introduction,
and three `Heading`-styled sections — giving real headings, prose, and (in the
Persian documents) numbers, dates, and named entities, without any authored
synthetic content.

**Known limitation, consistent with production (ADR-12):** `pdftotext -layout`
extraction of the two Persian-script PDFs (`doc-fa-pdf-coffee`,
`doc-mixed-pdf-ai-fa`) shows the same character/word-spacing degradation
already documented in `docs/ARCHITECTURE.md` for LibreOffice-generated PDFs
with variable-weight Arabic-script fonts (words split by spurious spaces,
e.g. "شطرنج" → "ش طرن ج"). This was not fixed here — it reproduces a known,
already-documented limitation of the production PDF pipeline, so the
benchmark measures embedding-model behavior against the same degraded input
production would actually see, rather than an artificially clean corpus. The
Persian DOCX documents (`doc-fa-docx-bicycle`,
`doc-mixed-docx-persian-language-en`) are unaffected, since DOCX text is read
directly rather than through `pdftotext`.

## Queries (`eval/queries.json`)

23 queries, each authored by reading the actual extracted document text and
manually verifying the expected answer before writing the query — not
generated from assumptions about document content.

- **Language**: 12 English, 11 Persian.
- **Category**:
  - `same_language` (10) — query language matches the target document's
    dominant language.
  - `cross_lingual` (9) — query language differs from the target document's
    dominant language, with no shared surface tokens (e.g. a Persian query
    about the English-only solar energy document).
  - `out_of_corpus` (4) — topics manually verified absent from all six
    documents (intermittent fasting, learning Chinese, blockchain
    proof-of-stake, traditional Iranian bread-baking).
- Every non-out-of-corpus query records `expected_doc_ids`, the document(s)
  that should be retrieved as relevant. `out_of_corpus` queries record an
  empty list.
- One query (`q17`) is flagged in its `note` as a partially-shared-surface-term
  case (the mixed AI document contains the English phrase "artificial
  intelligence" inline even though the query is answered by Persian prose) —
  worth reading before interpreting its score as purely cross-lingual.

## Reproducibility

- `eval/corpus/manifest.json` records, per document: source URL, MediaWiki
  revision ID, retrieval timestamp, and which article sections were used.
- The same corpus and query file are meant to be reused unchanged across every
  candidate embedding model benchmarked — do not regenerate or edit either
  file per-model.
- To regenerate from scratch: refetch each `manifest.json` URL at the recorded
  revision ID, rebuild DOCX files preserving intro + first three sections as
  headings, and convert the three PDF-designated documents via
  `soffice --headless --convert-to pdf` (LibreOffice), matching the project's
  own PDF-generation/extraction pipeline.

## Representativeness caveats

This is still a small corpus (6 documents, 23 queries) and should not be
treated as exhaustive:
- One document per topic — no test of ranking among multiple same-topic
  documents.
- Two mixed-language documents represent only two mixing patterns
  (Persian-dominant-with-English-terms, English-dominant-with-Persian-script);
  real product documents may mix languages differently (e.g. sentence-level
  code-switching).
- No scanned/image-based PDFs, tables, or complex layouts are represented.
