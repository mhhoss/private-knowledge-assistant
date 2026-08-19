# Embedding model benchmark: bge-m3 vs multilingual-e5-small vs multilingual-e5-base

Status: **final**. Run 2026-08-19, fully offline (no network, no downloads — `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `uv run --offline`), sequentially (one model loaded, benchmarked,
and released via `del` + `gc.collect()` before the next started). Hardware: Intel Core
i5-6200U, 2 cores / 4 threads, CPU-only.

Methodology, corpus, and queries: `eval/README.md`, `eval/corpus/`, `eval/queries.json`
(23 queries, 6 documents, unchanged). Script: `eval/run_benchmark.py`. Raw per-model output:
`eval/results/*.json`.

Text extraction and chunking reused the app's real pipeline unmodified
(`app.documents.loader`, `app.documents.processor`, `chunk_size=1024`/`chunk_overlap=128` —
the project's documented defaults) so the 45 benchmarked chunks are exactly what production
would index. Embedding used ONNX Runtime + `tokenizers` only (no torch/sentence-transformers)
against each model's local ONNX export — the same dependency surface a bundled deployment
ships, and a stricter/faster substitute for the Ollama-served path the earlier
`EMBEDDING_MODEL_DECISION.md` measured. bge-m3 uses CLS-token pooling per its published dense
recipe; the two e5 models use mean pooling with `query: ` / `passage: ` prefixes per their
model cards. All vectors L2-normalized; scores are cosine similarity.

## Results

| Model | Top-1 accuracy | Recall@5 | Separability gap | Ingestion (ms/chunk) | Ingestion throughput | Query latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **bge-m3** | **100.0%** (19/19) | **100.0%** | **+0.024** | 2604.9 | 0.38 chunks/s | 118.1 |
| multilingual-e5-small | 94.7% (18/19) | 94.7% | −0.047 | 241.4 | 4.14 chunks/s | 12.1 |
| multilingual-e5-base | 89.5% (17/19) | 100.0% | −0.049 | 642.8 | 1.56 chunks/s | 40.2 |

*Separability gap = min top-1 score among answerable queries − max top-1 score among the 4
out-of-corpus queries. Positive means a single fixed `RETRIEVAL_MIN_SCORE` threshold can
cleanly separate "answer found" from "nothing relevant" for this corpus; negative means the
answerable and out-of-corpus score ranges overlap, so no threshold value can avoid both false
accepts and false rejects.*

### By category (top-1 accuracy)

| Model | same_language (10) | cross_lingual (9) |
| --- | --- | --- |
| bge-m3 | 100% | 100% |
| multilingual-e5-small | 100% | 88.9% (1 miss: q11) |
| multilingual-e5-base | 100% | 77.8% (2 misses: q11, q17) |

Both e5 models' misses are cross-lingual: q11 (Persian query → English compound-interest
document) retrieved the wrong document on both; q17 (English query → Persian-dominant AI
document, the query's answer phrase "artificial intelligence" also appears inline in English
in that document per `eval/queries.json`'s own note) additionally failed on e5-base.

## Analysis

**Quality**: bge-m3 is unambiguously the strongest — perfect top-1 and recall@5 across every
query and category, including the hardest cross-lingual cases, on this corpus. It is also the
only model of the three with a *positive* separability gap: its own answerable/out-of-corpus
score ranges do not overlap, meaning a fixed threshold (the project's current
`RETRIEVAL_MIN_SCORE=0.60` was tuned against bge-m3 specifically) can work at all. Both e5
models have a small *negative* gap (−0.047, −0.049): their out-of-corpus queries scored higher
at top-1 than the hardest genuine answer did, so no single threshold value avoids both letting
through an ungrounded answer and rejecting a valid one — on this corpus, at this size. This
directly threatens the groundedness rule in `docs/REQUIREMENTS.md` that the retrieval-score
gate exists to enforce.

**CPU performance**: e5-small is ~10.8x faster to embed and ~9.8x faster per query than
bge-m3; e5-base is ~4.1x faster to embed and ~2.9x faster per query. This confirms
`EMBEDDING_MODEL_DECISION.md`'s prior finding that bge-m3's CPU cost is the dominant
ingestion bottleneck, now with a second, comparable data point rather than an assumption.

**Trade-off**: the two axes point in opposite directions. The faster models are cheaper to
run but measurably worse at exactly the two things this project's `docs/REQUIREMENTS.md` and
`ARCHITECTURE.md` (ADR-9, Persian/English support; the groundedness score gate) care about
most — cross-lingual retrieval correctness and a workable fixed threshold.

## Recommendation

**Keep `BAAI/bge-m3` as the production embedding model.** This benchmark reproduces and
strengthens, rather than overturns, the decision already recorded in
`EMBEDDING_MODEL_DECISION.md`: bge-m3 is now the only model of three actually measured against
this project's own corpus/pipeline with a clean, positive score-separability gap and perfect
cross-lingual accuracy — the property the groundedness gate depends on. Neither e5 model
would be a safe drop-in replacement without re-tuning (and likely widening) the retrieval
threshold, and even then their cross-lingual weakness (77.8–88.9% vs. 100%) would remain.

The CPU cost is real and unchanged: bge-m3 embeds ~4–11x slower than the e5 alternatives on
this hardware. As before, this is a deployment-tier problem, not an architectural one — R-08
already makes the embedding backend swappable with zero code change, so if ingestion latency
becomes operationally unacceptable, the correct next step is GPU-served or hosted bge-m3
inference (same measured quality, higher throughput), not a switch to a smaller, weaker model.

## Caveats

Carried forward from `eval/README.md`: 6 documents, 23 queries, one document per topic, only
two mixed-language patterns represented, no scanned/tabular/complex-layout PDFs. The negative
separability gaps for the e5 models are a small-corpus measurement (19 answerable / 4
out-of-corpus queries) and could narrow or widen at larger scale — but the same caveat already
applied to bge-m3's original single-model result, and this run at least gives a real
comparison point instead of none.
