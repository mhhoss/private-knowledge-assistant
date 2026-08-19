"""Offline embedding-model benchmark: bge-m3 vs multilingual-e5-small vs multilingual-e5-base.

Runs fully offline (no network, no downloads) and sequentially — one model loaded,
benchmarked, and released before the next starts. Reuses the app's own document
parsing and chunking (`app.documents.loader`, `app.documents.processor`) so the
benchmark measures the same chunks production would index, and embeds via ONNX
Runtime + `tokenizers` only (no torch/sentence-transformers), matching what a
bundled deployment would ship. Model weights are read from a local, already-exported
ONNX cache; nothing is downloaded here.

This script does not modify production code; it only reads it (loader/processor) and
writes benchmark artifacts under eval/results/.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.documents.loader import load  # noqa: E402
from app.documents.processor import process_document  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Matches app.config.Settings defaults exactly (not read from .env, which is off-limits
# and would make this benchmark's chunking non-reproducible if it ever changed).
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5

ONNX_MODELS_ROOT = Path(
    "/run/media/mhhoss/0C2EFC0D2EFBED98/veign-workspace/pka-bench/models"
)


def _snapshot(model_dir: str) -> Path:
    root = ONNX_MODELS_ROOT / model_dir / "snapshots"
    return next(root.iterdir())


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    onnx_path: Path
    tokenizer_path: Path
    pooling: str  # "mean" | "cls"
    query_prefix: str
    passage_prefix: str


MODEL_SPECS = [
    ModelSpec(
        key="bge-m3",
        hf_id="BAAI/bge-m3",
        onnx_path=_snapshot("models--BAAI--bge-m3") / "onnx" / "model.onnx",
        tokenizer_path=_snapshot("models--BAAI--bge-m3") / "tokenizer.json",
        pooling="cls",
        query_prefix="",
        passage_prefix="",
    ),
    ModelSpec(
        key="multilingual-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        onnx_path=_snapshot("models--intfloat--multilingual-e5-small") / "onnx" / "model.onnx",
        tokenizer_path=_snapshot("models--intfloat--multilingual-e5-small") / "tokenizer.json",
        pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    ModelSpec(
        key="multilingual-e5-base",
        hf_id="intfloat/multilingual-e5-base",
        onnx_path=_snapshot("models--intfloat--multilingual-e5-base") / "onnx" / "model.onnx",
        tokenizer_path=_snapshot("models--intfloat--multilingual-e5-base") / "tokenizer.json",
        pooling="mean",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
]


class OnnxEmbedder:
    def __init__(self, spec: ModelSpec, threads: int = 4):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(spec.onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(spec.tokenizer_path))
        self.tok.enable_truncation(max_length=512)
        pad_id = self.tok.token_to_id("<pad>")
        pad_tok = "<pad>"
        if pad_id is None:
            pad_id = self.tok.token_to_id("[PAD]")
            pad_tok = "[PAD]"
        self.tok.enable_padding(pad_id=pad_id or 0, pad_token=pad_tok)
        self.spec = spec

    def _forward(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        out = self.sess.run(None, feeds)[0]
        if self.spec.pooling == "cls":
            vec = out[:, 0, :]
        else:
            m = mask[:, :, None].astype(np.float32)
            vec = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        return vec.astype(np.float32)

    def embed(self, texts: list[str], *, is_query: bool, batch_size: int = 8) -> tuple[np.ndarray, float, float]:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        prefixed = [prefix + t for t in texts]
        outs = []
        t0, c0 = time.perf_counter(), time.process_time()
        for i in range(0, len(prefixed), batch_size):
            outs.append(self._forward(prefixed[i : i + batch_size]))
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        return np.vstack(outs), wall, cpu


def load_corpus_chunks() -> tuple[list[str], list[dict]]:
    """Extract + chunk every corpus doc via the app's real pipeline. Returns (texts, meta)."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    texts: list[str] = []
    meta: list[dict] = []
    for entry in manifest:
        doc_id = entry["doc_id"]
        fmt = entry["format"]
        path = CORPUS_DIR / f"{doc_id}.{fmt}"
        content = path.read_bytes()
        loaded = load(filename=path.name, content=content)
        chunks = process_document(
            document_id=doc_id,
            filename=path.name,
            file_type=loaded.file_type,
            raw_text=loaded.raw_text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for c in chunks:
            texts.append(c.text)
            meta.append({"document_id": doc_id, "chunk_id": c.chunk_id})
    return texts, meta


def load_queries() -> list[dict]:
    data = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    return data["queries"]


def evaluate(
    doc_vecs: np.ndarray,
    meta: list[dict],
    query_vecs: np.ndarray,
    queries: list[dict],
    top_k: int = TOP_K,
) -> dict:
    sims = query_vecs @ doc_vecs.T
    results = []
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi])[:top_k]
        top = [{"score": float(sims[qi][j]), **meta[j]} for j in order]
        top_docs = []
        seen = set()
        for t in top:
            if t["document_id"] not in seen:
                seen.add(t["document_id"])
                top_docs.append(t["document_id"])
        expected = set(q["expected_doc_ids"])
        rec = {
            "id": q["id"],
            "language": q["language"],
            "category": q["category"],
            "expected_doc_ids": q["expected_doc_ids"],
            "top1_score": top[0]["score"],
            "top1_doc": top[0]["document_id"],
            "top5_docs": top_docs,
        }
        if expected:
            rec["top1_correct"] = top[0]["document_id"] in expected
            rec["recall_at_5"] = bool(expected & set(top_docs))
        results.append(rec)

    answerable = [r for r in results if r["expected_doc_ids"]]
    out_of_corpus = [r for r in results if not r["expected_doc_ids"]]
    by_category: dict[str, list[dict]] = {}
    for r in answerable:
        by_category.setdefault(r["category"], []).append(r)

    ans_scores = [r["top1_score"] for r in answerable]
    ooc_scores = [r["top1_score"] for r in out_of_corpus]

    return {
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "n_out_of_corpus": len(out_of_corpus),
        "top1_accuracy": sum(r["top1_correct"] for r in answerable) / len(answerable),
        "recall_at_5": sum(r["recall_at_5"] for r in answerable) / len(answerable),
        "accuracy_by_category": {
            cat: sum(r["top1_correct"] for r in rs) / len(rs)
            for cat, rs in sorted(by_category.items())
        },
        "mean_top1_score_answerable": statistics.mean(ans_scores) if ans_scores else None,
        "min_top1_score_answerable": min(ans_scores) if ans_scores else None,
        "mean_top1_score_out_of_corpus": statistics.mean(ooc_scores) if ooc_scores else None,
        "max_top1_score_out_of_corpus": max(ooc_scores) if ooc_scores else None,
        "separability_gap": (
            (min(ans_scores) - max(ooc_scores)) if ans_scores and ooc_scores else None
        ),
        "per_query": results,
    }


def run_model(spec: ModelSpec, texts: list[str], meta: list[dict], queries: list[dict]) -> dict:
    print(f"\n=== {spec.key} ({spec.hf_id}) ===", flush=True)
    embedder = OnnxEmbedder(spec)

    query_texts = [q["query"] for q in queries]

    doc_vecs, doc_wall, doc_cpu = embedder.embed(texts, is_query=False, batch_size=8)
    query_vecs, q_wall, q_cpu = embedder.embed(query_texts, is_query=True, batch_size=8)

    quality = evaluate(doc_vecs, meta, query_vecs, queries)

    perf = {
        "n_chunks": len(texts),
        "ingestion_wall_seconds": doc_wall,
        "ingestion_cpu_seconds": doc_cpu,
        "ms_per_chunk_wall": doc_wall / len(texts) * 1000,
        "chunks_per_sec": len(texts) / doc_wall,
        "n_queries": len(query_texts),
        "query_wall_seconds": q_wall,
        "ms_per_query_wall": q_wall / len(query_texts) * 1000,
    }

    result = {"model_key": spec.key, "hf_id": spec.hf_id, "quality": quality, "performance": perf}

    print(
        f"  top1_accuracy={quality['top1_accuracy']:.3f}  recall@5={quality['recall_at_5']:.3f}  "
        f"separability_gap={quality['separability_gap']:.3f}"
    )
    print(
        f"  ingestion: {perf['ms_per_chunk_wall']:.1f} ms/chunk ({perf['chunks_per_sec']:.2f} chunks/s)  "
        f"query: {perf['ms_per_query_wall']:.1f} ms/query"
    )

    del embedder
    gc.collect()
    return result


def main() -> None:
    print("Loading and chunking eval corpus via app.documents pipeline...")
    texts, meta = load_corpus_chunks()
    queries = load_queries()
    print(f"{len(texts)} chunks from {len({m['document_id'] for m in meta})} documents, "
          f"{len(queries)} queries.")

    all_results = []
    for spec in MODEL_SPECS:
        result = run_model(spec, texts, meta, queries)
        all_results.append(result)
        (RESULTS_DIR / f"{spec.key}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (RESULTS_DIR / "all_models.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
