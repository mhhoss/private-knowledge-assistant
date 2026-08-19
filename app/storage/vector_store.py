"""Chroma persistence. The only module that knows the vector store is Chroma.

Owns collection lifecycle, document-level queries and deletion, and the embedding
fingerprint check described in ADR-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaClientSettings
from llama_index.vector_stores.chroma import ChromaVectorStore

# Cosine distance keeps similarity scores comparable to a fixed threshold; Chroma's
# default (L2) would make RETRIEVAL_MIN_SCORE meaningless.
_SPACE_KEY = "hnsw:space"
_SPACE = "cosine"
_FINGERPRINT_KEY = "embedding_model"


class EmbeddingMismatchError(RuntimeError):
    """The index was built with a different embedding configuration (R-11)."""


@dataclass(frozen=True)
class DocumentSummary:
    """One indexed document, as derived from chunk metadata (ADR-2)."""

    document_id: str
    filename: str
    file_type: str
    chunk_count: int


class VectorStore:
    """Persistent Chroma collection scoped to one embedding configuration."""

    def __init__(
        self,
        *,
        path: Path | str,
        collection_name: str,
        embedding_fingerprint: str,
    ) -> None:
        self._path = Path(path)
        self._name = collection_name
        self._fingerprint = embedding_fingerprint
        self._client = chromadb.PersistentClient(
            path=str(self._path),
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()
        self._llama_store = ChromaVectorStore(chroma_collection=self._collection)

    # --- lifecycle ---

    def _create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._name,
            metadata={_SPACE_KEY: _SPACE, _FINGERPRINT_KEY: self._fingerprint},
        )

    def _open_collection(self) -> Collection:
        """Open the collection, enforcing embedding compatibility (invariant 9)."""
        collection = self._create_collection()
        stored = (collection.metadata or {}).get(_FINGERPRINT_KEY)
        if stored == self._fingerprint:
            return collection

        if collection.count() == 0:
            # Nothing incompatible exists yet, so adopt the current configuration.
            # Recreated rather than modified: Chroma rejects metadata updates that
            # carry the distance function, even unchanged.
            self._client.delete_collection(name=self._name)
            return self._create_collection()

        raise EmbeddingMismatchError(
            f"Collection '{self._name}' was built with embedding model "
            f"{stored or 'unknown'!r} but the configured model is "
            f"{self._fingerprint!r}. Vectors from different models are not comparable. "
            "Reset the knowledge base and re-index, or restore the previous "
            "EMBEDDING_MODEL."
        )

    def reset(self) -> None:
        """Drop all indexed data and recreate an empty collection (R-07)."""
        self._client.delete_collection(name=self._name)
        self._collection = self._create_collection()
        self._llama_store = ChromaVectorStore(chroma_collection=self._collection)

    @property
    def embedding_fingerprint(self) -> str:
        """The embedding model this collection is currently keyed to (ADR-8)."""
        return self._fingerprint

    def adopt_embedding_fingerprint(self, fingerprint: str) -> None:
        """Repoint this collection at a new embedding configuration, if safe.

        Mirrors `_open_collection`'s own empty-adopt path: a no-op if `fingerprint`
        already matches, and only otherwise possible while the collection holds no
        chunks. Raises `EmbeddingMismatchError` — never silently — when chunks already
        exist under a different embedding model (ADR-8, R-11); recovery is still a
        knowledge-base reset, not this call, so it never resets or deletes data itself.
        """
        if fingerprint == self._fingerprint:
            return
        if self.count() != 0:
            raise EmbeddingMismatchError(
                f"Collection '{self._name}' has {self.count()} indexed chunk(s) built "
                f"with embedding model {self._fingerprint!r}; cannot switch to "
                f"{fingerprint!r} without resetting the knowledge base first."
            )
        self._fingerprint = fingerprint
        self._client.delete_collection(name=self._name)
        self._collection = self._create_collection()
        self._llama_store = ChromaVectorStore(chroma_collection=self._collection)

    @property
    def llama_vector_store(self) -> ChromaVectorStore:
        """LlamaIndex adapter for this collection, for indexing and retrieval."""
        return self._llama_store

    # --- documents ---

    def count(self) -> int:
        """Total number of indexed chunks."""
        return self._collection.count()

    def document_exists(self, document_id: str) -> bool:
        result = self._collection.get(
            where={"document_id": document_id}, limit=1, include=[]
        )
        return bool(result["ids"])

    def list_documents(self) -> list[DocumentSummary]:
        result = self._collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, object]] = {}
        for metadata in result["metadatas"] or []:
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "filename": metadata.get("filename", ""),
                    "file_type": metadata.get("file_type", ""),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] = int(entry["chunk_count"]) + 1  # type: ignore[call-overload]

        return sorted(
            (
                DocumentSummary(
                    document_id=document_id,
                    filename=str(entry["filename"]),
                    file_type=str(entry["file_type"]),
                    chunk_count=int(entry["chunk_count"]),  # type: ignore[call-overload]
                )
                for document_id, entry in grouped.items()
            ),
            key=lambda doc: (doc.filename, doc.document_id),
        )

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk of a document (invariant 6). Safe to call when absent."""
        self._collection.delete(where={"document_id": document_id})
