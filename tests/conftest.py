"""Shared test fixtures.

No test may reach a real provider or the project's `chroma_db/`.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms import CustomLLM, LLMMetadata
from pydantic import Field

from app.storage.vector_store import VectorStore

STUB_FINGERPRINT = "stub-embedding-v1"
_DIM = 32

_SETTINGS_ENV_PREFIXES = (
    "LLM_",
    "EMBEDDING_",
    "CHROMA_",
    "CHUNK_",
    "RETRIEVAL_",
    "API_",
)


class StubEmbedding(BaseEmbedding):
    """Deterministic offline embedding.

    Script-agnostic by construction: it hashes whitespace-separated tokens, so Persian
    and English text are treated identically and equal text always yields equal vectors.
    """

    @staticmethod
    def _vector(text: str) -> list[float]:
        buckets = [0.0] * _DIM
        for token in text.split():
            buckets[zlib.crc32(token.encode("utf-8")) % _DIM] += 1.0
        magnitude = sum(value * value for value in buckets) ** 0.5
        if magnitude == 0.0:
            buckets[0] = 1.0
            return buckets
        return [value / magnitude for value in buckets]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


@pytest.fixture
def embed_model() -> StubEmbedding:
    return StubEmbedding()


class StubLLM(CustomLLM):
    """Deterministic offline chat model: returns a scripted response, records calls.

    Used to assert both the content of a generated answer and — for the deterministic
    refusal path — that no call was made at all.
    """

    context_window: int = 4096
    num_output: int = 512
    response: str = ""
    call_count: int = 0
    received_messages: list[list[ChatMessage]] = Field(default_factory=list)
    error: Exception | None = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        self.call_count += 1
        self.received_messages.append(list(messages))
        if self.error is not None:
            raise self.error
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.response)
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError("generator.py uses chat(), not complete()")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def chroma_path(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def store(chroma_path: Path) -> VectorStore:
    return VectorStore(
        path=chroma_path,
        collection_name="test_kb",
        embedding_fingerprint=STUB_FINGERPRINT,
    )


@pytest.fixture
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove ambient configuration so default resolution can be asserted."""
    for name in list(os.environ):
        if name.startswith(_SETTINGS_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield


def settings_kwargs(**overrides: Any) -> dict[str, Any]:
    """Settings kwargs that ignore any local `.env`."""
    return {"_env_file": None, **overrides}
