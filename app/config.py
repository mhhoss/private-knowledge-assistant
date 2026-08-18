"""Typed settings and provider client construction.

The single source of credentials, model names, and provider URLs (invariant 5).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # heavy imports stay out of module import time
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM


class Settings(BaseSettings):
    """Application configuration, resolved from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chat/completion provider (OpenAI-compatible).
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # Embedding provider. Left unset, it reuses the LLM credentials and endpoint;
    # see ADR-5 for why these are separable at all.
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    # Defaults tuned for a slow (e.g. local CPU) embedding backend: measured
    # ~2.5-2.8s/chunk against a real CPU-served BAAI/bge-m3, at which the library
    # defaults (embed_batch_size=100, timeout=60s) reproducibly fail outright on any
    # document with more than ~20-24 chunks (one over-long request per batch exceeds
    # the timeout). embedding_batch_size=10 keeps each request comfortably short at
    # that measured rate; embedding_timeout_seconds=120 adds margin on top for slower
    # backends or larger chunk_size. A fast/hosted embedding provider can raise
    # embedding_batch_size for higher throughput without hitting this failure mode.
    embedding_batch_size: int = Field(default=10, ge=1)
    embedding_timeout_seconds: float = Field(default=120.0, gt=0.0)

    chroma_path: Path = Path("./chroma_db")
    # Chroma's own naming rule, enforced here so a bad value fails at startup.
    chroma_collection: str = Field(
        default="knowledge_base",
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$",
    )
    data_dir: Path = Path("./data")

    # Character counts, not tokens (ADR-9).
    chunk_size: int = Field(default=1024, ge=128)
    chunk_overlap: int = Field(default=128, ge=0)

    retrieval_top_k: int = Field(default=5, ge=1)
    # Measured against BAAI/bge-m3; see ARCHITECTURE.md open question 2.
    retrieval_min_score: float = Field(default=0.60, ge=0.0, le=1.0)

    api_base_url: str = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def _resolve(self) -> Settings:
        if self.embedding_api_key is None:
            self.embedding_api_key = self.llm_api_key
        if self.embedding_base_url is None:
            self.embedding_base_url = self.llm_base_url
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def embedding_fingerprint(self) -> str:
        """Identity of the embedding configuration, per ADR-8.

        The base URL is intentionally excluded: the same model behind a different
        gateway is still the same model.
        """
        return self.embedding_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_embedding_model(settings: Settings) -> BaseEmbedding:
    """Construct the embedding client. The only place embedding credentials are used.

    `model_name`, not `model`: `OpenAIEmbedding`'s `model` argument is validated against
    a fixed enum of legacy OpenAI model names and rejects anything else (including
    current OpenAI models the pinned llama-index version predates, and any non-OpenAI
    model served through an OpenAI-compatible gateway). `model_name` bypasses that
    lookup and is sent to the API as-is, which is what R-08's "any OpenAI-compatible
    provider, no code change" actually requires.
    """
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(
        model_name=settings.embedding_model,
        api_key=settings.embedding_api_key,
        api_base=settings.embedding_base_url,
        embed_batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout_seconds,
    )


def build_llm(settings: Settings) -> LLM:
    """Construct the chat/completion client. The only place LLM credentials are used.

    Works around the same kind of catalog restriction `build_embedding_model` already
    documents, but on the chat side: `OpenAI.metadata` (read on every `chat()` call, via
    `to_payload()`) computes `context_window` by looking `model` up in a fixed table of
    official OpenAI model names and raises `ValueError` for anything else. Unlike the
    embedding client, there is no `model_name=`-style escape hatch here, and the lookup
    happens lazily at call time, not construction time — so it does not surface until a
    real chat request is made with a non-catalog model name. Verified against a real
    OpenRouter call: `openai/gpt-4o-mini` (and every other gateway-prefixed or
    non-OpenAI model id) raises there. Without this override, R-08 ("any
    OpenAI-compatible provider, no code change") would be false for every provider
    except literal, unprefixed OpenAI model names.
    """
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai.utils import openai_modelname_to_contextsize

    class _GatewayCompatibleOpenAI(OpenAI):
        @property
        def metadata(self) -> LLMMetadata:
            try:
                context_window = openai_modelname_to_contextsize(self._get_model_name())
            except ValueError:
                # Not one of OpenAI's own catalog names (any OpenRouter/gateway
                # model id lands here). 128k is a generous, non-authoritative
                # estimate; an actual mismatch surfaces as a provider error on the
                # request itself, not a silent truncation.
                context_window = 128_000
            return LLMMetadata(
                context_window=context_window,
                num_output=self.max_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

    return _GatewayCompatibleOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_base_url,
    )
