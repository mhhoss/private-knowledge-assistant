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
    retrieval_min_score: float = Field(default=0.35, ge=0.0, le=1.0)

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
    )


def build_llm(settings: Settings) -> LLM:
    """Construct the chat/completion client. The only place LLM credentials are used."""
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_base_url,
    )
