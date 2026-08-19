"""Configuration resolution: provider fallbacks and rejected values."""

from __future__ import annotations

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

from app.config import Settings, build_embedding_model, build_llm, require_credentials
from tests.conftest import settings_kwargs

pytestmark = pytest.mark.usefixtures("clean_settings_env")


class TestEmbeddingProviderFallback:
    def test_embedding_defaults_to_llm_credentials(self) -> None:
        settings = Settings(
            **settings_kwargs(llm_api_key="key-1", llm_base_url="https://gateway/v1")
        )
        assert settings.embedding_api_key == "key-1"
        assert settings.embedding_base_url == "https://gateway/v1"

    def test_embedding_provider_can_be_separate(self) -> None:
        """ADR-5: gateways that serve chat but not embeddings."""
        settings = Settings(
            **settings_kwargs(
                llm_api_key="router-key",
                llm_base_url="https://openrouter.ai/api/v1",
                embedding_api_key="openai-key",
                embedding_base_url="https://api.openai.com/v1",
            )
        )
        assert settings.embedding_api_key == "openai-key"
        assert settings.embedding_base_url == "https://api.openai.com/v1"
        assert settings.llm_api_key == "router-key"


class TestEmbeddingFingerprint:
    def test_fingerprint_is_the_model_name(self) -> None:
        settings = Settings(**settings_kwargs(embedding_model="multilingual-e5"))
        assert settings.embedding_fingerprint == "multilingual-e5"

    def test_fingerprint_ignores_base_url(self) -> None:
        """ADR-8: the same model behind a different gateway is the same model."""
        common = {"embedding_model": "text-embedding-3-small"}
        direct = Settings(
            **settings_kwargs(embedding_base_url="https://api.openai.com/v1", **common)
        )
        proxied = Settings(
            **settings_kwargs(embedding_base_url="https://proxy.internal/v1", **common)
        )
        assert direct.embedding_fingerprint == proxied.embedding_fingerprint


class TestRequireCredentials:
    def test_passes_when_both_credentials_are_set(self) -> None:
        settings = Settings(**settings_kwargs(llm_api_key="key-1"))
        require_credentials(settings)  # must not raise

    def test_raises_when_llm_api_key_is_blank(self) -> None:
        settings = Settings(**settings_kwargs(llm_api_key=""))
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            require_credentials(settings)

    def test_raises_when_embedding_api_key_is_blank_despite_llm_key(self) -> None:
        """Only reachable if a future change stops resolving embedding_api_key from
        llm_api_key for a blank value; guards that fallback assumption explicitly."""
        settings = Settings(**settings_kwargs(llm_api_key="key-1"))
        settings.embedding_api_key = ""
        with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
            require_credentials(settings)


class TestValidation:
    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            Settings(**settings_kwargs(chunk_size=512, chunk_overlap=512))

    def test_collection_name_must_satisfy_chroma_rules(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**settings_kwargs(chroma_collection="kb"))

    @pytest.mark.parametrize("score", [-0.1, 1.5])
    def test_min_score_stays_within_similarity_range(self, score: float) -> None:
        with pytest.raises(ValidationError):
            Settings(**settings_kwargs(retrieval_min_score=score))

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**settings_kwargs(retrieval_top_k=0))

    def test_embedding_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**settings_kwargs(embedding_batch_size=0))

    def test_embedding_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Settings(**settings_kwargs(embedding_timeout_seconds=0.0))


class TestBuildEmbeddingModel:
    """Regression coverage for `build_embedding_model`. No network calls: constructing
    an OpenAI-compatible client never contacts the provider — only sending a request
    would.
    """

    def test_accepts_a_catalog_openai_model_name(self) -> None:
        settings = Settings(
            **settings_kwargs(
                embedding_model="text-embedding-3-small",
                embedding_api_key="dummy-key",
                embedding_base_url="http://localhost:9999/v1",
            )
        )
        embed_model = build_embedding_model(settings)

        assert isinstance(embed_model, BaseEmbedding)
        assert embed_model.model_name == "text-embedding-3-small"

    def test_accepts_an_arbitrary_non_catalog_model_name(self) -> None:
        """`OpenAIEmbedding`'s `model=` argument is validated against a fixed enum of
        legacy OpenAI names and previously rejected anything else — including a local
        multilingual model like `bge-m3` served through an OpenAI-compatible gateway
        (e.g. Ollama, TEI). `build_embedding_model` must use `model_name=` instead,
        which bypasses that lookup, or R-08 ("switch providers via env vars, no code
        change") is false for any model name outside that enum.
        """
        settings = Settings(
            **settings_kwargs(
                embedding_model="bge-m3",
                embedding_api_key="dummy-key",
                embedding_base_url="http://localhost:11434/v1",
            )
        )
        embed_model = build_embedding_model(settings)

        assert isinstance(embed_model, BaseEmbedding)
        assert embed_model.model_name == "bge-m3"

    def test_another_non_catalog_provider_model_name_also_works(self) -> None:
        """Pins the fix as general, not a `bge-m3` special case."""
        settings = Settings(
            **settings_kwargs(
                embedding_model="voyage-3",
                embedding_api_key="dummy-key",
                embedding_base_url="http://localhost:9999/v1",
            )
        )
        embed_model = build_embedding_model(settings)

        assert embed_model.model_name == "voyage-3"

    def test_falls_back_to_llm_credentials_when_embedding_settings_are_unset(
        self,
    ) -> None:
        """ADR-5's fallback still resolves before the client is built."""
        settings = Settings(
            **settings_kwargs(
                llm_api_key="llm-key",
                llm_base_url="http://localhost:9999/v1",
                embedding_model="bge-m3",
            )
        )
        embed_model = build_embedding_model(settings)

        assert isinstance(embed_model, OpenAIEmbedding)
        assert embed_model.api_key == "llm-key"
        assert embed_model.api_base == "http://localhost:9999/v1"


class TestEmbeddingBatchSizeAndTimeout:
    """Regression coverage for the ADR-13 fix: a real CPU-served `BAAI/bge-m3` embeds
    realistic chunks at ~2.5-2.8s/chunk (measured in the scale evaluation), at which
    llama-index's library defaults (`embed_batch_size=100`, `timeout=60.0`) reproducibly
    fail ingestion outright for any document over ~20-24 chunks — one over-long batch
    request exceeds the timeout. These tests pin that both settings actually reach the
    embedding client, and that the *defaults* leave a safety margin at that measured
    rate, without making any live provider calls.
    """

    # Measured steady-state cost for realistic (~750-char) chunks against a real
    # CPU-served BAAI/bge-m3 (see docs/ARCHITECTURE.md's Performance section).
    _MEASURED_WORST_CASE_SECONDS_PER_CHUNK = 2.8

    def test_configured_batch_size_and_timeout_reach_the_embedding_client(self) -> None:
        settings = Settings(
            **settings_kwargs(
                embedding_model="bge-m3",
                embedding_api_key="dummy-key",
                embedding_base_url="http://localhost:11434/v1",
                embedding_batch_size=7,
                embedding_timeout_seconds=45.0,
            )
        )
        embed_model = build_embedding_model(settings)

        assert isinstance(embed_model, OpenAIEmbedding)
        assert embed_model.embed_batch_size == 7
        assert embed_model.timeout == 45.0

    def test_defaults_are_safe_for_long_documents_at_the_measured_cpu_throughput(
        self,
    ) -> None:
        """The failure this fixes was a single over-long batch request exceeding the
        client's timeout, not raw throughput (raising the timeout or lowering the
        batch size are the only two levers). Pin that the *default* combination
        leaves real margin at the measured worst-case rate, so a regression that
        quietly raises the default batch size or lowers the default timeout is caught
        here rather than by re-running the multi-minute live benchmark.
        """
        settings = Settings(**settings_kwargs())

        worst_case_request_seconds = (
            settings.embedding_batch_size * self._MEASURED_WORST_CASE_SECONDS_PER_CHUNK
        )

        assert worst_case_request_seconds < settings.embedding_timeout_seconds
        # At least 2x margin, not just barely under the wire.
        assert worst_case_request_seconds * 2 <= settings.embedding_timeout_seconds

    def test_default_batch_size_and_timeout_values(self) -> None:
        settings = Settings(**settings_kwargs())

        assert settings.embedding_batch_size == 10
        assert settings.embedding_timeout_seconds == 120.0


class TestBuildLlm:
    def test_constructs_a_valid_llm_client_for_the_configured_model(self) -> None:
        settings = Settings(
            **settings_kwargs(
                llm_model="gpt-4o-mini",
                llm_api_key="dummy-key",
                llm_base_url="http://localhost:9999/v1",
            )
        )
        llm = build_llm(settings)

        assert isinstance(llm, OpenAI)
        assert llm.model == "gpt-4o-mini"

    def test_accepts_an_arbitrary_non_catalog_model_name(self) -> None:
        """Unlike the embedding client, `OpenAI.metadata` (read on every `chat()` call)
        validates `model` against a fixed catalog of official OpenAI names and raises
        `ValueError` for anything else — verified against a real OpenRouter call with
        `openai/gpt-4o-mini`. Construction alone doesn't trigger the lookup, so this
        must exercise `.metadata`, not just build the client, to actually catch a
        regression here."""
        settings = Settings(
            **settings_kwargs(
                llm_model="qwen/qwen-2.5-72b-instruct",
                llm_api_key="dummy-key",
                llm_base_url="https://openrouter.ai/api/v1",
            )
        )
        llm = build_llm(settings)

        assert isinstance(llm, OpenAI)
        assert llm.metadata.context_window > 0
        assert llm.model == "qwen/qwen-2.5-72b-instruct"


class TestEnvironmentBinding:
    def test_reads_documented_variable_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "gpt-from-env")
        monkeypatch.setenv("EMBEDDING_MODEL", "embed-from-env")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "9")
        # See the comment on the equivalent call in tests/integration/test_main.py.
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.llm_model == "gpt-from-env"
        assert settings.embedding_model == "embed-from-env"
        assert settings.retrieval_top_k == 9
