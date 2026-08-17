"""Query + retrieved context → grounded answer + sources (R-04, R-05, ADR-4).

Unit-level: a stubbed LLM (`tests/conftest.py::StubLLM`) stands in for any real
provider, since generator.py never calls one during tests (R-08's whole point).
"""

from __future__ import annotations

from app.rag.generator import Citation, ContextChunk, GeneratedAnswer, generate
from tests.conftest import StubLLM

ENGLISH_CHUNK = ContextChunk(
    text="Kubernetes cluster spending increased by twelve percent.",
    document_id="en1",
    filename="report.pdf",
    file_type="pdf",
    chunk_id="0000",
)
PERSIAN_CHUNK = ContextChunk(
    text="هزینه خوشه کوبرنتیز دوازده درصد افزایش یافت.",
    document_id="fa1",
    filename="گزارش.pdf",
    file_type="pdf",
    chunk_id="0000",
)
MIXED_CHUNK = ContextChunk(
    text="تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند.",
    document_id="mx1",
    filename="mixed.docx",
    file_type="docx",
    chunk_id="0000",
)


class TestGroundedAnswers:
    def test_english_answer_carries_the_model_output_and_sources(
        self, llm: StubLLM
    ) -> None:
        llm.response = "Kubernetes spending rose 12%."
        result = generate(query="How much did Kubernetes spending rise?", chunks=[ENGLISH_CHUNK], llm=llm)

        assert isinstance(result, GeneratedAnswer)
        assert result.is_refusal is False
        assert result.answer == "Kubernetes spending rose 12%."
        assert result.sources == [
            Citation(
                document_id="en1",
                filename="report.pdf",
                file_type="pdf",
                chunk_id="0000",
                excerpt=ENGLISH_CHUNK.text,
            )
        ]

    def test_persian_answer_carries_the_model_output_and_sources(
        self, llm: StubLLM
    ) -> None:
        llm.response = "هزینه کوبرنتیز دوازده درصد افزایش یافت."
        result = generate(query="هزینه کوبرنتیز چقدر افزایش یافت؟", chunks=[PERSIAN_CHUNK], llm=llm)

        assert result.is_refusal is False
        assert result.answer == "هزینه کوبرنتیز دوازده درصد افزایش یافت."
        assert result.sources[0].filename == "گزارش.pdf"

    def test_mixed_language_query_and_context(self, llm: StubLLM) -> None:
        llm.response = "تیم زیرساخت از Kubernetes و PostgreSQL استفاده می‌کند."
        result = generate(
            query="زیرساخت از چه Kubernetes tools استفاده می‌کند؟",
            chunks=[MIXED_CHUNK],
            llm=llm,
        )

        assert result.is_refusal is False
        assert "Kubernetes" in result.answer
        assert result.sources[0].document_id == "mx1"

    def test_multiple_chunks_all_become_sources(self, llm: StubLLM) -> None:
        llm.response = "Combined answer."
        result = generate(
            query="What infrastructure is used?",
            chunks=[ENGLISH_CHUNK, PERSIAN_CHUNK],
            llm=llm,
        )
        assert {source.document_id for source in result.sources} == {"en1", "fa1"}


class TestDeterministicRefusal:
    def test_empty_context_is_refused_without_calling_the_llm(self, llm: StubLLM) -> None:
        llm.response = "this must never be seen"
        result = generate(query="Anything?", chunks=[], llm=llm)

        assert result.is_refusal is True
        assert result.sources == []
        assert llm.call_count == 0
        assert result.answer != llm.response

    def test_empty_context_refusal_is_english_for_an_english_query(
        self, llm: StubLLM
    ) -> None:
        result = generate(query="What does the report say?", chunks=[], llm=llm)
        assert "enough information" in result.answer

    def test_empty_context_refusal_is_persian_for_a_persian_query(
        self, llm: StubLLM
    ) -> None:
        result = generate(query="گزارش چه می‌گوید؟", chunks=[], llm=llm)
        assert "اطلاعات کافی" in result.answer

    def test_empty_context_refusal_is_deterministic(self, llm: StubLLM) -> None:
        first = generate(query="Anything?", chunks=[], llm=llm)
        second = generate(query="Anything?", chunks=[], llm=llm)
        assert first == second


class TestModelSignaledRefusal:
    """ADR-4's second layer: context passed the cutoff but the model still refuses."""

    def test_model_refusal_token_produces_no_sources(self, llm: StubLLM) -> None:
        llm.response = "[[INSUFFICIENT_CONTEXT]]"
        result = generate(query="Unrelated question?", chunks=[ENGLISH_CHUNK], llm=llm)

        assert result.is_refusal is True
        assert result.sources == []
        assert llm.call_count == 1  # unlike the deterministic path, this DOES call it

    def test_model_refusal_replaces_token_with_canned_message(
        self, llm: StubLLM
    ) -> None:
        llm.response = "[[INSUFFICIENT_CONTEXT]]"
        result = generate(query="Unrelated question?", chunks=[ENGLISH_CHUNK], llm=llm)
        assert "[[INSUFFICIENT_CONTEXT]]" not in result.answer

    def test_model_refusal_in_persian_query_uses_persian_canned_message(
        self, llm: StubLLM
    ) -> None:
        llm.response = "[[INSUFFICIENT_CONTEXT]]"
        result = generate(query="این موضوع نامرتبط چیست؟", chunks=[PERSIAN_CHUNK], llm=llm)
        assert "اطلاعات کافی" in result.answer

    def test_refusal_token_anywhere_in_response_still_counts(self, llm: StubLLM) -> None:
        """A model that wraps the token in extra text is still treated as a refusal —
        never as a partial, unsupported answer (groundedness rule 5)."""
        llm.response = "Well, [[INSUFFICIENT_CONTEXT]] unfortunately."
        result = generate(query="What?", chunks=[ENGLISH_CHUNK], llm=llm)
        assert result.is_refusal is True
        assert result.sources == []


class TestPromptGrounding:
    def test_prompt_includes_context_text_and_query(self, llm: StubLLM) -> None:
        llm.response = "answer"
        generate(query="What increased?", chunks=[ENGLISH_CHUNK], llm=llm)

        (messages,) = llm.received_messages
        user_message = next(m for m in messages if m.role == "user")
        assert ENGLISH_CHUNK.text in user_message.content
        assert "What increased?" in user_message.content

    def test_system_prompt_forbids_outside_knowledge(self, llm: StubLLM) -> None:
        llm.response = "answer"
        generate(query="q", chunks=[ENGLISH_CHUNK], llm=llm)

        (messages,) = llm.received_messages
        system_message = next(m for m in messages if m.role == "system")
        assert "ONLY" in system_message.content
        assert "outside" in system_message.content.lower()

    def test_system_prompt_instructs_language_matching(self, llm: StubLLM) -> None:
        llm.response = "answer"
        generate(query="q", chunks=[ENGLISH_CHUNK], llm=llm)

        (messages,) = llm.received_messages
        system_message = next(m for m in messages if m.role == "system")
        assert "same language" in system_message.content

    def test_persian_context_is_passed_through_untranslated(self, llm: StubLLM) -> None:
        llm.response = "answer"
        generate(query="q", chunks=[PERSIAN_CHUNK], llm=llm)

        (messages,) = llm.received_messages
        user_message = next(m for m in messages if m.role == "user")
        assert PERSIAN_CHUNK.text in user_message.content


class TestNoDirectAccessToOtherLayers:
    def test_generator_module_does_not_import_storage_or_retrieval(self) -> None:
        import ast
        from pathlib import Path

        source = Path("app/rag/generator.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        } | {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        forbidden = {
            "app.storage.vector_store",
            "app.rag.retriever",
            "chromadb",
        }
        assert not (modules & forbidden), modules & forbidden
        assert "chromadb" not in imported

    def test_generate_is_a_pure_function_of_its_arguments(self, llm: StubLLM) -> None:
        """No hidden state: same inputs, same output, called twice."""
        llm.response = "deterministic"
        first = generate(query="q", chunks=[ENGLISH_CHUNK], llm=llm)
        second = generate(query="q", chunks=[ENGLISH_CHUNK], llm=llm)
        assert first == second
