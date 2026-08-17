"""Query + retrieved context → grounded answer + sources. No retrieval, no storage.

Receives context as an argument and never fetches it itself (invariant 2). This module
does not import Chroma, the vector store, or `rag/retriever.py` — `ContextChunk` is its
own minimal type, not `retriever.RetrievedChunk`, so generation stays decoupled from how
context was obtained. The caller (eventually `rag/engine.py`) adapts one to the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

# An arbitrary, ASCII-only marker the model is instructed to emit verbatim when the
# supplied context does not support an answer — never shown to the user. Its own wording
# is replaced with a deterministic, language-matched refusal (see `_refusal_message`),
# the same way the no-context case is handled, so a refusal always reads the same
# regardless of which of the two paths produced it.
_REFUSAL_TOKEN = "[[INSUFFICIENT_CONTEXT]]"

_REFUSAL_EN = "I don't have enough information in the indexed documents to answer that."
_REFUSAL_FA = "اطلاعات کافی در اسناد نمایه‌شده برای پاسخ به این پرسش وجود ندارد."

# A rough Arabic-script check, not language detection (invariant 10): the only question
# is which of two fixed refusal strings to show, per ADR-9.
_ARABIC_SCRIPT = range(0x0600, 0x06FF + 1)

_SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the \
context excerpts supplied below, taken from the user's own documents.

Rules:
- Use only information explicitly stated in the supplied context. Never use outside \
knowledge, and never assume, infer, or guess anything the context does not state.
- If the context does not contain enough information to answer the question, respond \
with exactly this text and nothing else: {_REFUSAL_TOKEN}
- Otherwise, answer the question directly and concisely, using only the given context.
- Respond in the same language as the question, whether English, Persian, or a mix of \
both — regardless of which language the context is written in.
"""


@dataclass(frozen=True)
class ContextChunk:
    """One piece of context to generate from — exactly what a citation needs and no
    more. Deliberately not `retriever.RetrievedChunk`: this module does not know that
    type exists, and has no use for a retrieval score.
    """

    text: str
    document_id: str
    filename: str
    file_type: str
    chunk_id: str


@dataclass(frozen=True)
class Citation:
    """One supporting source for an answer (R-05)."""

    document_id: str
    filename: str
    file_type: str
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """The result of one generation call: an answer, or a refusal with no sources."""

    answer: str
    sources: list[Citation]
    is_refusal: bool


def generate(*, query: str, chunks: list[ContextChunk], llm: LLM) -> GeneratedAnswer:
    """Generate a grounded answer from `chunks`, or a refusal if they cannot support one.

    `chunks` should already be retrieval-filtered (`rag/retriever.retrieve`); this
    function trusts them as-is and never re-scores or fetches more. An empty `chunks`
    is refused without calling `llm` — the deterministic half of ADR-4.
    `rag/engine.py` relies on this and does not special-case an empty result itself:
    calling the LLM with no context is never correct, so the guard lives here, at the
    one call site, rather than being duplicated by every caller.
    """
    if not chunks:
        return GeneratedAnswer(
            answer=_refusal_message(query), sources=[], is_refusal=True
        )

    from llama_index.core.llms import ChatMessage, MessageRole

    response = llm.chat(
        [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER, content=_build_user_prompt(query, chunks)
            ),
        ]
    )
    answer = (response.message.content or "").strip()

    if _REFUSAL_TOKEN in answer:
        return GeneratedAnswer(
            answer=_refusal_message(query), sources=[], is_refusal=True
        )

    return GeneratedAnswer(
        answer=answer,
        sources=[_citation(chunk) for chunk in chunks],
        is_refusal=False,
    )


def _build_user_prompt(query: str, chunks: list[ContextChunk]) -> str:
    excerpts = "\n\n".join(
        f'[{i}] (from "{chunk.filename}")\n{chunk.text}'
        for i, chunk in enumerate(chunks, start=1)
    )
    return f"Context:\n{excerpts}\n\nQuestion: {query}"


def _citation(chunk: ContextChunk) -> Citation:
    return Citation(
        document_id=chunk.document_id,
        filename=chunk.filename,
        file_type=chunk.file_type,
        chunk_id=chunk.chunk_id,
        excerpt=chunk.text,
    )


def _refusal_message(query: str) -> str:
    is_persian = any(ord(char) in _ARABIC_SCRIPT for char in query)
    return _REFUSAL_FA if is_persian else _REFUSAL_EN
