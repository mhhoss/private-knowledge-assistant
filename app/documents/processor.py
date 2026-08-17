"""Normalization, chunking, and metadata propagation.

The only language-aware code in the project (ADR-9). Everything here is deterministic
and script-agnostic: the same functions serve English, Persian, and mixed text, and the
same normalization must be applied to queries (invariant 7).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Codepoints that render identically but block matching: text typed on an Arabic
# keyboard, or extracted from a PDF, uses different characters than Persian input.
# Written as escapes because the pairs are indistinguishable on screen.
_CHAR_MAP = str.maketrans(
    {
        "\u064a": "\u06cc",  # Arabic yeh     -> Farsi yeh
        "\u0649": "\u06cc",  # alef maksura   -> Farsi yeh
        "\u0643": "\u06a9",  # Arabic kaf     -> Keheh
        **{chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)},  # Arabic-Indic digits
    }
)

# Tashkeel (harakat) and superscript alef: pronunciation aids, absent from most text,
# so keeping them would split otherwise identical words.
_MARKS = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel is pure justification padding.
_TATWEEL = "\u0640"

# Bidi controls and zero-width characters. U+200C (ZWNJ) is deliberately absent: it is
# a semantic letter separator in Persian, not formatting.
_INVISIBLE = re.compile(r"[\u200B\u200D-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Sentence terminators across both languages, including Persian question mark and
# semicolon. The Persian comma is excluded: it is not a sentence boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?\u061F\u061B\u2026])\s+")

_PARAGRAPH_SEP = "\n\n"


@dataclass(frozen=True)
class Chunk:
    """An indexable unit of text that knows where it came from."""

    document_id: str
    filename: str
    file_type: str
    chunk_id: str
    text: str

    @property
    def node_id(self) -> str:
        """Globally unique id; `chunk_id` is only unique within its document."""
        return f"{self.document_id}-{self.chunk_id}"

    def metadata(self) -> dict[str, str]:
        """Source identity carried with every chunk (invariant 1)."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_id": self.chunk_id,
        }


def normalize_text(text: str) -> str:
    """Normalize document or query text. Both must use this function (invariant 7).

    NFKC folds the Arabic presentation forms that Persian PDF extraction commonly
    produces (and English typographic ligatures) back to base characters.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP)
    text = _MARKS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _INVISIBLE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub(_PARAGRAPH_SEP, text)
    return text.strip()


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split normalized text into overlapping chunks of at most `chunk_size` characters.

    Sizing is by character count so that chunk length does not depend on how densely a
    script tokenizes (ADR-9). Boundaries prefer paragraphs, then sentences, then
    whitespace, and only split mid-word when a single word exceeds the limit.
    """
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for segment in _segments(text, chunk_size):
        if not current:
            current = segment
            continue
        if len(current) + 1 + len(segment) <= chunk_size:
            current = f"{current} {segment}"
            continue
        chunks.append(current)
        current = _carry_over(
            current, segment, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    if current:
        chunks.append(current)
    return chunks


def process_document(
    *,
    document_id: str,
    filename: str,
    file_type: str,
    raw_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Turn one document's extracted text into source-attributed chunks.

    Returns an empty list when the document holds no extractable text; the caller
    reports that as a failed file rather than indexing nothing (R-09).
    """
    normalized = normalize_text(raw_text)
    return [
        Chunk(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            chunk_id=f"{position:04d}",
            text=piece,
        )
        for position, piece in enumerate(
            chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    ]


def _segments(text: str, limit: int) -> list[str]:
    """Break text into units no longer than `limit`, splitting as coarsely as possible."""
    units: list[str] = []
    for paragraph in text.split(_PARAGRAPH_SEP):
        paragraph = paragraph.replace("\n", " ").strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= limit:
                units.append(sentence)
            else:
                units.extend(_words(sentence, limit))
    return units


def _words(sentence: str, limit: int) -> list[str]:
    """Individual words. Units stay small so that `chunk_text` controls all packing.

    A word longer than a whole chunk is chopped; nothing else splits mid-word.
    """
    pieces: list[str] = []
    for word in sentence.split(" "):
        while len(word) > limit:
            pieces.append(word[:limit])
            word = word[limit:]
        if word:
            pieces.append(word)
    return pieces


def _carry_over(
    previous: str, segment: str, *, chunk_size: int, chunk_overlap: int
) -> str:
    """Start the next chunk with the tail of the previous one, on a word boundary.

    The overlap is dropped rather than allowed to push the chunk over `chunk_size`.
    """
    if chunk_overlap <= 0:
        return segment
    tail = previous[-chunk_overlap:]
    if len(tail) < len(previous):
        # Trim a partial leading word so the overlap never begins mid-word. ZWNJ is not
        # whitespace, so Persian compound words stay intact.
        _, separator, remainder = tail.partition(" ")
        if separator:
            tail = remainder
    tail = tail.strip()
    if not tail or len(tail) + 1 + len(segment) > chunk_size:
        return segment
    return f"{tail} {segment}"
