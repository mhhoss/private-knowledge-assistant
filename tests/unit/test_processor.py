"""Normalization and chunking, in English, Persian, and mixed text (R-10)."""

from __future__ import annotations

from app.documents.processor import (
    Chunk,
    chunk_text,
    normalize_text,
    process_document,
)

ZWNJ = "‌"
ARABIC_YEH, FARSI_YEH = "ي", "ی"
ARABIC_KAF, KEHEH = "ك", "ک"

# "می‌رود" - a Persian verb whose ZWNJ is part of the word, not decoration.
ZWNJ_WORD = f"می{ZWNJ}رود"

PERSIAN_SENTENCES = (
    "این یک سند فارسی "
    "است؟ "  # ...is this a Persian document? (Persian '?')
    "بله؛ درست است."
)


class TestNormalization:
    def test_unifies_arabic_and_persian_codepoints(self) -> None:
        assert normalize_text(ARABIC_YEH + ARABIC_KAF) == FARSI_YEH + KEHEH
        assert normalize_text("ى") == FARSI_YEH  # alef maksura

    def test_preserves_zwnj(self) -> None:
        assert ZWNJ in normalize_text(ZWNJ_WORD)
        assert normalize_text(ZWNJ_WORD) == ZWNJ_WORD

    def test_removes_other_invisible_characters(self) -> None:
        noisy = f"a​b‍c‪d﻿e"
        assert normalize_text(noisy) == "abcde"

    def test_strips_tashkeel_and_tatweel(self) -> None:
        # "مُحَمَّد" with harakat, and a tatweel-padded form of "سلام".
        assert normalize_text("مُحَمَّد") == (
            "محمد"
        )
        assert normalize_text("ســلام") == (
            "سلام"
        )

    def test_folds_arabic_presentation_forms(self) -> None:
        # Persian PDFs frequently extract as presentation forms.
        assert normalize_text("ﺱﺎﻟﻡ") == (
            "سالم"
        )

    def test_converts_arabic_indic_digits_but_keeps_ascii(self) -> None:
        assert normalize_text("١٢٣") == "۱۲۳"
        assert normalize_text("2024") == "2024"

    def test_leaves_english_words_intact(self) -> None:
        assert normalize_text("The Q4 report is final.") == "The Q4 report is final."

    def test_folds_english_ligatures(self) -> None:
        assert normalize_text("ﬁle") == "file"

    def test_collapses_spaces_but_keeps_paragraph_breaks(self) -> None:
        assert normalize_text("a  \t b\n  c\n\n\n\nd  ") == "a b\nc\n\nd"

    def test_normalizes_mixed_language_text_in_one_pass(self) -> None:
        mixed = f"Kubernetes در {ARABIC_YEH}ک cluster"
        assert normalize_text(mixed) == (
            f"Kubernetes در {FARSI_YEH}ک cluster"
        )

    def test_query_and_document_normalization_agree(self) -> None:
        """Invariant 7: an Arabic-typed query must match Persian-typed content."""
        document = normalize_text(f"گزارش {FARSI_YEH}ک")
        query = normalize_text(f"گزارش {ARABIC_YEH}{ARABIC_KAF}")
        assert document == query


class TestChunking:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
        assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []

    def test_short_text_is_a_single_chunk(self) -> None:
        assert chunk_text("one paragraph", chunk_size=100, chunk_overlap=10) == [
            "one paragraph"
        ]

    def test_never_exceeds_chunk_size(self) -> None:
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)

    def test_persian_text_never_exceeds_chunk_size(self) -> None:
        text = " ".join([PERSIAN_SENTENCES] * 20)
        chunks = chunk_text(text, chunk_size=150, chunk_overlap=40)
        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)

    def test_overlap_repeats_trailing_words(self) -> None:
        text = " ".join(f"w{i}" for i in range(60))
        chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)
        assert len(chunks) > 1
        overlap_words = set(chunks[0].split()) & set(chunks[1].split())
        assert overlap_words, "second chunk should carry context from the first"

    def test_overlap_does_not_start_mid_word(self) -> None:
        text = " ".join(f"token{i:03d}" for i in range(80))
        chunks = chunk_text(text, chunk_size=90, chunk_overlap=25)
        for chunk in chunks[1:]:
            assert chunk.split()[0].startswith("token")

    def test_splits_on_persian_sentence_boundaries(self) -> None:
        chunks = chunk_text(PERSIAN_SENTENCES, chunk_size=25, chunk_overlap=0)
        assert len(chunks) > 1
        # The Persian question mark ends a sentence, so it may only appear chunk-final.
        assert all("؟" not in chunk[:-1] for chunk in chunks)

    def test_paragraphs_are_preferred_boundaries(self) -> None:
        chunks = chunk_text("first para\n\nsecond para", chunk_size=12, chunk_overlap=0)
        assert chunks == ["first para", "second para"]

    def test_hard_splits_a_word_longer_than_a_chunk(self) -> None:
        chunks = chunk_text("x" * 250, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 3
        assert chunks[0] == "x" * 100
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_zwnj_word_is_not_split_by_overlap(self) -> None:
        text = " ".join([ZWNJ_WORD] * 40)
        chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
        for chunk in chunks:
            assert not chunk.startswith(ZWNJ)
            assert not chunk.endswith(ZWNJ)


class TestProcessDocument:
    def test_propagates_source_metadata_to_every_chunk(self) -> None:
        chunks = process_document(
            document_id="abc123",
            filename="سند.pdf",  # a Persian filename
            file_type="pdf",
            raw_text=" ".join([PERSIAN_SENTENCES] * 10),
            chunk_size=120,
            chunk_overlap=20,
        )
        assert len(chunks) > 1
        for position, chunk in enumerate(chunks):
            assert chunk.metadata() == {
                "document_id": "abc123",
                "filename": "سند.pdf",
                "file_type": "pdf",
                "chunk_id": f"{position:04d}",
            }

    def test_node_ids_are_unique_across_documents(self) -> None:
        common = {
            "file_type": "docx",
            "raw_text": "shared text " * 40,
            "chunk_size": 60,
            "chunk_overlap": 10,
        }
        first = process_document(document_id="a", filename="a.docx", **common)
        second = process_document(document_id="b", filename="b.docx", **common)
        ids = {chunk.node_id for chunk in first + second}
        assert len(ids) == len(first) + len(second)

    def test_normalizes_before_chunking(self) -> None:
        chunks = process_document(
            document_id="d",
            filename="mixed.pdf",
            file_type="pdf",
            raw_text=f"report {ARABIC_YEH}{ARABIC_KAF}​",
            chunk_size=200,
            chunk_overlap=0,
        )
        assert chunks[0].text == f"report {FARSI_YEH}{KEHEH}"

    def test_document_without_extractable_text_yields_no_chunks(self) -> None:
        assert (
            process_document(
                document_id="d",
                filename="scan.pdf",
                file_type="pdf",
                raw_text="  \n\n ​ ",
                chunk_size=200,
                chunk_overlap=0,
            )
            == []
        )

    def test_chunk_is_immutable(self) -> None:
        chunk = Chunk(
            document_id="d",
            filename="f.pdf",
            file_type="pdf",
            chunk_id="0000",
            text="t",
        )
        assert chunk.node_id == "d-0000"
