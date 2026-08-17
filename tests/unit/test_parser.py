"""PDF/DOCX text extraction against real, minimal files (R-10).

The RTL tests build synthetic PDFs at the byte level (`tests/pdf_fixtures.py`) so the
fixture reproduces the real mechanism of the bug — visual- vs logical-order glyph
placement — instead of asserting behavior from documentation.
"""

from __future__ import annotations

import re

import pytest

from app.documents.parser import ParsingError, extract_text
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf

PERSIAN_SENTENCE = "این یک گزارش آزمایشی است."
MIXED_SENTENCE = "تیم زیرساخت Kubernetes را نصب کرد."


def _squeeze(text: str) -> str:
    """Collapse runs of spaces, as `processor.normalize_text` does downstream.

    `pypdf`'s layout mode pads text with spaces to preserve column position; that is
    the parser's business, not this test's, so comparisons squeeze it away.
    """
    return "\n\n".join(
        re.sub(r" +", " ", block).strip() for block in text.split("\n\n")
    )


class TestDispatch:
    def test_unknown_file_type_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file_type"):
            extract_text(file_type="txt", content=b"data")


class TestPdfExtraction:
    def test_extracts_english_text(self) -> None:
        pdf = build_pdf(["The quarterly report is final."])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == "The quarterly report is final."

    def test_extracts_well_authored_persian_text(self) -> None:
        """A producer that writes glyphs in true logical order extracts correctly."""
        pdf = build_pdf([PERSIAN_SENTENCE])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == PERSIAN_SENTENCE

    def test_extracts_mixed_persian_and_english_without_dropping_either(self) -> None:
        pdf = build_pdf([MIXED_SENTENCE])
        extracted = extract_text(file_type="pdf", content=pdf)
        for word in MIXED_SENTENCE.split(" "):
            assert word in extracted

    def test_joins_pages_as_paragraph_breaks(self) -> None:
        pdf = build_pdf(["First page.", "Second page."])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == "First page.\n\nSecond page."

    def test_skips_blank_pages(self) -> None:
        pdf = build_pdf(["First page.", "", "Third page."])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == "First page.\n\nThird page."

    def test_unreadable_bytes_raise_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="pdf", content=b"not a pdf at all")

    def test_empty_bytes_raise_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="pdf", content=b"")


class TestPdfRtlLimitation:
    """Pins the documented, unresolved behavior in ARCHITECTURE.md open question 4."""

    def test_plain_mode_would_drop_content_on_mixed_direction_lines(self) -> None:
        """Regression guard for why parser.py does not use pypdf's default mode.

        Verified directly against pypdf: its "plain" extraction mode's bidi heuristic
        can silently discard an entire run of text on a line mixing right-to-left and
        left-to-right content. If a pypdf upgrade fixes this, this test starts failing
        and parser.py's extraction_mode choice should be revisited, not the test.
        """
        import io

        import pypdf

        pdf = build_pdf([MIXED_SENTENCE])
        reader = pypdf.PdfReader(io.BytesIO(pdf))
        plain = reader.pages[0].extract_text()
        assert not all(word in plain for word in MIXED_SENTENCE.split(" "))

    def test_layout_mode_does_not_drop_content(self) -> None:
        """What parser.py actually uses does not have the defect above."""
        pdf = build_pdf([MIXED_SENTENCE])
        extracted = extract_text(file_type="pdf", content=pdf)
        assert all(word in extracted for word in MIXED_SENTENCE.split(" "))

    def test_visual_order_producer_extracts_mirror_reversed(self) -> None:
        """The residual, unfixed limitation: byte order is trusted, not corrected."""
        pdf = build_pdf(["گزارش"], reverse_bytes=True)
        extracted = extract_text(file_type="pdf", content=pdf)
        assert extracted != "گزارش"
        assert extracted == "".join(reversed("گزارش"))

    def test_logical_order_producer_is_unaffected_by_the_limitation(self) -> None:
        pdf = build_pdf(["گزارش"])
        assert extract_text(file_type="pdf", content=pdf) == "گزارش"


class TestDocxExtraction:
    def test_extracts_english_paragraphs(self) -> None:
        docx_bytes = build_docx(["First paragraph.", "Second paragraph."])
        assert extract_text(file_type="docx", content=docx_bytes) == (
            "First paragraph.\n\nSecond paragraph."
        )

    def test_extracts_persian_paragraphs_in_logical_order(self) -> None:
        """DOCX has no visual-order bug: `w:t` text is always logical order."""
        docx_bytes = build_docx([PERSIAN_SENTENCE, MIXED_SENTENCE])
        assert extract_text(file_type="docx", content=docx_bytes) == (
            f"{PERSIAN_SENTENCE}\n\n{MIXED_SENTENCE}"
        )

    def test_extracts_table_cells(self) -> None:
        docx_bytes = build_docx(
            ["Costs by team"],
            tables=[[["Team", "Cost"], ["زیرساخت", "۱۲۰"]]],
        )
        extracted = extract_text(file_type="docx", content=docx_bytes)
        assert "Costs by team" in extracted
        assert "Team\tCost" in extracted
        assert "زیرساخت\t۱۲۰" in extracted

    def test_skips_empty_paragraphs(self) -> None:
        docx_bytes = build_docx(["", "Only real paragraph.", "   "])
        assert extract_text(file_type="docx", content=docx_bytes) == (
            "Only real paragraph."
        )

    def test_document_with_no_text_yields_empty_string(self) -> None:
        assert extract_text(file_type="docx", content=build_docx([])) == ""

    def test_unreadable_bytes_raise_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="docx", content=b"not a docx at all")

    def test_empty_bytes_raise_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="docx", content=b"")
