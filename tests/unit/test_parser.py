"""PDF/DOCX text extraction against real, minimal files (R-10).

The RTL tests build synthetic PDFs at the byte level (`tests/pdf_fixtures.py`) so the
fixture reproduces the real mechanism of correct RTL extraction — visual-order glyph
placement, the only way a real PDF producer stores right-to-left text — instead of
asserting behavior from documentation. See that module's docstring for why.
"""

from __future__ import annotations

import re
import subprocess

import pytest

import app.documents.parser as parser_module
from app.documents.parser import ParsingError, extract_text
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf

PERSIAN_SENTENCE = "این یک گزارش آزمایشی است."
MIXED_SENTENCE = "تیم زیرساخت Kubernetes را نصب کرد."


_BIDI_CONTROL_CHARS = "\u202a\u202b\u202c\u202d\u202e"


def _squeeze(text: str) -> str:
    """Collapse runs of spaces and drop bidi-embedding control characters, both as
    `processor.normalize_text` does downstream.

    `pdftotext -layout` pads text with spaces to preserve column position, and wraps
    right-to-left runs in explicit bidi-embedding marks (RLE/PDF and similar) to
    preserve logical order unambiguously in plain text; both are the parser's business,
    not this test's, so comparisons remove them the same way normalization would.
    """
    stripped = "".join(char for char in text if char not in _BIDI_CONTROL_CHARS)
    return "\n\n".join(
        re.sub(r" +", " ", block).strip() for block in stripped.split("\n\n")
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

    def test_extracts_persian_text_in_correct_reading_order(self) -> None:
        """Real Persian PDFs store right-to-left glyph runs in visual order (see
        `pdf_fixtures.py`); `pdftotext -layout` corrects this back to logical order.
        """
        pdf = build_pdf([PERSIAN_SENTENCE])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == PERSIAN_SENTENCE

    def test_extracts_mixed_persian_and_english_in_correct_order(self) -> None:
        pdf = build_pdf([MIXED_SENTENCE])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == MIXED_SENTENCE

    def test_joins_pages_as_paragraph_breaks(self) -> None:
        pdf = build_pdf(["First page.", "Second page."])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == "First page.\n\nSecond page."

    def test_skips_blank_pages(self) -> None:
        pdf = build_pdf(["First page.", "", "Third page."])
        extracted = _squeeze(extract_text(file_type="pdf", content=pdf))
        assert extracted == "First page.\n\nThird page."

    def test_malformed_pdf_raises_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="pdf", content=b"not a pdf at all")

    def test_empty_bytes_raise_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            extract_text(file_type="pdf", content=b"")


class TestPdfExtractionFailureModes:
    """`pdftotext` failure modes that real malformed input can't reliably trigger on
    demand — the binary missing, a non-zero exit for a reason other than a bad PDF, and
    a hang — are exercised directly via a stubbed `subprocess.run` (invariant: no live
    process is spawned by these three tests).
    """

    def test_missing_binary_raises_parsing_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("pdftotext")

        monkeypatch.setattr(parser_module.subprocess, "run", _boom)
        with pytest.raises(ParsingError, match="poppler-utils"):
            extract_text(file_type="pdf", content=b"irrelevant")

    def test_nonzero_exit_raises_parsing_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fake_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=2, stdout="", stderr="some poppler failure"
            )

        monkeypatch.setattr(parser_module.subprocess, "run", _fake_run)
        with pytest.raises(ParsingError, match="some poppler failure"):
            extract_text(file_type="pdf", content=b"irrelevant")

    def test_timeout_raises_parsing_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _hang(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="pdftotext", timeout=30)

        monkeypatch.setattr(parser_module.subprocess, "run", _hang)
        with pytest.raises(ParsingError, match="did not finish"):
            extract_text(file_type="pdf", content=b"irrelevant")


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
