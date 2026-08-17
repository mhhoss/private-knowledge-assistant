"""File intake: type dispatch and content-derived identity (ADR-3)."""

from __future__ import annotations

import pytest

from app.documents.loader import (
    UnsupportedFileTypeError,
    compute_document_id,
    detect_file_type,
    load,
)
from app.documents.parser import ParsingError
from tests.docx_fixtures import build_docx
from tests.pdf_fixtures import build_pdf


class TestDetectFileType:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("report.pdf", "pdf"),
            ("REPORT.PDF", "pdf"),
            ("گزارش.pdf", "pdf"),
            ("notes.docx", "docx"),
            ("Notes.DOCX", "docx"),
            ("archive.tar.docx", "docx"),  # only the final suffix matters
        ],
    )
    def test_recognizes_supported_extensions_case_insensitively(
        self, filename: str, expected: str
    ) -> None:
        assert detect_file_type(filename) == expected

    @pytest.mark.parametrize(
        "filename", ["report.txt", "report.doc", "report.pptx", "noextension"]
    )
    def test_rejects_unsupported_extensions(self, filename: str) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(filename)


class TestComputeDocumentId:
    def test_identical_content_yields_identical_id(self) -> None:
        content = b"same bytes"
        assert compute_document_id(content) == compute_document_id(content)

    def test_different_content_yields_different_id(self) -> None:
        assert compute_document_id(b"a") != compute_document_id(b"b")

    def test_id_does_not_depend_on_filename(self) -> None:
        """ADR-3: identity is content, not the upload event."""
        content = "گزارش فصلی".encode()
        doc_a = load(filename="a.pdf", content=build_pdf(["گزارش فصلی"]))
        doc_b = load(filename="b_renamed.pdf", content=build_pdf(["گزارش فصلی"]))
        assert doc_a.document_id == doc_b.document_id
        assert compute_document_id(content) != doc_a.document_id  # sanity: not a fluke


class TestLoad:
    def test_loads_an_english_pdf(self) -> None:
        result = load(filename="report.pdf", content=build_pdf(["Quarterly report."]))
        assert result.file_type == "pdf"
        assert result.filename == "report.pdf"
        assert "Quarterly" in result.raw_text and "report." in result.raw_text
        assert result.document_id == compute_document_id(
            build_pdf(["Quarterly report."])
        )

    def test_loads_a_persian_docx(self) -> None:
        content = build_docx(["این یک سند فارسی است."])
        result = load(filename="سند.docx", content=content)
        assert result.file_type == "docx"
        assert result.filename == "سند.docx"
        assert result.raw_text == "این یک سند فارسی است."

    def test_unsupported_extension_is_rejected_before_parsing(self) -> None:
        """Bad extension fails even though the bytes are not valid PDF/DOCX either."""
        with pytest.raises(UnsupportedFileTypeError):
            load(filename="notes.txt", content=b"plain text, not a real document")

    def test_matching_extension_with_corrupt_content_raises_parsing_error(self) -> None:
        with pytest.raises(ParsingError):
            load(filename="broken.pdf", content=b"not actually a pdf")

    def test_document_with_no_extractable_text_loads_with_empty_raw_text(self) -> None:
        """An image-only/blank document is not a loader failure; it has no text."""
        result = load(filename="blank.docx", content=build_docx([]))
        assert result.raw_text == ""
