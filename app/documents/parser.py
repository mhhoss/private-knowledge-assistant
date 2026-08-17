"""PDF and DOCX text extraction.

Extracts raw text only. Normalization and chunking happen in `processor.py`; this module
does not know about languages, scripts, or the pipeline stages around it.

PDF extraction uses `pypdf`'s "layout" mode, not its default "plain" mode. Plain mode
applies a heuristic bidi correction that, on any line mixing right-to-left and
left-to-right runs (Persian text next to an English term or a number), can silently
drop entire runs of text rather than misorder them — verified directly against `pypdf`,
not assumed. Layout mode never drops content, at the cost of not attempting that
correction: a PDF producer that emits right-to-left glyphs in visual (not logical) order
still extracts mirror-reversed. That residual limitation is unfixable here — see open
question 4 in ARCHITECTURE.md — but silent content loss would have been a groundedness
bug, not just an ordering one, which is why layout mode is the correct default despite
being the less "clever" one.
"""

from __future__ import annotations

import io

import docx
import pypdf


class ParsingError(RuntimeError):
    """A file could not be read as valid PDF/DOCX, regardless of the reason."""


# Pages/paragraphs join with a blank line so `processor.chunk_text` treats each as a
# paragraph boundary rather than run-on text.
_BLOCK_SEP = "\n\n"


def extract_text(*, file_type: str, content: bytes) -> str:
    """Extract raw text for a file already identified as `file_type` by the loader."""
    if file_type == "pdf":
        return _extract_pdf(content)
    if file_type == "docx":
        return _extract_docx(content)
    raise ValueError(f"Unsupported file_type: {file_type!r}")


def _extract_pdf(content: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [
            page.extract_text(extraction_mode="layout") for page in reader.pages
        ]
    except Exception as error:  # noqa: BLE001 - any library failure is a parsing error
        raise ParsingError(f"Could not read PDF: {error}") from error
    return _BLOCK_SEP.join(page.strip() for page in pages if page.strip())


def _extract_docx(content: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        tables = [_render_table(table) for table in document.tables]
    except Exception as error:  # noqa: BLE001 - any library failure is a parsing error
        raise ParsingError(f"Could not read DOCX: {error}") from error
    return _BLOCK_SEP.join(block for block in (*paragraphs, *tables) if block)


def _render_table(table: docx.table.Table) -> str:
    """Render a table as one row per line, cells tab-separated."""
    rows = ("\t".join(cell.text for cell in row.cells) for row in table.rows)
    return "\n".join(row for row in rows if row.strip())
