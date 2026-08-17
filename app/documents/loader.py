"""File intake: type dispatch and document identity.

`document_id` is derived from file content, not the upload event (ADR-3): re-uploading
identical bytes yields the same id regardless of filename, so the indexer can detect it
as already-indexed rather than duplicating chunks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.documents.parser import extract_text

# 128 bits of a content hash is far more collision resistance than a single-user
# knowledge base needs; kept short for readable citations and logs.
_DOCUMENT_ID_LENGTH = 32

_EXTENSION_TO_FILE_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
}


class UnsupportedFileTypeError(ValueError):
    """The upload's extension is not one the pipeline can parse."""


@dataclass(frozen=True)
class LoadedDocument:
    """A file, identified and with its text extracted. Not yet normalized or chunked."""

    document_id: str
    filename: str
    file_type: str
    raw_text: str


def compute_document_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:_DOCUMENT_ID_LENGTH]


def detect_file_type(filename: str) -> str:
    """Resolve a supported `file_type` from a filename's extension.

    Raises `UnsupportedFileTypeError` for anything else — including no extension.
    """
    suffix = PurePosixPath(filename).suffix.lower()
    try:
        return _EXTENSION_TO_FILE_TYPE[suffix]
    except KeyError:
        supported = ", ".join(sorted(_EXTENSION_TO_FILE_TYPE))
        raise UnsupportedFileTypeError(
            f"Unsupported file type {suffix or '(none)'!r} for {filename!r}. "
            f"Supported: {supported}."
        ) from None


def load(*, filename: str, content: bytes) -> LoadedDocument:
    """Identify, dispatch, and extract text for one uploaded file.

    Raises `UnsupportedFileTypeError` for an unrecognized extension, or
    `parser.ParsingError` if the file cannot be read as its detected type. Both are
    caller errors for the orchestrator (`rag/engine.py`) to convert into a failed
    ingestion outcome (R-09) — this function never itself returns a failure value.
    """
    file_type = detect_file_type(filename)
    raw_text = extract_text(file_type=file_type, content=content)
    return LoadedDocument(
        document_id=compute_document_id(content),
        filename=filename,
        file_type=file_type,
        raw_text=raw_text,
    )
