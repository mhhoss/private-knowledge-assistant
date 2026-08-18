"""Hand-built minimal PDFs for parser tests.

No PDF-writing library is a project dependency, so these are assembled at the byte
level: a single-page (or multi-page) PDF using a `Type0`/`Identity-H` composite font
with a `ToUnicode` CMap mapping arbitrary CID codes to arbitrary Unicode codepoints
(including Persian), the same standards-compliant mechanism real PDF producers use to
embed non-Latin text. This lets tests control exactly what glyph order lands in each
page's content stream.

Real RTL text in a PDF content stream is always stored in *visual* (shaped) order, never
raw logical/typing order — a text-shaping engine (HarfBuzz, CoreText, DirectWrite, ...)
resolves bidi and glyph order before anything is drawn, because the PDF renderer itself
never reorders glyphs; it just places them at the given coordinates in stream order. This
was verified directly against real LibreOffice- and Chrome-generated Persian PDFs during
the poppler evaluation, not assumed. `build_pdf`'s default (`visual_order=True`)
reproduces that: each maximal run of Arabic-script characters within a "word" is stored
character-reversed (trailing neutral punctuation attaches to its preceding strong-direction
run, matching standard bidi resolution for weak/neutral characters), while embedded Latin
runs (e.g. an English term inside a Persian sentence) are left untouched — reproducing the
real mechanism, not asserting it from documentation.
"""

from __future__ import annotations

import io

_ARABIC_SCRIPT = range(0x0600, 0x06FF + 1)


def build_pdf(
    pages: list[str], *, visual_order: bool = True, page_width: int = 6000
) -> bytes:
    """A minimal PDF whose extracted text is each page in `pages`, joined by `pdftotext`.

    `visual_order=True` (the default, and the only realistic case — see module
    docstring) stores each word's Arabic-script runs character-reversed, as a real
    bidi-aware producer would. `visual_order=False` stores every character in raw
    left-to-right typing order regardless of script, an unrealistic case no real PDF
    producer emits for RTL text; it exists only to demonstrate the parser is trusting
    the stream, not correcting it.

    `page_width` is generous (not U.S. Letter's 612pt) so that `pdftotext -layout`,
    which — unlike the old `pypdf` extraction this project used previously — respects
    page geometry, never clips a test sentence's word advances off the visible page.
    """
    charset = sorted({char for page in pages for char in page})
    if len(charset) > 0xFFFF:
        raise ValueError("fixture supports at most 65535 distinct characters")
    code_of = {
        char: index + 1 for index, char in enumerate(charset)
    }  # CID 0 is reserved

    tounicode = _build_tounicode_cmap(charset, code_of)
    font_obj = 3
    descendant_obj = 4
    descriptor_obj = 5
    tounicode_obj = 6

    objects: dict[int, bytes] = {
        font_obj: (
            b"<< /Type /Font /Subtype /Type0 /BaseFont /Fake+Custom "
            b"/Encoding /Identity-H /DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>"
        )
        % (descendant_obj, tounicode_obj),
        descendant_obj: (
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Fake+Custom "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            b"/FontDescriptor %d 0 R /DW 1000 >>"
        )
        % (descriptor_obj,),
        descriptor_obj: (
            b"<< /Type /FontDescriptor /FontName /Fake+Custom /Flags 4 "
            b"/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200 "
            b"/CapHeight 700 /StemV 80 >>"
        ),
        tounicode_obj: _stream_object(tounicode),
    }

    page_objs = []
    next_obj = 7
    for page_text in pages:
        content = _content_stream(page_text, code_of, visual_order=visual_order)

        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2

        objects[content_obj] = _stream_object(content)
        objects[page_obj] = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> >> "
            b"/MediaBox [0 0 %d 792] /Contents %d 0 R >>"
        ) % (font_obj, page_width, content_obj)
        page_objs.append(page_obj)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = b" ".join(b"%d 0 R" % obj for obj in page_objs)
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_objs))

    return _assemble(objects, root=1)


def _content_stream(text: str, code_of: dict[str, int], *, visual_order: bool) -> bytes:
    """One `Tj` per word, each preceded by a rightward `Td` advance.

    This is how real layout engines place a line of text at the PDF-content-stream
    level: each word is its own showing operator, positioned left to right by whatever
    order the producer hands them over.
    """
    ops = [b"BT /F1 24 Tf 72 720 Td"]
    for word in text.split(" "):
        glyphs = _visual_order(word) if visual_order else word
        codes = [code_of[char] for char in glyphs]
        hex_codes = "".join(f"{code:04X}" for code in codes)
        ops.append(f"<{hex_codes}> Tj".encode("ascii"))
        # Advance well past this word's own width (no /Widths array is defined, so
        # readers fall back to generous default glyph widths) plus a clear gap, so
        # layout-mode extraction reliably sees a word boundary.
        advance = max(400, 60 * len(word) + 250)
        ops.append(b"%d 0 Td" % advance)
    ops.append(b"ET")
    return b"\n".join(ops)


def _is_rtl(char: str) -> bool:
    return ord(char) in _ARABIC_SCRIPT


def _visual_order(word: str) -> str:
    """Reverse only maximal runs of Arabic-script characters within `word`, leaving
    embedded Latin/digit/punctuation runs untouched — except that a neutral character
    (punctuation, digits) attaches to its preceding strong-direction run rather than
    starting a new one, matching standard bidi weak/neutral-character resolution.
    """
    if not word:
        return word

    def strong_class(char: str) -> str | None:
        if _is_rtl(char):
            return "R"
        if char.isalpha():
            return "L"
        return None  # neutral

    classes = [strong_class(char) for char in word]
    first_strong = next((c for c in classes if c is not None), "R")
    resolved = []
    last_strong = first_strong
    for cls in classes:
        if cls is None:
            resolved.append(last_strong)
        else:
            resolved.append(cls)
            last_strong = cls

    runs: list[tuple[str, list[str]]] = []
    for char, cls in zip(word, resolved):
        if runs and runs[-1][0] == cls:
            runs[-1][1].append(char)
        else:
            runs.append((cls, [char]))

    return "".join(
        "".join(reversed(chars)) if cls == "R" else "".join(chars)
        for cls, chars in runs
    )


def _build_tounicode_cmap(charset: list[str], code_of: dict[str, int]) -> bytes:
    entries = "\n".join(f"<{code_of[char]:04X}> <{ord(char):04X}>" for char in charset)
    return f"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
{len(charset)} beginbfchar
{entries}
endbfchar
endcmap
end
end
""".encode("ascii")


def _stream_object(body: bytes) -> bytes:
    return b"<< /Length %d >>\nstream\n%s\nendstream" % (len(body), body)


def _assemble(objects: dict[int, bytes], *, root: int) -> bytes:
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = out.tell()
        out.write(b"%d 0 obj\n" % num)
        out.write(objects[num])
        out.write(b"\nendobj\n")

    xref_offset = out.tell()
    count = max(objects) + 1
    out.write(b"xref\n0 %d\n" % count)
    out.write(b"0000000000 65535 f \n")
    for num in range(1, count):
        offset = offsets.get(num, 0)
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n")
    out.write(b"<< /Size %d /Root %d 0 R >>\n" % (count, root))
    out.write(b"startxref\n%d\n%%%%EOF" % xref_offset)
    return out.getvalue()
