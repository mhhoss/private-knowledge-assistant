"""Hand-built minimal PDFs for parser tests.

No PDF-writing library is a project dependency, so these are assembled at the byte
level: a single-page (or multi-page) PDF with one Type1 font whose `/ToUnicode` CMap
maps arbitrary byte codes to arbitrary Unicode codepoints (including Persian). This lets
tests control exactly what byte order lands in each page's content stream, which is the
mechanism of the real-world bug being tested: some PDF producers place right-to-left
glyphs in visual (not logical) order, so `pypdf` — which decodes bytes via ToUnicode in
stream order — extracts them reversed. Building the fixture this way reproduces that bug
directly rather than asserting it from documentation.
"""

from __future__ import annotations

import io


def build_pdf(pages: list[str], *, reverse_bytes: bool = False) -> bytes:
    """A minimal PDF whose extracted text is each page in `pages`, joined by pypdf.

    `reverse_bytes=True` reverses the byte order within each page's content stream
    (not the ToUnicode mapping), simulating a producer that draws right-to-left glyphs
    in visual order — the classic "reversed Farsi" PDF bug.
    """
    charset = sorted({char for page in pages for char in page})
    if len(charset) > 256:
        raise ValueError("fixture supports at most 256 distinct characters")
    code_of = {char: index for index, char in enumerate(charset)}

    tounicode = _build_tounicode_cmap(charset)
    font_obj = 3
    tounicode_obj = 4

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        font_obj: (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/ToUnicode %d 0 R >>" % tounicode_obj
        ),
        tounicode_obj: _stream_object(tounicode),
    }

    page_objs = []
    next_obj = 5
    for page_text in pages:
        content = _content_stream(page_text, code_of, reverse_bytes=reverse_bytes)

        content_obj = next_obj
        page_obj = next_obj + 1
        next_obj += 2

        objects[content_obj] = _stream_object(content)
        objects[page_obj] = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents %d 0 R >>"
        ) % (font_obj, content_obj)
        page_objs.append(page_obj)

    kids = b" ".join(b"%d 0 R" % obj for obj in page_objs)
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_objs))

    return _assemble(objects, root=1)


def _content_stream(
    text: str, code_of: dict[str, int], *, reverse_bytes: bool
) -> bytes:
    """One `Tj` per word, each preceded by a rightward `Td` advance.

    This is how real, non-bidi-aware layout engines place a line of text: each word is
    its own showing operator, positioned left to right in whatever order the producer
    hands them over — never one giant string spanning direction changes. Encoding a
    whole mixed-direction sentence into a single `Tj` is not representative of real PDF
    output and confuses `pypdf`'s own bidi heuristics in ways a real document would not.
    """
    ops = [b"BT /F1 24 Tf 72 720 Td"]
    for word in text.split(" "):
        codes = bytes(code_of[char] for char in word)
        if reverse_bytes:
            codes = bytes(reversed(codes))
        ops.append(b"<" + codes.hex().encode("ascii") + b"> Tj")
        # Advance well past this word's own width (no /Widths array is defined, so
        # readers fall back to generous default glyph widths) plus a clear gap, so
        # layout-mode extraction reliably sees a word boundary.
        advance = max(400, 60 * len(word) + 250)
        ops.append(b"%d 0 Td" % advance)
    ops.append(b"ET")
    return b"\n".join(ops)


def _build_tounicode_cmap(charset: list[str]) -> bytes:
    entries = "\n".join(
        f"<{i:02X}> <{ord(char):04X}>" for i, char in enumerate(charset)
    )
    return f"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<00> <FF>
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
