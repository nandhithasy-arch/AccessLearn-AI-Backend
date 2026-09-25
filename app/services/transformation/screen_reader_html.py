"""
Deterministic (not AI) HTML generation from the classified sections.

This is intentionally code, not a model call: heading hierarchy, landmarks,
table headers and alt text are structural facts we already have in the
StructuredDocument -- generating them via a prompt would just reintroduce
the risk of an LLM silently dropping structure.
"""
from __future__ import annotations

from html import escape

from app.models.document import DocumentSection, ElementType, StructuredDocument


class ScreenReaderHTMLGenerator:
    def generate(self, document: StructuredDocument) -> str:
        parts: list[str] = [
            "<!DOCTYPE html>",
            f'<html lang="{document.language or "en"}">',
            "<head><meta charset='utf-8'>",
            f"<title>{escape(document.document_title or document.source_filename)}</title>",
            "</head><body>",
            "<main>",
        ]
        for section in document.sections:
            parts.append(self._render_section(section))
        parts.append("</main></body></html>")
        return "\n".join(parts)

    def _render_section(self, section: DocumentSection) -> str:
        if section.type == ElementType.HEADING:
            level = section.heading_level or 2
            return f"<h{level}>{escape(section.text or '')}</h{level}>"
        if section.type == ElementType.PARAGRAPH:
            return f"<p>{escape(section.text or '')}</p>"
        if section.type == ElementType.LIST:
            tag = "ol" if section.ordered else "ul"
            items = "".join(f"<li>{escape(i)}</li>" for i in (section.list_items or []))
            return f"<{tag}>{items}</{tag}>"
        if section.type == ElementType.TABLE:
            return self._render_table(section)
        if section.type in (ElementType.IMAGE, ElementType.DIAGRAM, ElementType.CHART):
            alt = escape(section.alt_text or "")
            desc = f"<figcaption>{escape(section.description)}</figcaption>" if section.description else ""
            return f'<figure><img src="{escape(section.image_ref or "")}" alt="{alt}">{desc}</figure>'
        if section.type == ElementType.EQUATION:
            spoken = escape(section.equation_spoken or section.equation_raw or "")
            mathml = section.equation_mathml or ""
            return f'<div role="math" aria-label="{spoken}">{mathml or spoken}</div>'
        if section.type == ElementType.CODE:
            return f"<pre><code>{escape(section.text or '')}</code></pre>"
        # footnote / citation / caption / decorative -> fall through to plain text
        return f"<p>{escape(section.text or '')}</p>" if section.text else ""

    def _render_table(self, section: DocumentSection) -> str:
        if not section.cells:
            return ""
        max_row = max(c.row for c in section.cells)
        rows_html = []
        for r in range(max_row + 1):
            row_cells = [c for c in section.cells if c.row == r]
            row_cells.sort(key=lambda c: c.col)
            cells_html = "".join(
                f'<th scope="col">{escape(c.text)}</th>' if c.is_header else f"<td>{escape(c.text)}</td>"
                for c in row_cells
            )
            rows_html.append(f"<tr>{cells_html}</tr>")
        return f"<table>{''.join(rows_html)}</table>"
