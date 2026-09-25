"""
DOCX / PPTX / plain-text / HTML extraction (Stage: multi-format support).

Deterministic — no AI. Produces the same `RawBlock` list shape as
`PDFExtractor` so the existing classifier / issue detector / OCR path
can consume it unchanged.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.services.extraction.pdf_extractor import ExtractionError, RawBlock, RawTableCell

logger = logging.getLogger(__name__)

_HEADING_STYLE_RE = re.compile(r"^Heading\s*(\d+)$", re.IGNORECASE)


class OfficeExtractor:
    """Extract structured blocks from Office and plain-text documents."""

    def extract_docx(self, file_path: str) -> list[RawBlock]:
        try:
            from docx import Document
            from docx.table import Table as DocxTable
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise ExtractionError(
                "python-docx is required for DOCX extraction. "
                "Install it (see requirements.txt)."
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            doc = Document(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open DOCX: {exc}") from exc

        blocks: list[RawBlock] = []
        order_y = 0.0

        # python-docx body is a sequence of paragraphs and tables interleaved.
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
            if tag == "p":
                para = Paragraph(element, doc)
                text = (para.text or "").strip()
                if not text:
                    continue
                style_name = ""
                try:
                    style_name = para.style.name if para.style is not None else ""
                except Exception:
                    style_name = ""
                heading_level = None
                font_size = 12.0
                is_bold = False
                m = _HEADING_STYLE_RE.match(style_name or "")
                if m:
                    heading_level = max(1, min(6, int(m.group(1))))
                    font_size = 22.0 - (heading_level - 1) * 2.0
                    is_bold = True
                elif style_name and "title" in style_name.lower():
                    heading_level = 1
                    font_size = 22.0
                    is_bold = True
                # Bullet / numbered lists via style name heuristic.
                list_items = None
                if style_name and "list" in style_name.lower():
                    list_items = [text]
                blocks.append(
                    RawBlock(
                        page=1,
                        text=text if list_items is None else "\n".join(list_items),
                        x0=72.0,
                        y0=order_y,
                        x1=540.0,
                        y1=order_y + 14.0,
                        font_size=font_size,
                        is_bold=is_bold,
                    )
                )
                # Stash heading_level in text prefix? Classifier uses font_size.
                # For list, use multi-line with bullet so classifier list heuristic fires.
                if list_items:
                    blocks[-1].text = f"• {text}"
                order_y += 20.0
            elif tag == "tbl":
                table = DocxTable(element, doc)
                cells: list[RawTableCell] = []
                flat_parts: list[str] = []
                for r_idx, row in enumerate(table.rows):
                    for c_idx, cell in enumerate(row.cells):
                        cell_text = (cell.text or "").strip()
                        cells.append(
                            RawTableCell(
                                row=r_idx,
                                col=c_idx,
                                text=cell_text,
                                is_header=(r_idx == 0),
                            )
                        )
                        if cell_text:
                            flat_parts.append(cell_text)
                if cells:
                    blocks.append(
                        RawBlock(
                            page=1,
                            text=" | ".join(flat_parts),
                            x0=72.0,
                            y0=order_y,
                            x1=540.0,
                            y1=order_y + 40.0,
                            cells=cells,
                        )
                    )
                    order_y += 48.0

        if not blocks:
            raise ExtractionError("DOCX contained no extractable text or tables")
        return blocks

    def extract_pptx(self, file_path: str) -> list[RawBlock]:
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError as exc:
            raise ExtractionError(
                "python-pptx is required for PPTX extraction. "
                "Install it (see requirements.txt)."
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            prs = Presentation(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PPTX: {exc}") from exc

        blocks: list[RawBlock] = []
        for page_index, slide in enumerate(prs.slides):
            page = page_index + 1
            y = 0.0
            for shape in slide.shapes:
                if shape.has_table:
                    table = shape.table
                    cells: list[RawTableCell] = []
                    flat: list[str] = []
                    for r_idx, row in enumerate(table.rows):
                        for c_idx, cell in enumerate(row.cells):
                            text = (cell.text or "").strip()
                            cells.append(
                                RawTableCell(
                                    row=r_idx,
                                    col=c_idx,
                                    text=text,
                                    is_header=(r_idx == 0),
                                )
                            )
                            if text:
                                flat.append(text)
                    if cells:
                        blocks.append(
                            RawBlock(
                                page=page,
                                text=" | ".join(flat),
                                x0=40.0,
                                y0=y,
                                x1=680.0,
                                y1=y + 40.0,
                                cells=cells,
                            )
                        )
                        y += 48.0
                    continue

                if not shape.has_text_frame:
                    continue
                text = (shape.text_frame.text or "").strip()
                if not text:
                    continue
                # Title placeholder often means heading.
                is_title = False
                try:
                    is_title = bool(shape.is_placeholder and shape.placeholder_format.idx == 0)
                except Exception:
                    is_title = False
                font_size = 20.0 if is_title else 14.0
                blocks.append(
                    RawBlock(
                        page=page,
                        text=text,
                        x0=40.0,
                        y0=y,
                        x1=680.0,
                        y1=y + (24.0 if is_title else 16.0),
                        font_size=font_size,
                        is_bold=is_title,
                    )
                )
                y += 28.0 if is_title else 20.0

                # Nested images on slides are not extracted as files here;
                # OCR path covers pure image uploads.

        if not blocks:
            raise ExtractionError("PPTX contained no extractable text or tables")
        return blocks

    def extract_text(self, file_path: str) -> list[RawBlock]:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionError(f"Could not read text file: {exc}") from exc

        blocks: list[RawBlock] = []
        y = 0.0
        for para in content.splitlines():
            text = para.strip()
            if not text:
                y += 8.0
                continue
            is_heading = text.startswith("#")
            clean = text.lstrip("#").strip() if is_heading else text
            level = len(text) - len(text.lstrip("#")) if is_heading else 0
            blocks.append(
                RawBlock(
                    page=1,
                    text=clean,
                    x0=72.0,
                    y0=y,
                    x1=540.0,
                    y1=y + 14.0,
                    font_size=22.0 - max(0, level - 1) * 2.0 if is_heading else 12.0,
                    is_bold=is_heading,
                )
            )
            y += 18.0
        if not blocks:
            raise ExtractionError("Text file was empty")
        return blocks
