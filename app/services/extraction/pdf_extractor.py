"""
Stage 2 of the pipeline: text and layout extraction.

Deterministic code (per spec section 9) -- no AI calls here. This produces
the raw block list with coordinates; classification into heading/paragraph/
table/etc happens later in services/analysis/classifier.py.

Implementation status:
    - PyMuPDF (fitz) walks pages, extracts text blocks with bboxes, font
      size and boldness (needed by the classifier's heading heuristic).
    - Multi-column layout: naive x-coordinate gap clustering.
    - Embedded images extracted and written via FileStorage.
    - Table extraction via pdfplumber, producing TABLE-shaped RawBlocks
      with row/col cell data (overlapping text blocks are suppressed so
      tables are not also emitted as paragraphs).
    - Pages with near-zero extractable text are flagged by is_scanned();
      OCR is handled by ocr_service.py and the /analyze route.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import fitz  # PyMuPDF

if TYPE_CHECKING:
    from app.storage.file_storage import FileStorage

logger = logging.getLogger(__name__)

_COLUMN_GAP_THRESHOLD_PT = 50.0
_BOLD_FLAG_BIT = 1 << 4
_SCANNED_CHARS_PER_PAGE_THRESHOLD = 20
# Text block whose bbox overlaps a detected table by more than this fraction
# of its own area is dropped (the table RawBlock already carries that content).
_TABLE_OVERLAP_DROP_RATIO = 0.5


class ExtractionError(Exception):
    """Raised when a PDF can't be opened or read at all."""


@dataclass
class RawTableCell:
    row: int
    col: int
    text: str
    is_header: bool = False


@dataclass
class RawBlock:
    page: int  # 1-indexed
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    is_image: bool = False
    image_ref: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    column: int = field(default=0, compare=False)
    # Table payload -- when set, classifier emits ElementType.TABLE.
    cells: list[RawTableCell] | None = None
    # OCR confidence (0-1) when this block came from OCRService.
    ocr_confidence: float | None = None

    @property
    def is_table(self) -> bool:
        return bool(self.cells)


class PDFExtractor:
    def extract(
        self,
        file_path: str,
        *,
        document_id: str | None = None,
        storage: "FileStorage | None" = None,
    ) -> list[RawBlock]:
        """
        Returns raw, un-classified blocks in corrected reading order.
        Raises ExtractionError if the file is encrypted or unreadable.
        """
        doc = self._open(file_path)
        try:
            blocks: list[RawBlock] = []
            for page_index in range(doc.page_count):
                page = doc[page_index]
                blocks.extend(
                    self._extract_page(page, page_index + 1, document_id, storage)
                )
        finally:
            doc.close()

        table_blocks = self._extract_tables(file_path)
        if table_blocks:
            blocks = self._merge_tables(blocks, table_blocks)

        return self._reorder_reading_order(blocks)

    def is_scanned(self, file_path: str) -> bool:
        """True if pages contain ~no extractable text -> needs OCR."""
        doc = self._open(file_path)
        try:
            if doc.page_count == 0:
                return False
            total_chars = sum(len(page.get_text("text").strip()) for page in doc)
            return (total_chars / doc.page_count) < _SCANNED_CHARS_PER_PAGE_THRESHOLD
        finally:
            doc.close()

    # -- internals -----------------------------------------------------

    def _open(self, file_path: str) -> "fitz.Document":
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PDF: {exc}") from exc

        if doc.is_encrypted and not doc.authenticate(""):
            doc.close()
            raise ExtractionError("PDF is encrypted and cannot be read without a password")
        if doc.page_count == 0:
            doc.close()
            raise ExtractionError("PDF has no pages")
        return doc

    def _extract_page(
        self,
        page: "fitz.Page",
        page_number: int,
        document_id: str | None,
        storage: "FileStorage | None",
    ) -> list["RawBlock"]:
        page_blocks: list[RawBlock] = []
        raw = page.get_text("dict")

        for i, block in enumerate(raw.get("blocks", [])):
            if block.get("type") == 1:
                page_blocks.append(
                    self._image_block(block, page_number, i, document_id, storage)
                )
                continue

            text_block = self._text_block(block, page_number)
            if text_block is not None:
                page_blocks.append(text_block)

        return page_blocks

    def _image_block(self, block: dict, page_number: int, index: int, document_id, storage) -> "RawBlock":
        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        image_ref = None
        image_bytes = block.get("image")
        if image_bytes and document_id and storage is not None:
            ext = (block.get("ext") or "png").lstrip(".")
            image_ref = storage.save_output(
                document_id, f"images/p{page_number}_{index}.{ext}", image_bytes
            )
        return RawBlock(
            page=page_number,
            text="",
            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
            is_image=True,
            image_ref=image_ref,
        )

    def _text_block(self, block: dict, page_number: int) -> "RawBlock | None":
        lines_text: list[str] = []
        sizes: list[float] = []
        bold_votes = 0
        span_count = 0

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                lines_text.append(line_text)
            for span in spans:
                sizes.append(span.get("size", 0.0))
                span_count += 1
                if span.get("flags", 0) & _BOLD_FLAG_BIT:
                    bold_votes += 1

        text = "\n".join(lines_text).strip()
        if not text:
            return None

        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        return RawBlock(
            page=page_number,
            text=text,
            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
            font_size=(sum(sizes) / len(sizes)) if sizes else None,
            is_bold=span_count > 0 and (bold_votes / span_count) > 0.5,
        )

    def _extract_tables(self, file_path: str) -> list[RawBlock]:
        """Use pdfplumber to recover real TABLE blocks with cell structure."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning(
                "pdfplumber is not installed; tables will extract as plain text. "
                "Add pdfplumber to requirements to enable structured table extraction."
            )
            return []

        table_blocks: list[RawBlock] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    page_number = page_index + 1
                    try:
                        found = page.find_tables() or []
                    except Exception as exc:  # noqa: BLE001 -- never fail extract on one table
                        logger.warning("pdfplumber find_tables failed on page %s: %s", page_number, exc)
                        continue
                    for table in found:
                        try:
                            data = table.extract()
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("pdfplumber extract failed on page %s: %s", page_number, exc)
                            continue
                        if not data or not any(any(cell for cell in (row or [])) for row in data):
                            continue
                        cells = self._rows_to_cells(data)
                        if not cells:
                            continue
                        bbox = getattr(table, "bbox", None) or (0.0, 0.0, 0.0, 0.0)
                        # Flatten text for fallback / search.
                        flat = "\n".join(
                            " | ".join((c or "").strip() for c in (row or []) if c)
                            for row in data
                        )
                        table_blocks.append(
                            RawBlock(
                                page=page_number,
                                text=flat.strip(),
                                x0=float(bbox[0]),
                                y0=float(bbox[1]),
                                x1=float(bbox[2]),
                                y1=float(bbox[3]),
                                cells=cells,
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber table extraction failed: %s", exc)
            return []

        return table_blocks

    @staticmethod
    def _rows_to_cells(rows: list) -> list[RawTableCell]:
        cells: list[RawTableCell] = []
        for r_idx, row in enumerate(rows):
            if row is None:
                continue
            for c_idx, value in enumerate(row):
                text = (value or "").strip() if value is not None else ""
                cells.append(
                    RawTableCell(
                        row=r_idx,
                        col=c_idx,
                        text=text,
                        is_header=(r_idx == 0),
                    )
                )
        return cells

    def _merge_tables(
        self, text_blocks: list[RawBlock], table_blocks: list[RawBlock]
    ) -> list[RawBlock]:
        """Drop text blocks that mostly sit inside a detected table bbox."""
        if not table_blocks:
            return text_blocks

        kept: list[RawBlock] = []
        for block in text_blocks:
            if block.is_image or block.is_table:
                kept.append(block)
                continue
            if any(
                t.page == block.page and self._overlap_ratio(block, t) >= _TABLE_OVERLAP_DROP_RATIO
                for t in table_blocks
            ):
                continue
            kept.append(block)
        return kept + table_blocks

    @staticmethod
    def _overlap_ratio(inner: RawBlock, outer: RawBlock) -> float:
        ix0 = max(inner.x0, outer.x0)
        iy0 = max(inner.y0, outer.y0)
        ix1 = min(inner.x1, outer.x1)
        iy1 = min(inner.y1, outer.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area = max((inner.x1 - inner.x0) * (inner.y1 - inner.y0), 1e-6)
        return inter / area

    def _reorder_reading_order(self, blocks: list["RawBlock"]) -> list["RawBlock"]:
        by_page: dict[int, list[RawBlock]] = {}
        for b in blocks:
            by_page.setdefault(b.page, []).append(b)

        ordered: list[RawBlock] = []
        for page_number in sorted(by_page):
            for column_index, column_blocks in enumerate(self._cluster_columns(by_page[page_number])):
                for b in column_blocks:
                    b.column = column_index
                ordered.extend(sorted(column_blocks, key=lambda b: (round(b.y0, 1), b.x0)))
        return ordered

    def _cluster_columns(self, blocks: list["RawBlock"]) -> list[list["RawBlock"]]:
        if len(blocks) < 2:
            return [blocks]

        xs = sorted(b.x0 for b in blocks)
        page_width = xs[-1] - xs[0]
        if page_width <= 0:
            return [blocks]

        gaps = [(xs[i + 1] - xs[i], (xs[i] + xs[i + 1]) / 2) for i in range(len(xs) - 1)]
        widest_gap, split_x = max(gaps, key=lambda g: g[0])
        relative_position = (split_x - xs[0]) / page_width

        if widest_gap < _COLUMN_GAP_THRESHOLD_PT or not (0.25 < relative_position < 0.75):
            return [blocks]

        left = [b for b in blocks if b.x0 <= split_x]
        right = [b for b in blocks if b.x0 > split_x]
        return [left, right]
