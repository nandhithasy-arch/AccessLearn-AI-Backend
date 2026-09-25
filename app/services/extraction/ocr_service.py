"""
Stage 3: OCR for scanned PDFs / images.

Uses Tesseract via pytesseract. Returns per-word confidence and page
coordinates so low-confidence spans can be flagged for human review and
downstream classification can still place text relative to diagrams.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.services.extraction.pdf_extractor import RawBlock

logger = logging.getLogger(__name__)

# Tesseract confidence is 0-100; we normalise to 0-1.
_TESS_MIN_CONF = 0.0


@dataclass
class OCRWord:
    text: str
    confidence: float  # 0.0 - 1.0
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class OCRError(Exception):
    """Raised when the OCR engine cannot process the input at all."""


class OCRService:
    def __init__(self, confidence_threshold: Optional[float] = None) -> None:
        settings = get_settings()
        self._threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.ocr_confidence_threshold
        )

    def run(self, image_path: str, *, page: int = 1) -> list[OCRWord]:
        """OCR a single image file. Returns per-word results with confidence."""
        path = Path(image_path)
        if not path.exists():
            raise OCRError(f"Image not found: {image_path}")

        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise OCRError(
                "pytesseract and Pillow are required for OCR. "
                "Install them and ensure the Tesseract binary is on PATH."
            ) from exc

        try:
            image = Image.open(path)
        except Exception as exc:
            raise OCRError(f"Could not open image for OCR: {exc}") from exc

        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRError(
                "Tesseract binary not found. Install tesseract-ocr and ensure it is on PATH."
            ) from exc
        except Exception as exc:
            raise OCRError(f"Tesseract failed on {image_path}: {exc}") from exc

        return self._parse_tesseract_dict(data, page=page)

    def run_pdf(self, pdf_path: str, *, dpi: int = 200) -> list[OCRWord]:
        """Rasterise each PDF page and OCR it. Used for scanned PDFs."""
        try:
            import fitz
        except ImportError as exc:
            raise OCRError("PyMuPDF (fitz) is required to OCR PDF pages") from exc

        path = Path(pdf_path)
        if not path.exists():
            raise OCRError(f"PDF not found: {pdf_path}")

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise OCRError(f"Could not open PDF for OCR: {exc}") from exc

        words: list[OCRWord] = []
        try:
            if doc.page_count == 0:
                return []
            scale = dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            with tempfile.TemporaryDirectory(prefix="accesslearn_ocr_") as tmp:
                tmp_path = Path(tmp)
                for page_index in range(doc.page_count):
                    page = doc[page_index]
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    img_file = tmp_path / f"page_{page_index + 1}.png"
                    pix.save(str(img_file))
                    # Coordinates from Tesseract are in raster pixels; scale
                    # back to PDF points so they align with other extractors.
                    page_words = self.run(str(img_file), page=page_index + 1)
                    for w in page_words:
                        words.append(
                            OCRWord(
                                text=w.text,
                                confidence=w.confidence,
                                page=w.page,
                                x0=w.x0 / scale,
                                y0=w.y0 / scale,
                                x1=w.x1 / scale,
                                y1=w.y1 / scale,
                            )
                        )
        finally:
            doc.close()

        return words

    def words_to_blocks(self, words: list[OCRWord]) -> list[RawBlock]:
        """Group OCR words into line-level RawBlocks for the classifier.

        Lines are grouped by page + approximate y0. Each block carries the
        mean word confidence so issue detection can flag low-quality spans.
        """
        if not words:
            return []

        # Sort reading order: page, top-to-bottom, left-to-right.
        ordered = sorted(words, key=lambda w: (w.page, round(w.y0, 0), w.x0))
        lines: list[list[OCRWord]] = []
        current: list[OCRWord] = []
        current_page: Optional[int] = None
        current_y: Optional[float] = None
        y_tol = 6.0  # points; words within this band are the same line

        for w in ordered:
            if not w.text.strip():
                continue
            if (
                current
                and current_page == w.page
                and current_y is not None
                and abs(w.y0 - current_y) <= y_tol
            ):
                current.append(w)
            else:
                if current:
                    lines.append(current)
                current = [w]
                current_page = w.page
                current_y = w.y0
        if current:
            lines.append(current)

        blocks: list[RawBlock] = []
        for line in lines:
            text = " ".join(w.text for w in line).strip()
            if not text:
                continue
            confs = [w.confidence for w in line]
            mean_conf = sum(confs) / len(confs)
            blocks.append(
                RawBlock(
                    page=line[0].page,
                    text=text,
                    x0=min(w.x0 for w in line),
                    y0=min(w.y0 for w in line),
                    x1=max(w.x1 for w in line),
                    y1=max(w.y1 for w in line),
                    ocr_confidence=mean_conf,
                )
            )
        return blocks

    def low_confidence_spans(self, words: list[OCRWord]) -> list[OCRWord]:
        """Words below threshold -- surface as review warnings, never silent."""
        return [w for w in words if w.confidence < self._threshold]

    # -- internals ---------------------------------------------------

    @staticmethod
    def _parse_tesseract_dict(data: dict, *, page: int) -> list[OCRWord]:
        n = len(data.get("text") or [])
        words: list[OCRWord] = []
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf_raw = float(data["conf"][i])
            except (TypeError, ValueError, KeyError):
                conf_raw = -1.0
            # Tesseract uses -1 for non-word rows (e.g. level headers).
            if conf_raw < 0:
                continue
            conf = max(_TESS_MIN_CONF, min(1.0, conf_raw / 100.0))
            try:
                x0 = float(data["left"][i])
                y0 = float(data["top"][i])
                w = float(data["width"][i])
                h = float(data["height"][i])
            except (TypeError, ValueError, KeyError):
                continue
            words.append(
                OCRWord(
                    text=text,
                    confidence=conf,
                    page=page,
                    x0=x0,
                    y0=y0,
                    x1=x0 + w,
                    y1=y0 + h,
                )
            )
        return words
