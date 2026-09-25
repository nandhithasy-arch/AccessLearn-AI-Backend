"""
Stage 4: element classification.

Turns raw extracted blocks into typed `DocumentSection`s (heading, paragraph,
list, table, equation, etc). Font size/weight heuristics can handle most
heading/paragraph/list distinctions deterministically; the AI client is used
for the harder calls (is this a caption vs a footnote? is this decorative or
informative?).

Implementation status:
    - Deterministic pass: DONE. Font size relative to the document's median
      ("body text") size + boldness decides heading vs paragraph; a
      bullet/number-prefix majority on a multi-line block decides list vs
      paragraph; image blocks pass straight through as IMAGE (diagram/chart/
      decorative vs "plain" image needs visual judgment, not attempted here).
    - AI pass (batched, not one call per block) for genuinely ambiguous
      blocks -- pull-quote vs citation, caption vs footnote, decorative vs
      informative image -- NOT done yet. `self._ai` is wired up (lazily, so
      importing/instantiating this class doesn't require an API key) but
      unused until that pass exists; see TODO below.
"""
from __future__ import annotations

import re
import statistics
from typing import Optional

from app.core.config import get_settings
from app.models.document import BoundingBox, DocumentSection, ElementType
from app.services.ai_client import AIClient, get_ai_client
from app.services.extraction.pdf_extractor import RawBlock

CLASSIFY_SYSTEM_PROMPT = """You classify blocks of extracted document text into \
one of: heading, paragraph, list, table, image, diagram, chart, equation, \
code, caption, footnote, citation, decorative. Return a JSON array where \
each item is {"index": <int>, "type": "<one of the above>", \
"heading_level": <1-6 or null>}."""

_BULLET_RE = re.compile(r"^\s*[•‣◦▪·\u2022]\s+|^\s*[-*]\s+")
_NUMBERED_RE = re.compile(r"^\s*(?:\d{1,3}|[a-zA-Z])[\.\)]\s+")

# A block's average font size must exceed the page's median ("body text")
# size by at least this ratio to be considered a heading at all; the finer
# ratio bands below then decide heading_level 1-4. Anything not clearly
# larger (or bold-and-only-slightly-larger) than body text stays a paragraph
# -- ambiguous cases are exactly what the (not-yet-built) AI pass is for.
_HEADING_LEVEL_RATIOS = [
    (1.8, 1),
    (1.5, 2),
    (1.25, 3),
]
_BOLD_HEADING_RATIO = 1.05  # bold text only slightly larger than body -> h4


class ElementClassifier:
    def __init__(
        self,
        ai_client: Optional[AIClient] = None,
        ocr_confidence_threshold: Optional[float] = None,
    ) -> None:
        # Deliberately NOT calling get_ai_client() eagerly here: classify()
        # below is 100% heuristic and doesn't touch the network, and
        # constructing AIClient() requires ANTHROPIC_API_KEY to be set --
        # no reason to force that requirement on every ElementClassifier()
        # until the AI-assisted pass actually exists.
        self._ai_override = ai_client
        settings = get_settings()
        self._ocr_threshold = (
            ocr_confidence_threshold
            if ocr_confidence_threshold is not None
            else settings.ocr_confidence_threshold
        )

    @property
    def _ai(self) -> AIClient:
        return self._ai_override or get_ai_client()

    def classify(self, blocks: list[RawBlock]) -> list[DocumentSection]:
        """
        Deterministic heuristics only, for now (see module docstring).
        """
        body_size = self._body_font_size(blocks)

        sections: list[DocumentSection] = []
        for order, block in enumerate(blocks):
            if block.is_image:
                sections.append(
                    DocumentSection(
                        type=ElementType.IMAGE,
                        order=order,
                        page=block.page,
                        bbox=_bbox(block),
                        image_ref=block.image_ref,
                        ocr_confidence=getattr(block, "ocr_confidence", None),
                    )
                )
                continue

            if getattr(block, "is_table", False) and block.cells:
                from app.models.document import TableCell

                sections.append(
                    DocumentSection(
                        type=ElementType.TABLE,
                        order=order,
                        page=block.page,
                        bbox=_bbox(block),
                        text=block.text.strip() or None,
                        cells=[
                            TableCell(
                                row=c.row,
                                col=c.col,
                                text=c.text,
                                is_header=c.is_header,
                            )
                            for c in block.cells
                        ],
                        ocr_confidence=getattr(block, "ocr_confidence", None),
                    )
                )
                continue

            text = block.text.strip()
            if not text:
                continue

            heading_level = self._heading_level(block, body_size)
            if heading_level is not None:
                sections.append(
                    DocumentSection(
                        type=ElementType.HEADING,
                        order=order,
                        page=block.page,
                        bbox=_bbox(block),
                        text=text,
                        heading_level=heading_level,
                        ocr_confidence=getattr(block, "ocr_confidence", None),
                        needs_human_review=(
                            block.ocr_confidence is not None
                            and block.ocr_confidence < self._ocr_threshold
                        ),
                    )
                )
                continue

            list_items = self._list_items(text)
            if list_items is not None:
                first_line = text.splitlines()[0]
                sections.append(
                    DocumentSection(
                        type=ElementType.LIST,
                        order=order,
                        page=block.page,
                        bbox=_bbox(block),
                        list_items=list_items,
                        ordered=bool(_NUMBERED_RE.match(first_line)),
                        ocr_confidence=getattr(block, "ocr_confidence", None),
                        needs_human_review=(
                            block.ocr_confidence is not None
                            and block.ocr_confidence < self._ocr_threshold
                        ),
                    )
                )
                continue

            sections.append(
                DocumentSection(
                    type=ElementType.PARAGRAPH,
                    order=order,
                    page=block.page,
                    bbox=_bbox(block),
                    text=text,
                    ocr_confidence=getattr(block, "ocr_confidence", None),
                    needs_human_review=(
                        block.ocr_confidence is not None
                        and block.ocr_confidence < self._ocr_threshold
                    ),
                )
            )

        return sections

    # -- heuristics ------------------------------------------------------

    def _body_font_size(self, blocks: list[RawBlock]) -> float:
        sizes = [b.font_size for b in blocks if not b.is_image and b.font_size]
        return statistics.median(sizes) if sizes else 12.0

    def _heading_level(self, block: RawBlock, body_size: float) -> int | None:
        if not block.font_size or body_size <= 0:
            return None
        ratio = block.font_size / body_size
        for min_ratio, level in _HEADING_LEVEL_RATIOS:
            if ratio >= min_ratio:
                return level
        if block.is_bold and ratio >= _BOLD_HEADING_RATIO:
            return 4
        return None

    def _list_items(self, text: str) -> list[str] | None:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        marker_lines = [line for line in lines if _BULLET_RE.match(line) or _NUMBERED_RE.match(line)]
        if len(marker_lines) < len(lines) * 0.6:
            return None
        return [_NUMBERED_RE.sub("", _BULLET_RE.sub("", line)).strip() for line in lines]

    # -- TODO (Person 3 / Person 4) --------------------------------------
    # def _classify_ambiguous(self, blocks: list[RawBlock]) -> dict[int, DocumentSection]:
    #     """Batch every block the heuristics above left as a plain paragraph
    #     but that looks suspicious (very short, italic, small, page-edge
    #     bbox -> maybe a caption/footnote/citation) into ONE
    #     self._ai.complete_json(CLASSIFY_SYSTEM_PROMPT, ...) call, not one
    #     call per block."""


def _bbox(block: RawBlock) -> BoundingBox:
    return BoundingBox(page=block.page, x0=block.x0, y0=block.y0, x1=block.x1, y1=block.y1)
