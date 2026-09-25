"""
Structured document representation.

This is the output of Stages 1-4 of the pipeline (validation -> extraction ->
OCR -> classification). It is intentionally NOT free text: every block is
typed and ordered so downstream transformation/validation steps can reason
about structure, not just words.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"
    CHART = "chart"
    EQUATION = "equation"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    CITATION = "citation"
    DECORATIVE = "decorative"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class BoundingBox(BaseModel):
    """Page coordinates, used to preserve reading order and detect columns."""
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


class DocumentSection(BaseModel):
    """
    One classified block of content. Almost every field is optional because
    the shape of the payload depends on `type` (a paragraph has `text`;
    a table has `cells`; a diagram has `image_ref` and no text at all until
    a description is generated).
    """
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    type: ElementType
    order: int
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None

    text: Optional[str] = None
    heading_level: Optional[int] = None  # 1-6, only for HEADING

    list_items: Optional[list[str]] = None
    ordered: Optional[bool] = None  # only for LIST

    cells: Optional[list[TableCell]] = None  # only for TABLE

    image_ref: Optional[str] = None  # storage key, for IMAGE/DIAGRAM/CHART
    description: Optional[str] = None  # generated long description
    alt_text: Optional[str] = None  # generated short alt text

    equation_raw: Optional[str] = None  # OCR'd / extracted raw form
    equation_latex: Optional[str] = None
    equation_mathml: Optional[str] = None
    equation_spoken: Optional[str] = None
    equation_confidence: Optional[float] = None

    code_language: Optional[str] = None

    ocr_confidence: Optional[float] = None
    needs_human_review: bool = False


class AccessibilityIssue(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    element_id: str  # references DocumentSection.id
    issue: str
    severity: Severity
    suggested_action: str
    confidence: Optional[float] = None


class AccessibilityScore(BaseModel):
    overall: int = Field(ge=0, le=100)
    text: int = Field(ge=0, le=100)
    visual: int = Field(ge=0, le=100)
    audio: int = Field(ge=0, le=100)
    structural: int = Field(ge=0, le=100)
    note: str = "AI-generated assessment, not a formal compliance certification."


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    TRANSFORMING = "transforming"
    READY = "ready"
    FAILED = "failed"


class StructuredDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: f"doc_{uuid4().hex[:10]}")
    document_title: Optional[str] = None
    source_filename: str
    source_format: str  # pdf | image | docx | pptx | txt | html | audio | video
    language: Optional[str] = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.utcnow)

    sections: list[DocumentSection] = []
    accessibility_issues: list[AccessibilityIssue] = []
    accessibility_score: Optional[AccessibilityScore] = None

    reading_complexity_before: Optional[float] = None  # e.g. Flesch-Kincaid grade
