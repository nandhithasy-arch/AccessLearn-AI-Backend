"""
Canonical semantic representation.

Extracted once from the StructuredDocument, then used as the fixed
"source of truth" that every transformation (simplified text, screen-reader
HTML, audio script, translation...) is later validated against. Nothing
here is regenerated per-transformation -- that would let each output be
validated against a different, drifting target.
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Concept(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    name: str
    definition: str
    source_element_ids: list[str] = []  # which DocumentSection(s) it came from
    is_technical_term: bool = False


class Relationship(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    source: str  # Concept.name or id
    relation: str  # e.g. causes, is_part_of, precedes, enables, contrasts_with
    target: str
    source_element_ids: list[str] = []


class Quantity(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    value: str  # kept as string: "6", "less than 5", "20-30"
    unit: Optional[str] = None
    context: str
    comparator: Optional[str] = None  # <, >, <=, >=, =, range
    source_element_ids: list[str] = []


class Negation(BaseModel):
    """A negation/exception clause that must survive transformation intact."""
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    text: str  # e.g. "only under anaerobic conditions"
    trigger_word: str  # not, never, except, unless, cannot, only
    source_element_ids: list[str] = []


class ProcedureStep(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    order: int
    text: str
    source_element_ids: list[str] = []


class SemanticRepresentation(BaseModel):
    document_id: str
    main_topic: Optional[str] = None
    learning_objectives: list[str] = []
    concepts: list[Concept] = []
    relationships: list[Relationship] = []
    quantities: list[Quantity] = []
    negations: list[Negation] = []
    procedure_steps: list[ProcedureStep] = []
    visual_only_information: list[str] = []  # info present ONLY in a diagram/image
    warnings: list[str] = []
