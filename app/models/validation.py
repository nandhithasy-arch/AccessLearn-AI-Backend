from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.document import Severity


class PreservationWarning(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    category: str  # concept | relationship | numerical | negation | terminology | sequence | visual
    severity: Severity
    message: str
    original_reference: Optional[str] = None  # e.g. concept id, source element id
    transformed_reference: Optional[str] = None


class ConceptPreservationCheck(BaseModel):
    total: int
    preserved: int
    missing_concept_ids: list[str] = []


class RelationshipPreservationCheck(BaseModel):
    total: int
    preserved: int
    missing_relationship_ids: list[str] = []


class NumericalPreservationCheck(BaseModel):
    total: int
    preserved: int
    mismatched_quantity_ids: list[str] = []  # value/unit/comparator changed, not just missing


class NegationPreservationCheck(BaseModel):
    total: int
    preserved: int
    missing_negation_ids: list[str] = []


class SequencePreservationCheck(BaseModel):
    total_steps: int
    order_preserved: bool
    out_of_order_step_ids: list[str] = []


class ReviewerDecision(str, Enum):
    """Outcome recorded on the audit trail after a human reviews a validation."""

    ACCEPTED = "accepted"  # warnings acknowledged; output approved as-is
    OVERRIDDEN = "overridden"  # reviewer overrode some/all warnings; still approved
    REJECTED = "rejected"  # output must not be released; re-transform needed
    PENDING = "pending"


class WarningResolution(BaseModel):
    """Per-warning decision inside a review action."""

    warning_id: str
    action: str  # accept | override | dismiss
    note: Optional[str] = None


class ReviewAction(BaseModel):
    """Payload submitted by the reviewer UI."""

    decision: ReviewerDecision
    reviewer_name: str = "reviewer"
    notes: Optional[str] = None
    warning_resolutions: list[WarningResolution] = []


class ValidationResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    document_id: str
    transformed_version_id: str

    overall_consistency_score: float = Field(ge=0, le=100)

    concepts: ConceptPreservationCheck
    relationships: RelationshipPreservationCheck
    numerical: NumericalPreservationCheck
    negation: NegationPreservationCheck
    sequence: Optional[SequencePreservationCheck] = None

    warnings: list[PreservationWarning] = []
    requires_human_review: bool = False

    # Audit trail: original → transformation → validation → reviewer decision
    reviewer_decision: ReviewerDecision = ReviewerDecision.PENDING
    reviewer_name: Optional[str] = None
    reviewer_notes: Optional[str] = None
    warning_resolutions: list[WarningResolution] = []
    reviewed_at: Optional[datetime] = None
