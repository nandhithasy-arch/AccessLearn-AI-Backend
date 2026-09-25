"""
Read/write functions between the Pydantic models (app/models/*.py) and the
ORM tables (app/db/models.py).

Routes should import THIS module, never `app.db.models` directly and never
hold a module-level dict. Every function takes a `Session` (from
`Depends(get_db)` in a route) so nothing here holds connection/session state
of its own -- that's what made the old `_DOCUMENTS` dict unsafe to reason
about (shared mutable state, no transaction boundaries, wiped on restart).

Each function returns/accepts the Pydantic model, not the ORM row --
callers (routes, services) should never need to know the storage shape.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    DocumentRecord,
    SemanticRepresentationRecord,
    TransformedVersionRecord,
    ValidationResultRecord,
)
from app.models.document import StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import TargetFormat, TransformedVersion
from app.models.validation import ValidationResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def create_document(db: Session, doc: StructuredDocument, source_path: str | None = None) -> StructuredDocument:
    now = _now()
    row = DocumentRecord(
        document_id=doc.document_id,
        status=doc.status.value,
        source_filename=doc.source_filename,
        source_format=doc.source_format,
        source_path=source_path,
        language=doc.language,
        created_at=now,
        updated_at=now,
        data=doc.model_dump(mode="json"),
    )
    db.add(row)
    db.commit()
    return doc


def get_document(db: Session, document_id: str) -> StructuredDocument | None:
    row = db.get(DocumentRecord, document_id)
    if row is None:
        return None
    return StructuredDocument.model_validate(row.data)


def get_document_source_path(db: Session, document_id: str) -> str | None:
    """The path FileStorage wrote the originally-uploaded file to (see
    `source_path` on DocumentRecord). Not part of the StructuredDocument
    Pydantic model itself -- it's an internal storage detail, not something
    the API response should expose -- so this reads the ORM row directly
    rather than going through `get_document`."""
    row = db.get(DocumentRecord, document_id)
    return row.source_path if row is not None else None


def update_document(db: Session, doc: StructuredDocument) -> StructuredDocument:
    """Full replace of a document's stored state (sections, issues, score,
    status, ...). Used after each pipeline stage (extract -> classify ->
    detect issues -> score) so the DB always reflects the latest
    StructuredDocument, the same object the routes hand back to the client.
    """
    row = db.get(DocumentRecord, doc.document_id)
    if row is None:
        raise ValueError(f"No document {doc.document_id} to update")
    row.status = doc.status.value
    row.language = doc.language
    row.updated_at = _now()
    row.data = doc.model_dump(mode="json")
    db.commit()
    return doc


def list_documents(db: Session, limit: int = 50) -> list[StructuredDocument]:
    rows = db.scalars(
        select(DocumentRecord).order_by(DocumentRecord.created_at.desc()).limit(limit)
    ).all()
    return [StructuredDocument.model_validate(r.data) for r in rows]


def delete_document(db: Session, document_id: str) -> bool:
    row = db.get(DocumentRecord, document_id)
    if row is None:
        return False
    db.delete(row)  # cascades to semantic_representations / transformed_versions / validation_results
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Semantic representation (one per document)
# ---------------------------------------------------------------------------

def save_semantic_representation(db: Session, rep: SemanticRepresentation) -> SemanticRepresentation:
    now = _now()
    row = db.get(SemanticRepresentationRecord, rep.document_id)
    if row is None:
        row = SemanticRepresentationRecord(
            document_id=rep.document_id,
            created_at=now,
            updated_at=now,
            data=rep.model_dump(mode="json"),
        )
        db.add(row)
    else:
        row.updated_at = now
        row.data = rep.model_dump(mode="json")
    db.commit()
    return rep


def get_semantic_representation(db: Session, document_id: str) -> SemanticRepresentation | None:
    row = db.get(SemanticRepresentationRecord, document_id)
    if row is None:
        return None
    return SemanticRepresentation.model_validate(row.data)


# ---------------------------------------------------------------------------
# Transformed versions (many per document)
# ---------------------------------------------------------------------------

def add_transformed_versions(db: Session, versions: list[TransformedVersion]) -> list[TransformedVersion]:
    now = _now()
    for v in versions:
        row = TransformedVersionRecord(
            id=v.id,
            document_id=v.document_id,
            format=v.format.value,
            learner_profile=v.learner_profile.value if v.learner_profile else None,
            created_at=now,
            data=v.model_dump(mode="json"),
        )
        db.add(row)
    db.commit()
    return versions


def list_transformed_versions(
    db: Session, document_id: str, format: TargetFormat | None = None
) -> list[TransformedVersion]:
    stmt = select(TransformedVersionRecord).where(TransformedVersionRecord.document_id == document_id)
    if format is not None:
        stmt = stmt.where(TransformedVersionRecord.format == format.value)
    stmt = stmt.order_by(TransformedVersionRecord.created_at.asc())
    rows = db.scalars(stmt).all()
    return [TransformedVersion.model_validate(r.data) for r in rows]


def get_transformed_version(db: Session, version_id: str) -> TransformedVersion | None:
    row = db.get(TransformedVersionRecord, version_id)
    if row is None:
        return None
    return TransformedVersion.model_validate(row.data)


# ---------------------------------------------------------------------------
# Validation results (many per document, one-or-more per transformed version)
# ---------------------------------------------------------------------------

def add_validation_results(db: Session, results: list[ValidationResult]) -> list[ValidationResult]:
    now = _now()
    for r in results:
        row = ValidationResultRecord(
            id=r.id,
            document_id=r.document_id,
            transformed_version_id=r.transformed_version_id,
            overall_consistency_score=r.overall_consistency_score,
            requires_human_review=r.requires_human_review,
            created_at=now,
            data=r.model_dump(mode="json"),
        )
        db.add(row)
    db.commit()
    return results


def list_validation_results(db: Session, document_id: str) -> list[ValidationResult]:
    stmt = (
        select(ValidationResultRecord)
        .where(ValidationResultRecord.document_id == document_id)
        .order_by(ValidationResultRecord.created_at.asc())
    )
    rows = db.scalars(stmt).all()
    return [ValidationResult.model_validate(r.data) for r in rows]


def get_validation_result(db: Session, validation_id: str) -> ValidationResult | None:
    row = db.get(ValidationResultRecord, validation_id)
    if row is None:
        return None
    return ValidationResult.model_validate(row.data)


def list_validations_needing_review(
    db: Session, *, document_id: str | None = None
) -> list[ValidationResult]:
    """Validation rows flagged requires_human_review that are still pending."""
    stmt = select(ValidationResultRecord).where(
        ValidationResultRecord.requires_human_review.is_(True)
    )
    if document_id:
        stmt = stmt.where(ValidationResultRecord.document_id == document_id)
    stmt = stmt.order_by(ValidationResultRecord.created_at.asc())
    from app.models.validation import ReviewerDecision

    results: list[ValidationResult] = []
    for row in db.scalars(stmt).all():
        try:
            vr = ValidationResult.model_validate(row.data)
        except Exception:
            continue
        decision = vr.reviewer_decision
        if decision is None or decision == ReviewerDecision.PENDING:
            results.append(vr)
    return results


def apply_review_decision(
    db: Session, validation_id: str, action: "ReviewAction"
) -> ValidationResult | None:
    """Record reviewer decision on the validation row (audit trail)."""
    from datetime import datetime, timezone

    from app.models.validation import ReviewerDecision

    row = db.get(ValidationResultRecord, validation_id)
    if row is None:
        return None
    vr = ValidationResult.model_validate(row.data)
    vr.reviewer_decision = action.decision
    vr.reviewer_name = action.reviewer_name
    vr.reviewer_notes = action.notes
    vr.warning_resolutions = list(action.warning_resolutions or [])
    vr.reviewed_at = datetime.now(timezone.utc)
    # Accepted / overridden clear the queue flag; rejected keeps it visible
    # until a new transform+validate cycle replaces the row.
    if action.decision in (ReviewerDecision.ACCEPTED, ReviewerDecision.OVERRIDDEN):
        vr.requires_human_review = False
        row.requires_human_review = False
    row.data = vr.model_dump(mode="json")
    db.commit()
    return vr
