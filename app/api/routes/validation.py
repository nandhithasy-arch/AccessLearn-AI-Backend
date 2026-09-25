from __future__ import annotations

import json
import zipfile
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import repository
from app.db.database import get_db
from app.models.validation import ReviewAction, ValidationResult
from app.services.validation.semantic_validator import SemanticValidator

router = APIRouter(prefix="/documents", tags=["validation"])
_validator = SemanticValidator()


@router.post("/{document_id}/validate", response_model=list[ValidationResult])
async def validate_document(document_id: str, db: Session = Depends(get_db)) -> list[ValidationResult]:
    """
    Runs SemanticValidator against every TransformedVersion currently stored
    for this document. Call after /transform.
    """
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    versions = repository.list_transformed_versions(db, document_id)
    if not versions:
        raise HTTPException(
            status_code=409,
            detail="No transformed versions to validate -- call /transform first",
        )

    semantic_representation = repository.get_semantic_representation(db, document_id)
    if semantic_representation is None:
        raise HTTPException(
            status_code=409,
            detail="No semantic representation cached -- call /analyze first",
        )

    results = [
        _validator.validate(semantic_representation, version) for version in versions
    ]
    results = repository.add_validation_results(db, results)
    return results


@router.get("/{document_id}/validations", response_model=list[ValidationResult])
async def list_validations(document_id: str, db: Session = Depends(get_db)) -> list[ValidationResult]:
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return repository.list_validation_results(db, document_id)


@router.get("/{document_id}/validations/{validation_id}", response_model=ValidationResult)
async def get_validation(
    document_id: str, validation_id: str, db: Session = Depends(get_db)
) -> ValidationResult:
    vr = repository.get_validation_result(db, validation_id)
    if not vr or vr.document_id != document_id:
        raise HTTPException(status_code=404, detail="Validation result not found")
    return vr


@router.post(
    "/{document_id}/validations/{validation_id}/review",
    response_model=ValidationResult,
)
async def review_validation(
    document_id: str,
    validation_id: str,
    action: ReviewAction,
    db: Session = Depends(get_db),
) -> ValidationResult:
    """
    Record a human review decision on a validation result.

    Audit trail: original document → transformation → validation → this decision.
    Accepting or overriding clears requires_human_review; rejecting keeps the
    item flagged until a new transform/validate cycle.
    """
    vr = repository.get_validation_result(db, validation_id)
    if not vr or vr.document_id != document_id:
        raise HTTPException(status_code=404, detail="Validation result not found")

    updated = repository.apply_review_decision(db, validation_id, action)
    if updated is None:
        raise HTTPException(status_code=404, detail="Validation result not found")
    return updated


@router.get("/{document_id}/export")
async def export_document(
    document_id: str, format: str = "zip", db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Bundles original + all transformed versions + validation reports
    (including reviewer decisions) into a downloadable package.
    """
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if format != "zip":
        raise HTTPException(status_code=400, detail="Only format=zip is supported right now")

    versions = repository.list_transformed_versions(db, document_id)
    validations = repository.list_validation_results(db, document_id)

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("document.json", doc.model_dump_json(indent=2))
        for v in versions:
            zf.writestr(f"versions/{v.format.value}_{v.id}.json", v.model_dump_json(indent=2))
        for r in validations:
            zf.writestr(f"validation/{r.id}.json", r.model_dump_json(indent=2))
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "document_id": document_id,
                    "version_count": len(versions),
                    "validation_count": len(validations),
                    "pending_review_count": sum(
                        1
                        for r in validations
                        if r.requires_human_review
                        and str(getattr(r.reviewer_decision, "value", r.reviewer_decision))
                        == "pending"
                    ),
                },
                indent=2,
            ),
        )
        # Explicit audit trail file for reviewers / compliance export.
        trail = []
        for r in validations:
            trail.append(
                {
                    "validation_id": r.id,
                    "transformed_version_id": r.transformed_version_id,
                    "score": r.overall_consistency_score,
                    "requires_human_review": r.requires_human_review,
                    "reviewer_decision": getattr(
                        r.reviewer_decision, "value", r.reviewer_decision
                    ),
                    "reviewer_name": r.reviewer_name,
                    "reviewer_notes": r.reviewer_notes,
                    "warning_resolutions": [
                        wr.model_dump() for wr in (r.warning_resolutions or [])
                    ],
                    "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                    "warnings": [w.model_dump() for w in r.warnings],
                }
            )
        zf.writestr("audit_trail.json", json.dumps(trail, indent=2))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_export.zip"'},
    )
