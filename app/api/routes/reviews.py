"""Global human-review queue endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import repository
from app.db.database import get_db
from app.models.validation import ValidationResult

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/queue", response_model=list[ValidationResult])
async def review_queue(db: Session = Depends(get_db)) -> list[ValidationResult]:
    """All validation results that still require human review (pending)."""
    return repository.list_validations_needing_review(db)
