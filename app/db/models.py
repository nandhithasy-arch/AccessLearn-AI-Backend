"""
SQLAlchemy ORM tables.

Each row stores its full Pydantic model as a JSON blob in `data` (source of
truth), plus a handful of normalized columns that routes/repository.py
actually filter or sort on. When you extend a Pydantic model in
app/models/*.py, you generally do NOT need to touch this file -- the new
field just shows up inside `data`. Only add a real column if something
needs to query on it at the DB level (e.g. `WHERE status = 'ready'`).

This mirrors the *shape* of accesslearn-data-model/database/schema.sql
(same entities: documents, elements-as-part-of-documents, transformed
versions, validation results) but collapses each entity's sub-fields into
JSON instead of fully normalizing every column -- see the note in
database.py for why.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, index=True, nullable=False)
    source_filename: Mapped[str] = mapped_column(String, nullable=False)
    source_format: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Full StructuredDocument.model_dump(mode="json") -- sections,
    # accessibility_issues, accessibility_score, everything.
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class SemanticRepresentationRecord(Base):
    __tablename__ = "semantic_representations"

    # One canonical representation per document (spec section 7: "run once,
    # cache it, never regenerate per-transformation"), so document_id is the
    # primary key, not just a foreign key.
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.document_id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Full SemanticRepresentation.model_dump(mode="json").
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class TransformedVersionRecord(Base):
    __tablename__ = "transformed_versions"
    __table_args__ = (Index("idx_transformed_versions_document_format", "document_id", "format"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    format: Mapped[str] = mapped_column(String, nullable=False)
    learner_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)

    # Full TransformedVersion.model_dump(mode="json").
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class ValidationResultRecord(Base):
    __tablename__ = "validation_results"
    __table_args__ = (Index("idx_validation_results_document", "document_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    transformed_version_id: Mapped[str] = mapped_column(
        String, ForeignKey("transformed_versions.id", ondelete="CASCADE"), nullable=False
    )
    overall_consistency_score: Mapped[float] = mapped_column(nullable=False)
    requires_human_review: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)

    # Full ValidationResult.model_dump(mode="json").
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
