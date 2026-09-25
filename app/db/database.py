"""
SQLAlchemy engine/session setup.

Persistence strategy (unchanged from the original note, now actually wired
in): store StructuredDocument / SemanticRepresentation / TransformedVersion /
ValidationResult as JSON columns (sqlite/Postgres JSON type) rather than a
fully normalized schema -- far faster to build for the MVP, and the Pydantic
models already ARE the schema. We normalize only the fields other routes/
services need to query or filter on directly: document_id, status, format,
created_at. See app/db/models.py for the table definitions and
app/db/repository.py for the read/write functions routes should call.

The fully normalized schema in accesslearn-data-model/database/schema.sql
documents the target Postgres shape for when this needs to scale past a
JSON-blob-per-document; it's not what's wired up here yet.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # pool_pre_ping keeps Postgres connections healthy after idle timeouts
    pool_pre_ping=not _is_sqlite,
)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """SQLite disables foreign keys by default -- turn them on so ON DELETE
    CASCADE from documents → semantic / transformed / validation actually
    fires (Postgres enforces FKs without this)."""
    if _is_sqlite:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that don't exist yet. Called once on app startup.

    Safe to call repeatedly (create_all is a no-op for existing tables).
    For anything beyond the hackathon MVP, replace this with Alembic
    migrations -- create_all() never alters existing tables.
    """
    # Import inside the function, not at module top, so app/db/models.py
    # (which imports Base from this module) can't create a circular import
    # at import time.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
