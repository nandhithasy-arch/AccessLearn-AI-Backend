from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import repository
from app.db.database import get_db
from app.models.document import AccessibilityIssue, DocumentStatus, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.services.analysis.classifier import ElementClassifier
from app.services.analysis.issue_detector import IssueDetector
from app.services.analysis.semantic_extractor import SemanticExtractor
from app.services.extraction.ocr_service import OCRError, OCRService
from app.services.extraction.office_extractor import OfficeExtractor
from app.services.extraction.pdf_extractor import ExtractionError, PDFExtractor
from app.storage.file_storage import FileStorage

router = APIRouter(prefix="/documents", tags=["documents"])
_storage = FileStorage()
_pdf_extractor = PDFExtractor()
_office_extractor = OfficeExtractor()
_ocr_service = OCRService()
_classifier = ElementClassifier()
_issue_detector = IssueDetector()

# Formats with a working extraction path on /analyze.
_EXTRACTABLE_FORMATS = {"pdf", "docx", "pptx", "image", "text", "markdown"}


def _extract_blocks(source_format: str, source_path: str, document_id: str):
    """Dispatch to the correct extractor for the uploaded source format."""
    if source_format == "pdf":
        if _pdf_extractor.is_scanned(source_path):
            ocr_words = _ocr_service.run_pdf(source_path)
            return _ocr_service.words_to_blocks(ocr_words)
        return _pdf_extractor.extract(
            source_path, document_id=document_id, storage=_storage
        )

    if source_format == "image":
        # Single-image upload: OCR the file directly.
        ocr_words = _ocr_service.run(source_path, page=1)
        return _ocr_service.words_to_blocks(ocr_words)

    if source_format == "docx":
        return _office_extractor.extract_docx(source_path)

    if source_format == "pptx":
        return _office_extractor.extract_pptx(source_path)

    if source_format in ("text", "markdown"):
        return _office_extractor.extract_text(source_path)

    raise ExtractionError(f"No extractor for source_format={source_format}")

# Lazy: SemanticExtractor() pulls AIClient which requires ANTHROPIC_API_KEY.
# Building it at import time would take down every /documents route when the
# key isn't set yet; construct on first /analyze instead.
_semantic_extractor: SemanticExtractor | None = None


def _get_semantic_extractor() -> SemanticExtractor:
    global _semantic_extractor
    if _semantic_extractor is None:
        _semantic_extractor = SemanticExtractor()
    return _semantic_extractor

# Stage 1 input validation (spec section 6): extension -> source_format.
# MVP prioritizes pdf/image/text/docx per the spec; audio/video accepted as
# input but the extraction pipeline for them is a later-phase TODO.
_EXTENSION_TO_FORMAT = {
    "pdf": "pdf",
    "docx": "docx",
    "pptx": "pptx",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "webp": "image",
    "tiff": "image",
    "txt": "text",
    "md": "markdown",
    "html": "html",
    "htm": "html",
    "mp3": "audio",
    "wav": "audio",
    "mp4": "video",
    "mov": "video",
}


@router.post("/upload", response_model=StructuredDocument, status_code=201)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> StructuredDocument:
    """
    Stage 1: input validation (file type / size / non-empty), then persist
    the raw file to storage and create a StructuredDocument row with
    status=UPLOADED. Extraction happens in /analyze, not here, so this stays
    fast. Now backed by `documents` in the DB instead of an in-memory dict --
    it survives a restart and multiple workers can share it.
    """
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower().lstrip(".")
    source_format = _EXTENSION_TO_FORMAT.get(extension)
    if source_format is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{extension}'. Supported: {sorted(set(_EXTENSION_TO_FORMAT.values()))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")

    stored_path = _storage.save_upload(filename, content)

    doc = StructuredDocument(
        source_filename=filename,
        source_format=source_format,
        status=DocumentStatus.UPLOADED,
    )
    return repository.create_document(db, doc, source_path=stored_path)


@router.get("", response_model=list[StructuredDocument])
async def list_documents(limit: int = 50, db: Session = Depends(get_db)) -> list[StructuredDocument]:
    """Powers the Dashboard screen (spec section 11): recent documents + status."""
    return repository.list_documents(db, limit=limit)


@router.get("/{document_id}", response_model=StructuredDocument)
async def get_document(document_id: str, db: Session = Depends(get_db)) -> StructuredDocument:
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/{document_id}/analyze", response_model=StructuredDocument)
async def analyze_document(
    document_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
) -> StructuredDocument:
    """
    Runs stages 2-4 (extraction -> OCR if needed -> classification -> issue
    detection/scoring) as a background job in production; for the hackathon
    MVP this runs synchronously for small documents.

    Wired for real now, for the PDF extraction path only (deliberately one
    path end-to-end before spreading to docx/pptx/images -- see STATUS.md):
    `pdf_extractor` (PyMuPDF, deterministic) -> `classifier` (font-size /
    boldness / list-marker heuristics, deterministic, no AI call) ->
    `issue_detector` (deterministic). The result is persisted via
    `repository.update_document` and read back by GET /documents/{id} --
    that DB round-trip is the thing most likely to reveal a schema problem
    (a field that doesn't serialize cleanly, an id that collides), which is
    why this stage was wired up before touching anything AI-dependent.

    Stage 5 (semantic extraction) runs after classification/scoring and
    caches the canonical SemanticRepresentation via the repository. If the
    AI call fails, extract() returns an empty representation with warnings
    so /analyze still succeeds; /transform and /validate can then see the
    (possibly empty) cached rep instead of 409ing forever.
    """
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Idempotent by default: already-analyzed docs are returned as-is so
    # revisiting /processing does not error. Use ?force=true to re-extract.
    if doc.status not in (DocumentStatus.UPLOADED, DocumentStatus.FAILED) and not force:
        return doc

    if doc.source_format not in _EXTRACTABLE_FORMATS:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Extraction for source_format='{doc.source_format}' not implemented yet. "
                f"Supported: {sorted(_EXTRACTABLE_FORMATS)}"
            ),
        )

    source_path = repository.get_document_source_path(db, document_id)
    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=500, detail="Original uploaded file is missing from storage")

    try:
        raw_blocks = _extract_blocks(doc.source_format, source_path, document_id)
    except OCRError as exc:
        doc.status = DocumentStatus.FAILED
        repository.update_document(db, doc)
        raise HTTPException(status_code=422, detail=f"OCR failed: {exc}") from exc
    except ExtractionError as exc:
        doc.status = DocumentStatus.FAILED
        repository.update_document(db, doc)
        raise HTTPException(
            status_code=422, detail=f"Could not extract {doc.source_format}: {exc}"
        ) from exc

    if not raw_blocks:
        doc.status = DocumentStatus.FAILED
        repository.update_document(db, doc)
        raise HTTPException(
            status_code=422,
            detail=f"Extraction produced no readable content from this {doc.source_format} file",
        )

    doc.sections = _classifier.classify(raw_blocks)
    doc.accessibility_issues = _issue_detector.detect(doc.sections)
    doc.accessibility_score = _issue_detector.score(doc.sections, doc.accessibility_issues)
    doc.status = DocumentStatus.ANALYZED
    doc = repository.update_document(db, doc)

    # Stage 5: canonical semantic representation (extract once, cache, never
    # regenerate per-transformation). Always persist *something* so /transform
    # and /validate stop 409ing with "no semantic representation cached".
    # extract() already degrades AIClientError / parse failure to an empty
    # rep + warnings; this outer try covers construction (missing API key)
    # and DB save failures the same way.
    try:
        rep = _get_semantic_extractor().extract(doc)
    except Exception as exc:  # noqa: BLE001 -- never let Stage 5 kill Stages 2-4
        import logging
        logging.getLogger(__name__).exception(
            "Semantic extraction failed for %s: %s", document_id, exc
        )
        rep = SemanticRepresentation(
            document_id=doc.document_id,
            warnings=[
                f"Semantic extraction failed ({exc}); empty representation "
                "cached so /transform can proceed and flag for human review."
            ],
        )
    try:
        repository.save_semantic_representation(db, rep)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception(
            "Persisting semantic representation failed for %s: %s", document_id, exc
        )
    return doc


@router.get("/{document_id}/issues", response_model=list[AccessibilityIssue])
async def get_issues(document_id: str, db: Session = Depends(get_db)) -> list[AccessibilityIssue]:
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.accessibility_issues


@router.delete("/{document_id}", status_code=200)
async def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    """Responsible-AI requirement (spec section 18): let a document and every
    derived row (semantic representation, transformed versions, validation
    results) be removed on request, not just expire silently."""
    deleted = repository.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    _storage.delete_document_files(document_id)
