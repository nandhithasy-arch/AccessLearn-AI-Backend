from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import repository
from app.db.database import get_db
from app.models.document import DocumentStatus, ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import (
    LearnerProfile,
    ReadingLevel,
    TargetFormat,
    TransformationRequest,
    TransformedVersion,
)
from app.services.transformation.alt_text_generator import AltTextGenerator
from app.services.transformation.audio_script import AudioScriptGenerator
from app.services.transformation.captions import CaptionsGenerator
from app.services.transformation.math_spoken import MathSpokenGenerator
from app.services.transformation.screen_reader_html import ScreenReaderHTMLGenerator
from app.services.transformation.simplifier import Simplifier
from app.services.transformation.summary import SummaryGenerator
from app.services.transformation.translator import Translator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["transformation"])

# Lazy AI-backed services so a missing ANTHROPIC_API_KEY doesn't break import.
_simplifier: Simplifier | None = None
_audio: AudioScriptGenerator | None = None
_captions: CaptionsGenerator | None = None
_math: MathSpokenGenerator | None = None
_translator: Translator | None = None
_summary: SummaryGenerator | None = None
_alt_text: AltTextGenerator | None = None
_screen_reader = ScreenReaderHTMLGenerator()

_IMPLEMENTED_FORMATS = {
    TargetFormat.SIMPLIFIED_TEXT,
    TargetFormat.SCREEN_READER_HTML,
    TargetFormat.AUDIO_SCRIPT,
    TargetFormat.CAPTIONS,
    TargetFormat.MATH_SPOKEN,
    TargetFormat.TRANSLATION,
    TargetFormat.GLOSSARY,
    TargetFormat.SUMMARY,
    TargetFormat.DYSLEXIA_FRIENDLY,  # reading-level elementary via Simplifier
    TargetFormat.ALT_TEXT,  # Stage 5: vision-based alt text for image sections
}

_IMAGE_TYPES = {ElementType.IMAGE, ElementType.DIAGRAM, ElementType.CHART}

# When a learner profile is set and target_formats is empty, fill defaults
# so the backend actually adapts outputs to visual / reading / hearing /
# language / cognitive needs (priority #7).
_PROFILE_DEFAULTS: dict[LearnerProfile, tuple[list[TargetFormat], ReadingLevel]] = {
    LearnerProfile.VISUAL_ACCESSIBILITY: (
        [
            TargetFormat.ALT_TEXT,
            TargetFormat.SCREEN_READER_HTML,
            TargetFormat.MATH_SPOKEN,
            TargetFormat.AUDIO_SCRIPT,
        ],
        ReadingLevel.UNCHANGED,
    ),
    LearnerProfile.READING_COMPLEXITY: (
        [
            TargetFormat.SIMPLIFIED_TEXT,
            TargetFormat.GLOSSARY,
            TargetFormat.SUMMARY,
        ],
        ReadingLevel.MIDDLE_SCHOOL,
    ),
    LearnerProfile.HEARING_ACCESSIBILITY: (
        [
            TargetFormat.CAPTIONS,
            TargetFormat.AUDIO_SCRIPT,
            TargetFormat.SCREEN_READER_HTML,
        ],
        ReadingLevel.UNCHANGED,
    ),
    LearnerProfile.LANGUAGE_ACCESSIBILITY: (
        [
            TargetFormat.TRANSLATION,
            TargetFormat.GLOSSARY,
            TargetFormat.SIMPLIFIED_TEXT,
        ],
        ReadingLevel.MIDDLE_SCHOOL,
    ),
    LearnerProfile.COGNITIVE_ACCESSIBILITY: (
        [
            TargetFormat.SIMPLIFIED_TEXT,
            TargetFormat.SUMMARY,
            TargetFormat.GLOSSARY,
            TargetFormat.DYSLEXIA_FRIENDLY,
        ],
        ReadingLevel.ELEMENTARY,
    ),
    LearnerProfile.DYSLEXIA_FRIENDLY: (
        [
            TargetFormat.DYSLEXIA_FRIENDLY,
            TargetFormat.AUDIO_SCRIPT,
            TargetFormat.GLOSSARY,
        ],
        ReadingLevel.ELEMENTARY,
    ),
}


def _apply_profile_defaults(request: TransformationRequest) -> TransformationRequest:
    """Fill target_formats / reading_level from learner_profile when needed."""
    if request.learner_profile is None:
        return request
    defaults = _PROFILE_DEFAULTS.get(request.learner_profile)
    if defaults is None:
        return request
    default_formats, default_level = defaults
    if not request.target_formats:
        request.target_formats = list(default_formats)
    # Only override reading level when the client left the generic default
    # and the profile implies a different one.
    if (
        request.reading_level == ReadingLevel.MIDDLE_SCHOOL
        and default_level != ReadingLevel.MIDDLE_SCHOOL
    ):
        request.reading_level = default_level
    return request


def _get_simplifier() -> Simplifier:
    global _simplifier
    if _simplifier is None:
        _simplifier = Simplifier()
    return _simplifier


def _get_audio() -> AudioScriptGenerator:
    global _audio
    if _audio is None:
        _audio = AudioScriptGenerator()
    return _audio


def _get_captions() -> CaptionsGenerator:
    global _captions
    if _captions is None:
        _captions = CaptionsGenerator()
    return _captions


def _get_math() -> MathSpokenGenerator:
    global _math
    if _math is None:
        _math = MathSpokenGenerator()
    return _math


def _get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def _get_summary() -> SummaryGenerator:
    global _summary
    if _summary is None:
        _summary = SummaryGenerator()
    return _summary


def _get_alt_text() -> AltTextGenerator:
    global _alt_text
    if _alt_text is None:
        _alt_text = AltTextGenerator()
    return _alt_text


def _generate_alt_text_version(doc: StructuredDocument) -> TransformedVersion:
    """Run vision alt-text generation over every image/diagram/chart section.

    Returns a TransformedVersion whose content is a JSON array of
    {section_id, alt_text, description, needs_human_review}. Also mutates
    the in-memory sections so subsequent screen-reader HTML / export can
    pick up the filled alt_text when the caller persists the document.
    """
    generator = _get_alt_text()
    image_sections = [s for s in doc.sections if s.type in _IMAGE_TYPES]
    if not image_sections:
        return TransformedVersion(
            document_id=doc.document_id,
            format=TargetFormat.ALT_TEXT,
            content=json.dumps([], indent=2),
            generation_confidence=1.0,
            what_changed="No image/diagram/chart sections found.",
        )

    # Build surrounding text once for context (cheap, helps the model).
    surrounding = "\n".join(
        (s.text or "")
        for s in sorted(doc.sections, key=lambda x: x.order)
        if s.type in (ElementType.HEADING, ElementType.PARAGRAPH, ElementType.CAPTION)
        and s.text
    )

    results: list[dict] = []
    confidences: list[float] = []
    generated = 0
    for section in image_sections:
        image_bytes = b""
        if section.image_ref:
            path = Path(section.image_ref)
            if path.exists():
                try:
                    image_bytes = path.read_bytes()
                except OSError as exc:
                    logger.warning(
                        "Could not read image %s for alt-text: %s",
                        section.image_ref,
                        exc,
                    )

        updated = generator.generate(section, image_bytes, surrounding)
        # generator mutates and returns the same section instance
        entry = {
            "section_id": updated.id,
            "type": updated.type.value if hasattr(updated.type, "value") else str(updated.type),
            "alt_text": updated.alt_text or "",
            "description": updated.description,
            "needs_human_review": bool(updated.needs_human_review),
            "image_ref": updated.image_ref,
        }
        results.append(entry)
        if updated.alt_text or updated.description:
            generated += 1
        if updated.ocr_confidence is not None:
            confidences.append(float(updated.ocr_confidence))

    mean_conf = sum(confidences) / len(confidences) if confidences else (
        0.85 if generated else 0.5
    )
    return TransformedVersion(
        document_id=doc.document_id,
        format=TargetFormat.ALT_TEXT,
        content=json.dumps(results, indent=2),
        generation_confidence=round(mean_conf, 2),
        what_changed=(
            f"Generated alt text / long descriptions for {generated} of "
            f"{len(image_sections)} image section(s)."
        ),
    )


def _run_transform(
    target_format: TargetFormat,
    doc: StructuredDocument,
    semantics: SemanticRepresentation,
    request: TransformationRequest,
) -> TransformedVersion:
    if target_format == TargetFormat.SIMPLIFIED_TEXT:
        version = _get_simplifier().simplify(doc, semantics, request.reading_level)
    elif target_format == TargetFormat.DYSLEXIA_FRIENDLY:
        # Spec maps dyslexia-friendly to simpler structure/vocabulary; reuse
        # elementary reading level rather than a separate model path.
        from app.models.transformation import ReadingLevel

        version = _get_simplifier().simplify(doc, semantics, ReadingLevel.ELEMENTARY)
        version.format = TargetFormat.DYSLEXIA_FRIENDLY
        version.what_changed = (
            (version.what_changed or "")
            + " Dyslexia-friendly profile: elementary reading level applied."
        ).strip()
    elif target_format == TargetFormat.SCREEN_READER_HTML:
        html = _screen_reader.generate(doc)
        version = TransformedVersion(
            document_id=doc.document_id,
            format=TargetFormat.SCREEN_READER_HTML,
            content=html,
            generation_confidence=1.0,
            what_changed=(
                "Deterministic screen-reader HTML from classified sections "
                "(headings, landmarks, table structure, alt text)."
            ),
        )
    elif target_format == TargetFormat.AUDIO_SCRIPT:
        version = _get_audio().generate(doc, semantics)
    elif target_format == TargetFormat.CAPTIONS:
        version = _get_captions().generate(doc, semantics)
    elif target_format == TargetFormat.MATH_SPOKEN:
        version = _get_math().generate(doc, semantics)
    elif target_format == TargetFormat.TRANSLATION:
        version = _get_translator().translate(
            doc, semantics, target_language=request.language or "es"
        )
    elif target_format == TargetFormat.GLOSSARY:
        version = _get_translator().glossary_only(
            doc, semantics, target_language=request.language or "en"
        )
    elif target_format == TargetFormat.SUMMARY:
        version = _get_summary().generate(doc, semantics)
    elif target_format == TargetFormat.ALT_TEXT:
        version = _generate_alt_text_version(doc)
    else:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Transformation for target_format='{target_format.value}' is not "
                f"implemented yet. Available: "
                f"{sorted(f.value for f in _IMPLEMENTED_FORMATS)}."
            ),
        )

    if request.learner_profile is not None and version.learner_profile is None:
        version.learner_profile = request.learner_profile
    return version


@router.post("/{document_id}/transform", response_model=list[TransformedVersion])
async def transform_document(
    document_id: str, request: TransformationRequest, db: Session = Depends(get_db)
) -> list[TransformedVersion]:
    """
    Fans out to transformation services for each requested target_format,
    using the cached SemanticRepresentation as shared context.

    Implemented (Stage 5 accessibility expansion):
      simplified_text, dyslexia_friendly, screen_reader_html, audio_script,
      captions, math_spoken, translation, glossary, summary, alt_text.
    """
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status not in (
        DocumentStatus.ANALYZED,
        DocumentStatus.TRANSFORMING,
        DocumentStatus.READY,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Document is '{doc.status.value}' -- call /analyze before /transform",
        )

    semantic_representation = repository.get_semantic_representation(db, document_id)
    if semantic_representation is None:
        raise HTTPException(
            status_code=409,
            detail="No semantic representation cached -- call /analyze first",
        )

    # Profile-based adaptation: if the client selected a learner profile but
    # left target_formats empty, fill the recommended format set + reading level.
    request = _apply_profile_defaults(request)

    if not request.target_formats:
        raise HTTPException(
            status_code=400,
            detail=(
                "target_formats must contain at least one format "
                "(or set learner_profile to auto-select defaults)"
            ),
        )

    unsupported = [f for f in request.target_formats if f not in _IMPLEMENTED_FORMATS]
    if unsupported and not any(f in _IMPLEMENTED_FORMATS for f in request.target_formats):
        raise HTTPException(
            status_code=501,
            detail=(
                f"None of the requested formats are implemented yet: "
                f"{[f.value for f in unsupported]}. Available: "
                f"{sorted(f.value for f in _IMPLEMENTED_FORMATS)}."
            ),
        )

    doc.status = DocumentStatus.TRANSFORMING
    repository.update_document(db, doc)

    versions: list[TransformedVersion] = []
    errors: list[str] = []
    for target_format in request.target_formats:
        if target_format not in _IMPLEMENTED_FORMATS:
            errors.append(f"{target_format.value}: not implemented")
            continue
        try:
            versions.append(
                _run_transform(target_format, doc, semantic_representation, request)
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Transform %s failed for document %s: %s",
                target_format.value,
                document_id,
                exc,
            )
            errors.append(f"{target_format.value}: {exc}")

    if not versions:
        doc.status = DocumentStatus.ANALYZED
        repository.update_document(db, doc)
        raise HTTPException(
            status_code=500,
            detail=(
                "No transformed versions were produced. "
                + ("; ".join(errors) if errors else "Unknown failure.")
            ),
        )

    versions = repository.add_transformed_versions(db, versions)
    doc.status = DocumentStatus.READY
    repository.update_document(db, doc)
    return versions


@router.get("/{document_id}/versions", response_model=list[TransformedVersion])
async def list_versions(
    document_id: str, format: TargetFormat | None = None, db: Session = Depends(get_db)
) -> list[TransformedVersion]:
    doc = repository.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return repository.list_transformed_versions(db, document_id, format=format)
