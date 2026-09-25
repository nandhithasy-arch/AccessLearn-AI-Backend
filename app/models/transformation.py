from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TargetFormat(str, Enum):
    SIMPLIFIED_TEXT = "simplified_text"
    SCREEN_READER_HTML = "screen_reader_html"
    AUDIO_SCRIPT = "audio_script"
    ALT_TEXT = "alt_text"
    GLOSSARY = "glossary"
    TRANSLATION = "translation"
    CAPTIONS = "captions"
    MATH_SPOKEN = "math_spoken"
    DYSLEXIA_FRIENDLY = "dyslexia_friendly"
    HIGH_CONTRAST = "high_contrast"
    SUMMARY = "summary"


class LearnerProfile(str, Enum):
    VISUAL_ACCESSIBILITY = "visual_accessibility"
    READING_COMPLEXITY = "reading_complexity"
    HEARING_ACCESSIBILITY = "hearing_accessibility"
    LANGUAGE_ACCESSIBILITY = "language_accessibility"
    COGNITIVE_ACCESSIBILITY = "cognitive_accessibility"
    DYSLEXIA_FRIENDLY = "dyslexia_friendly"


class ReadingLevel(str, Enum):
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    UNCHANGED = "unchanged"


class TransformationRequest(BaseModel):
    document_id: str
    target_formats: list[TargetFormat]
    language: str = "en"
    reading_level: ReadingLevel = ReadingLevel.MIDDLE_SCHOOL
    learner_profile: Optional[LearnerProfile] = None


class GlossaryTerm(BaseModel):
    term: str
    definition: str
    original_term: Optional[str] = None  # for bilingual glossaries


class TransformedVersion(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    document_id: str
    format: TargetFormat
    learner_profile: Optional[LearnerProfile] = None
    content: str  # HTML, plain text, VTT/SRT, MathML, etc. depending on `format`
    glossary: list[GlossaryTerm] = []
    generation_confidence: Optional[float] = None
    what_changed: Optional[str] = None  # human-readable summary of changes made
