"""
Translation + bilingual glossary.

Translates document prose into the requested language while preserving
technical terms (with original + translated forms in the glossary) and all
quantities/negations from the semantic representation.
"""
from __future__ import annotations

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import GlossaryTerm, TargetFormat, TransformedVersion
from app.services.ai_client import AIClient, AIClientError, get_ai_client

TRANSLATE_SYSTEM_PROMPT = """You translate educational content. Rules:
1. Translate into the target language named by the user.
2. NEVER drop a concept, number, unit, condition, or negation from the \
   preservation checklist -- keep exact numeric values.
3. Keep technical/scientific terms; on first use you may add a short gloss.
4. Return ONLY the translated text (no preamble).
"""

GLOSSARY_SYSTEM_PROMPT = """You build a bilingual glossary for educational \
content. Return a JSON array of objects:
{"term": "<translated or original technical term>", \
"definition": "<short definition in the target language>", \
"original_term": "<source-language term>"}.
Include every technical concept from the checklist. Return ONLY JSON."""


class Translator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def translate(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
        target_language: str = "es",
    ) -> TransformedVersion:
        source = _extract_source(document)
        if not source.strip():
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.TRANSLATION,
                content="",
                generation_confidence=1.0,
                what_changed="No text to translate.",
            )

        if target_language.lower() in ("en", "english"):
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.TRANSLATION,
                content=source,
                glossary=_glossary_from_semantics(semantics),
                generation_confidence=1.0,
                what_changed="Target language is English -- original text returned with glossary.",
            )

        checklist = _preservation_checklist(semantics)
        user = (
            f"Target language: {target_language}\n\n"
            f"--- MUST BE PRESERVED ---\n{checklist}\n\n"
            f"--- DOCUMENT ---\n{source}"
        )
        try:
            translated = self._ai.complete(
                TRANSLATE_SYSTEM_PROMPT, user, max_tokens=4000, temperature=0.3
            )
        except AIClientError as exc:
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.TRANSLATION,
                content=source,
                glossary=_glossary_from_semantics(semantics),
                generation_confidence=0.0,
                what_changed=f"Translation failed ({exc}); original text returned for review.",
            )

        glossary = self._bilingual_glossary(semantics, target_language)

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.TRANSLATION,
            content=translated.strip(),
            glossary=glossary,
            generation_confidence=0.85,
            what_changed=f"Translated to {target_language} with bilingual technical glossary.",
        )

    def glossary_only(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
        target_language: str = "en",
    ) -> TransformedVersion:
        """Pure glossary output (TargetFormat.GLOSSARY) from semantics + optional AI fill."""
        base = _glossary_from_semantics(semantics)
        if target_language.lower() in ("en", "english") or not base:
            content = "\n".join(f"{g.term}: {g.definition}" for g in base)
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.GLOSSARY,
                content=content,
                glossary=base,
                generation_confidence=1.0,
                what_changed="Glossary built from extracted technical concepts.",
            )

        glossary = self._bilingual_glossary(semantics, target_language)
        content = "\n".join(
            f"{g.original_term or g.term} → {g.term}: {g.definition}" for g in glossary
        )
        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.GLOSSARY,
            content=content,
            glossary=glossary,
            generation_confidence=0.9,
            what_changed=f"Bilingual glossary for {target_language}.",
        )

    def _bilingual_glossary(
        self, semantics: SemanticRepresentation, target_language: str
    ) -> list[GlossaryTerm]:
        technical = [c for c in semantics.concepts if c.is_technical_term] or list(
            semantics.concepts
        )
        if not technical:
            return []

        checklist = "\n".join(f"- {c.name}: {c.definition}" for c in technical)
        user = (
            f"Target language: {target_language}\n\n"
            f"Technical concepts:\n{checklist}"
        )
        try:
            data = self._ai.complete_json(
                GLOSSARY_SYSTEM_PROMPT, user, max_tokens=2000, temperature=0.2
            )
        except AIClientError:
            return _glossary_from_semantics(semantics)

        if not isinstance(data, list):
            return _glossary_from_semantics(semantics)

        glossary: list[GlossaryTerm] = []
        for item in data:
            if not isinstance(item, dict) or not item.get("term"):
                continue
            glossary.append(
                GlossaryTerm(
                    term=str(item["term"]),
                    definition=str(item.get("definition") or ""),
                    original_term=(
                        str(item["original_term"])
                        if item.get("original_term")
                        else None
                    ),
                )
            )
        return glossary or _glossary_from_semantics(semantics)


def _extract_source(document: StructuredDocument) -> str:
    parts: list[str] = []
    for section in sorted(document.sections, key=lambda s: s.order):
        if section.type == ElementType.HEADING and section.text:
            level = section.heading_level or 1
            parts.append(f"{'#' * level} {section.text}")
        elif section.type == ElementType.LIST and section.list_items:
            parts.append("\n".join(f"- {i}" for i in section.list_items))
        elif section.type in (
            ElementType.PARAGRAPH,
            ElementType.CAPTION,
            ElementType.FOOTNOTE,
        ) and section.text:
            parts.append(section.text)
    return "\n\n".join(parts)


def _glossary_from_semantics(semantics: SemanticRepresentation) -> list[GlossaryTerm]:
    return [
        GlossaryTerm(term=c.name, definition=c.definition)
        for c in semantics.concepts
        if c.is_technical_term or c.definition
    ]


def _preservation_checklist(semantics: SemanticRepresentation) -> str:
    lines: list[str] = []
    for c in semantics.concepts:
        lines.append(f"- {c.name}: {c.definition}")
    for q in semantics.quantities:
        lines.append(f"- {q.value}{f' {q.unit}' if q.unit else ''} ({q.context})")
    for n in semantics.negations:
        lines.append(f"- negation: {n.text}")
    return "\n".join(lines) if lines else "(none)"
