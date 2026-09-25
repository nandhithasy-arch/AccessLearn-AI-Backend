"""
Audio narration script: rewrites document content as a speakable script
suitable for TTS (clear pauses, expanded abbreviations, no raw tables).
"""
from __future__ import annotations

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import TargetFormat, TransformedVersion
from app.services.ai_client import AIClient, AIClientError, get_ai_client

AUDIO_SYSTEM_PROMPT = """You convert educational document text into a narration \
script for text-to-speech. Rules:
1. Write for the ear: short sentences, natural phrasing, expand abbreviations \
   on first use (e.g. "CO2 (carbon dioxide)").
2. NEVER drop a concept, number, unit, condition, or negation from the \
   preservation checklist.
3. Mark brief pauses with [pause] between major sections; do not invent sound \
   effects or stage directions beyond that.
4. Spell out equations in spoken form when present (e.g. "six CO two plus \
   six H two O").
5. Skip pure decorative content; describe diagrams only if a description is \
   provided in the source.
Return the narration script only."""


class AudioScriptGenerator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def generate(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
    ) -> TransformedVersion:
        source = _extract_prose(document)
        if not source.strip():
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.AUDIO_SCRIPT,
                content="",
                generation_confidence=1.0,
                what_changed="No narratable text found.",
            )

        checklist = _preservation_checklist(semantics)
        user = (
            "Write an audio narration script for the following educational content.\n\n"
            f"--- MUST BE PRESERVED ---\n{checklist}\n\n"
            f"--- DOCUMENT ---\n{source}"
        )
        try:
            script = self._ai.complete(AUDIO_SYSTEM_PROMPT, user, max_tokens=4000, temperature=0.3)
        except AIClientError as exc:
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.AUDIO_SCRIPT,
                content=source,
                generation_confidence=0.0,
                what_changed=f"Audio script generation failed ({exc}); plain text returned for review.",
            )

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.AUDIO_SCRIPT,
            content=script.strip(),
            generation_confidence=0.85,
            what_changed="Converted to TTS-oriented narration script with pauses and expanded abbreviations.",
        )


_TEXT_TYPES = {
    ElementType.HEADING,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
    ElementType.EQUATION,
}


def _extract_prose(document: StructuredDocument) -> str:
    parts: list[str] = []
    for section in sorted(document.sections, key=lambda s: s.order):
        if section.type not in _TEXT_TYPES:
            if section.type in (ElementType.IMAGE, ElementType.DIAGRAM, ElementType.CHART):
                desc = section.description or section.alt_text
                if desc:
                    parts.append(f"[Image description] {desc}")
            continue
        if section.type == ElementType.HEADING and section.text:
            parts.append(section.text)
        elif section.type == ElementType.LIST and section.list_items:
            parts.append(". ".join(section.list_items))
        elif section.type == ElementType.EQUATION:
            parts.append(
                section.equation_spoken
                or section.equation_latex
                or section.equation_raw
                or section.text
                or ""
            )
        elif section.text:
            parts.append(section.text)
    return "\n\n".join(p for p in parts if p and p.strip())


def _preservation_checklist(semantics: SemanticRepresentation) -> str:
    lines: list[str] = []
    if semantics.concepts:
        lines.append("Concepts:")
        lines.extend(f"  - {c.name}: {c.definition}" for c in semantics.concepts)
    if semantics.quantities:
        lines.append("Quantities:")
        lines.extend(
            f"  - {q.value}{f' {q.unit}' if q.unit else ''} ({q.context})"
            for q in semantics.quantities
        )
    if semantics.negations:
        lines.append("Negations/exceptions:")
        lines.extend(f"  - {n.text}" for n in semantics.negations)
    return "\n".join(lines) if lines else "(none extracted)"
