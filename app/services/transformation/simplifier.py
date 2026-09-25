"""
Reading-complexity transformation: rewrites content at a target reading
level while preserving technical terms (with inline definitions) and all
exceptions/conditions/numbers. The rule from spec section 4:

    Simplification must reduce language difficulty, not remove academic
    meaning.

The prompt below encodes that rule explicitly rather than leaving it
implicit, since it's the single most likely failure mode for this feature.
"""
from __future__ import annotations

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import GlossaryTerm, ReadingLevel, TransformedVersion, TargetFormat
from app.services.ai_client import AIClient, AIClientError, get_ai_client

SIMPLIFY_SYSTEM_PROMPT = """You rewrite educational text to a target reading \
level. Rules, in priority order:
1. NEVER remove a concept, number, unit, condition, exception, or negation \
   present in the semantic representation provided.
2. Preserve technical/scientific terms; on first use, add a short plain-\
   language definition in parentheses rather than deleting the term.
3. Shorten and simplify sentence structure, not the underlying claims.
4. Break long paragraphs into shorter ones; do not merge distinct ideas.
5. If a concept genuinely cannot be simplified without changing its \
   meaning, keep the original phrasing for that specific sentence rather \
   than guessing.
Return the simplified text only."""

# What "target reading level" concretely means for the model -- left vague,
# a model tends to just shorten sentences without changing vocabulary or
# structure. Framed as CEFR/US-grade equivalents since that's the axis
# reading-level research (and Flesch-Kincaid, which the rest of the spec
# already uses for `reading_complexity_before`) is usually discussed in.
_READING_LEVEL_GUIDANCE: dict[ReadingLevel, str] = {
    ReadingLevel.ELEMENTARY: (
        "Elementary school level (roughly US grades 3-5 / age 8-10). Short, "
        "simple sentences (aim for under 15 words each). Common everyday "
        "words wherever the meaning survives; define every technical term "
        "in plain language the first time it appears."
    ),
    ReadingLevel.MIDDLE_SCHOOL: (
        "Middle school level (roughly US grades 6-8 / age 11-13). Clear, "
        "direct sentences (aim for under 20 words each). Everyday "
        "vocabulary where possible; technical terms are kept but briefly "
        "defined on first use."
    ),
    ReadingLevel.HIGH_SCHOOL: (
        "High school level (roughly US grades 9-10 / age 14-16). More "
        "varied sentence structure is fine, but avoid unnecessarily dense, "
        "jargon-heavy academic phrasing. Technical terms are kept; define "
        "any that are unlikely to already be familiar at this level."
    ),
}

# Element types that carry readable prose. Tables/equations/images are
# deliberately left out here -- they have their own transformation targets
# (structured table output, MATH_SPOKEN, ALT_TEXT) and simplifying their
# raw payload as if it were prose would just corrupt it.
_TEXT_ELEMENT_TYPES = {
    ElementType.HEADING,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
}


class Simplifier:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def simplify(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
        reading_level: ReadingLevel,
    ) -> TransformedVersion:
        """Rewrite the document's readable text at `reading_level`.

        Returns a TransformedVersion(format=SIMPLIFIED_TEXT). Does NOT
        self-report a meaning-preservation confidence score -- that's
        services/validation/semantic_validator.py's job, against this
        output. `generation_confidence` here only reflects whether the
        generation step itself completed cleanly.
        """
        source_text = self._extract_source_text(document)
        if not source_text.strip():
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.SIMPLIFIED_TEXT,
                content="",
                generation_confidence=1.0,
                what_changed="No paragraph/heading/list/caption text found in this document to simplify.",
            )

        if reading_level == ReadingLevel.UNCHANGED:
            # No rewrite requested -- skip the AI call entirely rather than
            # asking the model to (hopefully) leave the text alone.
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.SIMPLIFIED_TEXT,
                content=source_text,
                glossary=self._glossary_from_semantics(semantics),
                generation_confidence=1.0,
                what_changed="Reading level set to 'unchanged' -- original text returned as-is.",
            )

        checklist = self._build_preservation_checklist(semantics)
        system = f"{SIMPLIFY_SYSTEM_PROMPT}\n\nTarget reading level: {_READING_LEVEL_GUIDANCE[reading_level]}"
        user = (
            f"Rewrite the following document text for the target reading level.\n\n"
            f"--- MUST BE PRESERVED (do not drop any of these) ---\n{checklist}\n\n"
            f"--- DOCUMENT TEXT ---\n{source_text}"
        )

        try:
            simplified = self._ai.complete(system, user, max_tokens=4000, temperature=0.3)
        except AIClientError as exc:
            # Degrade to "flag for human review" rather than letting a model
            # outage take down the whole /transform request -- consistent
            # with how AIClientError is meant to be handled per ai_client.py.
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.SIMPLIFIED_TEXT,
                content=source_text,
                glossary=self._glossary_from_semantics(semantics),
                generation_confidence=0.0,
                what_changed=f"Simplification failed ({exc}); original text returned unchanged and flagged for review.",
            )

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.SIMPLIFIED_TEXT,
            content=simplified.strip(),
            glossary=self._glossary_from_semantics(semantics),
            generation_confidence=0.85,
            what_changed=(
                f"Rewritten at {reading_level.value.replace('_', ' ')} reading level; "
                "technical terms preserved with inline definitions."
            ),
        )

    # -- internals ---------------------------------------------------

    @staticmethod
    def _extract_source_text(document: StructuredDocument) -> str:
        """Pulls readable prose out in document order, tagging headings so
        the model preserves structure instead of flattening everything
        into one undifferentiated block."""
        parts: list[str] = []
        for section in sorted(document.sections, key=lambda s: s.order):
            if section.type not in _TEXT_ELEMENT_TYPES:
                continue
            if section.type == ElementType.HEADING and section.text:
                level = section.heading_level or 1
                parts.append(f"{'#' * level} {section.text}")
            elif section.type == ElementType.LIST and section.list_items:
                parts.append("\n".join(f"- {item}" for item in section.list_items))
            elif section.text:
                parts.append(section.text)
        return "\n\n".join(parts)

    @staticmethod
    def _build_preservation_checklist(semantics: SemanticRepresentation) -> str:
        """Compact, model-readable dump of the exact things the source
        rule ('reduce difficulty, not meaning') says can never be dropped:
        concepts, quantities, and negations/exceptions."""
        lines: list[str] = []

        technical_concepts = [c for c in semantics.concepts if c.is_technical_term]
        if technical_concepts:
            lines.append("Technical terms (keep the term; may add a plain-language definition):")
            lines.extend(f"  - {c.name}: {c.definition}" for c in technical_concepts)

        if semantics.quantities:
            lines.append("Numbers/quantities (keep exact values and units):")
            lines.extend(
                f"  - {q.value}{f' {q.unit}' if q.unit else ''} ({q.context})"
                for q in semantics.quantities
            )

        if semantics.negations:
            lines.append("Negations/exceptions/conditions (must survive intact):")
            lines.extend(f"  - {n.text}" for n in semantics.negations)

        return "\n".join(lines) if lines else "(none extracted)"

    @staticmethod
    def _glossary_from_semantics(semantics: SemanticRepresentation) -> list[GlossaryTerm]:
        """Technical terms double as a starting glossary -- reuses the same
        canonical definitions rather than asking the model to invent a
        second, possibly inconsistent, set."""
        return [
            GlossaryTerm(term=c.name, definition=c.definition)
            for c in semantics.concepts
            if c.is_technical_term
        ]
