"""
Summary transformation: concise overview that still covers every concept /
quantity / negation from the semantic representation.
"""
from __future__ import annotations

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import TargetFormat, TransformedVersion
from app.services.ai_client import AIClient, AIClientError, get_ai_client

SUMMARY_SYSTEM_PROMPT = """You write a concise educational summary. Rules:
1. Cover every concept, quantity, and negation in the preservation checklist.
2. Prefer clarity over length; aim for roughly 20-30% of the original length.
3. Do not invent facts. Return the summary text only."""


class SummaryGenerator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def generate(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
    ) -> TransformedVersion:
        parts: list[str] = []
        for section in sorted(document.sections, key=lambda s: s.order):
            if section.type in (
                ElementType.HEADING,
                ElementType.PARAGRAPH,
                ElementType.CAPTION,
            ) and section.text:
                parts.append(section.text)
            elif section.type == ElementType.LIST and section.list_items:
                parts.extend(section.list_items)
        source = "\n\n".join(parts)
        if not source.strip():
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.SUMMARY,
                content="",
                generation_confidence=1.0,
                what_changed="No text to summarise.",
            )

        checklist_lines = []
        for c in semantics.concepts:
            checklist_lines.append(f"- {c.name}: {c.definition}")
        for q in semantics.quantities:
            checklist_lines.append(f"- {q.value} {q.unit or ''} ({q.context})")
        for n in semantics.negations:
            checklist_lines.append(f"- {n.text}")
        checklist = "\n".join(checklist_lines) if checklist_lines else "(none)"

        user = (
            f"--- MUST BE PRESERVED ---\n{checklist}\n\n"
            f"--- DOCUMENT ---\n{source}"
        )
        try:
            summary = self._ai.complete(
                SUMMARY_SYSTEM_PROMPT, user, max_tokens=2000, temperature=0.3
            )
        except AIClientError as exc:
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.SUMMARY,
                content=source[:1500],
                generation_confidence=0.0,
                what_changed=f"Summary failed ({exc}); truncated original returned.",
            )

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.SUMMARY,
            content=summary.strip(),
            generation_confidence=0.85,
            what_changed="Generated concise summary preserving key concepts and quantities.",
        )
