"""
Math accessibility: spoken forms + MathML for equations in the document.

Prefers already-extracted equation_latex / equation_mathml / equation_spoken
when present; otherwise asks the model to produce them. Never invents
numbers that are not in the source.
"""
from __future__ import annotations

import json
from typing import Any

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import TargetFormat, TransformedVersion
from app.services.ai_client import AIClient, AIClientError, get_ai_client

MATH_SYSTEM_PROMPT = """You convert mathematical expressions into accessible \
forms. For each equation, return JSON array items:
{"id": str, "raw": str, "spoken": str, "latex": str, "mathml": str}.
spoken must be plain English a screen reader can read (e.g. "a squared plus \
b squared equals c squared"). mathml must be valid Presentation MathML when \
possible. Do not invent symbols or values absent from the source. Return ONLY \
a JSON array."""


class MathSpokenGenerator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def generate(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
    ) -> TransformedVersion:
        equations = [
            s
            for s in document.sections
            if s.type == ElementType.EQUATION
            and (s.equation_raw or s.equation_latex or s.text)
        ]

        if not equations:
            # Also surface quantities that look like formulas from semantics.
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.MATH_SPOKEN,
                content=json.dumps([], indent=2),
                generation_confidence=1.0,
                what_changed="No equation sections found in the document.",
            )

        # Prefer deterministic enrichment when spoken/mathml already present.
        already_complete = all(
            (s.equation_spoken or s.equation_mathml) for s in equations
        )
        if already_complete:
            payload = [_section_to_entry(s) for s in equations]
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.MATH_SPOKEN,
                content=json.dumps(payload, indent=2),
                generation_confidence=1.0,
                what_changed="Packaged existing spoken/MathML equation data (no AI call).",
            )

        items = []
        for s in equations:
            items.append(
                {
                    "id": s.id,
                    "raw": s.equation_raw or s.text or "",
                    "latex": s.equation_latex or "",
                    "spoken": s.equation_spoken or "",
                    "mathml": s.equation_mathml or "",
                }
            )
        user = (
            "Fill in missing spoken and MathML forms for these equations. "
            "Keep existing non-empty fields unchanged.\n\n"
            f"{json.dumps(items, indent=2)}"
        )

        try:
            data = self._ai.complete_json(
                MATH_SYSTEM_PROMPT, user, max_tokens=3000, temperature=0.1
            )
        except AIClientError as exc:
            payload = [_section_to_entry(s) for s in equations]
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.MATH_SPOKEN,
                content=json.dumps(payload, indent=2),
                generation_confidence=0.0,
                what_changed=f"Math enrichment failed ({exc}); raw equation data returned.",
            )

        if not isinstance(data, list):
            data = []
        by_id = {str(item.get("id")): item for item in data if isinstance(item, dict)}
        payload = []
        for s in equations:
            enriched = by_id.get(s.id) or {}
            payload.append(
                {
                    "id": s.id,
                    "raw": s.equation_raw or s.text or enriched.get("raw") or "",
                    "latex": s.equation_latex or enriched.get("latex") or "",
                    "spoken": s.equation_spoken or enriched.get("spoken") or "",
                    "mathml": s.equation_mathml or enriched.get("mathml") or "",
                }
            )

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.MATH_SPOKEN,
            content=json.dumps(payload, indent=2),
            generation_confidence=0.85,
            what_changed="Generated spoken forms and MathML for equation sections.",
        )


def _section_to_entry(s) -> dict[str, Any]:
    return {
        "id": s.id,
        "raw": s.equation_raw or s.text or "",
        "latex": s.equation_latex or "",
        "spoken": s.equation_spoken or "",
        "mathml": s.equation_mathml or "",
    }
