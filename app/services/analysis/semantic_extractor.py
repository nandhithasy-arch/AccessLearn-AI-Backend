"""
Stage 5: semantic analysis.

Builds the ONE canonical `SemanticRepresentation` for a document. This is
the most important service in the whole system -- every later
transformation gets validated against this, and only this. Run it once,
cache it on the document, never regenerate it per-transformation-request.

Implementation note: ask the model to extract concepts/relationships/
quantities/negations/steps in a SINGLE structured call over the full
document text (not per-paragraph) so relationships that span paragraphs
aren't missed. Chunk only if the document exceeds context limits, and if
you do, stitch relationships across chunk boundaries manually.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import (
    Concept,
    Negation,
    ProcedureStep,
    Quantity,
    Relationship,
    SemanticRepresentation,
)
from app.services.ai_client import AIClient, AIClientError, get_ai_client

logger = logging.getLogger(__name__)

SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """You extract a structured semantic \
representation from educational content. Identify: main_topic, \
learning_objectives, concepts (name + definition), relationships between \
concepts (source, relation, target), quantities (value, unit, context, \
comparator), negations/exceptions (text, trigger_word), procedure_steps \
(order, text) if the content describes a process, and \
visual_only_information (facts present ONLY in a diagram/image, not in \
the body text). Be exhaustive -- this representation will be used to check \
whether later simplified/translated versions accidentally drop information. \
Return JSON matching the SemanticRepresentation schema."""

# Element types whose text (or description) should be fed to the model.
# Visual types are included when they already have a generated description
# so visual-only facts can be flagged; raw image bytes are not sent here
# (that is alt_text_generator's job).
_TEXT_BEARING_TYPES = {
    ElementType.HEADING,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.TABLE,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
    ElementType.CITATION,
    ElementType.EQUATION,
    ElementType.CODE,
    ElementType.IMAGE,
    ElementType.DIAGRAM,
    ElementType.CHART,
}


class SemanticExtractor:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def extract(self, document: StructuredDocument) -> SemanticRepresentation:
        """Build the canonical SemanticRepresentation for `document`.

        Serializes all text-bearing sections (with element ids so the model
        can cite sources), calls AIClient.complete_json once over the full
        text, and coerces the result into SemanticRepresentation.

        On AIClientError or a persistent JSON parse failure (complete_json
        already retries once with a stricter reminder), returns an empty
        representation with a pipeline warning rather than crashing --
        matching the "flag for human review" contract used by Simplifier.
        """
        source_text = self._serialize_document(document)
        if not source_text.strip():
            return SemanticRepresentation(
                document_id=document.document_id,
                warnings=[
                    "No extractable text found in document sections; "
                    "semantic representation is empty."
                ],
            )

        user = (
            "Extract the semantic representation from the following educational "
            "document. Each block is tagged with its element id in [id=...] so you "
            "can fill source_element_ids accurately.\n\n"
            "Return a single JSON object with exactly these fields:\n"
            "  main_topic: string or null\n"
            "  learning_objectives: array of strings\n"
            "  concepts: array of {name, definition, is_technical_term, source_element_ids}\n"
            "  relationships: array of {source, relation, target, source_element_ids}\n"
            "  quantities: array of {value, unit, context, comparator, source_element_ids}\n"
            "  negations: array of {text, trigger_word, source_element_ids}\n"
            "  procedure_steps: array of {order, text, source_element_ids}\n"
            "  visual_only_information: array of strings\n\n"
            f"--- DOCUMENT ---\n{source_text}"
        )

        try:
            data = self._ai.complete_json(
                SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
                user,
                max_tokens=4000,
                temperature=0.2,
            )
        except AIClientError as exc:
            logger.warning("Semantic extraction AI call failed: %s", exc)
            return SemanticRepresentation(
                document_id=document.document_id,
                warnings=[
                    f"Semantic extraction failed ({exc}); empty representation "
                    "returned for human review."
                ],
            )

        if data is None:
            return SemanticRepresentation(
                document_id=document.document_id,
                warnings=[
                    "Model response could not be parsed as JSON after retry; "
                    "empty representation returned for human review."
                ],
            )

        if not isinstance(data, dict):
            return SemanticRepresentation(
                document_id=document.document_id,
                warnings=[
                    f"Model returned {type(data).__name__} instead of a JSON object; "
                    "empty representation returned for human review."
                ],
            )

        return self._coerce(document.document_id, data)

    # -- internals ---------------------------------------------------

    @staticmethod
    def _serialize_document(document: StructuredDocument) -> str:
        """Compact, ordered dump of every text-bearing section, tagged with
        element id so the model can populate source_element_ids. Mirrors
        Simplifier._extract_source_text but keeps ids and includes tables /
        equations / image descriptions when present."""
        parts: list[str] = []
        for section in sorted(document.sections, key=lambda s: s.order):
            if section.type not in _TEXT_BEARING_TYPES:
                continue

            tag = f"[id={section.id} type={section.type.value}]"
            body: Optional[str] = None

            if section.type == ElementType.HEADING and section.text:
                level = section.heading_level or 1
                body = f"{'#' * level} {section.text}"
            elif section.type == ElementType.LIST and section.list_items:
                body = "\n".join(f"- {item}" for item in section.list_items)
            elif section.type == ElementType.TABLE and section.cells:
                # Flatten cells into a readable grid so quantities/relations
                # inside tables are not lost.
                rows: dict[int, list[str]] = {}
                for cell in section.cells:
                    rows.setdefault(cell.row, []).append(cell.text)
                body = "\n".join(" | ".join(cols) for _, cols in sorted(rows.items()))
            elif section.type == ElementType.EQUATION:
                body = (
                    section.equation_spoken
                    or section.equation_latex
                    or section.equation_raw
                    or section.text
                )
            elif section.type in (
                ElementType.IMAGE,
                ElementType.DIAGRAM,
                ElementType.CHART,
            ):
                # Prefer a generated description; fall back to alt_text.
                body = section.description or section.alt_text
                if body:
                    body = f"(visual) {body}"
            elif section.text:
                body = section.text

            if body and body.strip():
                parts.append(f"{tag}\n{body.strip()}")

        return "\n\n".join(parts)

    @staticmethod
    def _coerce(document_id: str, data: dict[str, Any]) -> SemanticRepresentation:
        """Map a model JSON dict onto SemanticRepresentation, tolerating
        missing/extra fields and type noise so a slightly off-schema response
        still yields a usable representation rather than a validation crash."""
        warnings: list[str] = []
        if isinstance(data.get("warnings"), list):
            warnings.extend(str(w) for w in data["warnings"])

        concepts: list[Concept] = []
        for raw in data.get("concepts") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            concepts.append(
                Concept(
                    name=str(raw["name"]),
                    definition=str(raw.get("definition") or ""),
                    is_technical_term=bool(raw.get("is_technical_term", False)),
                    source_element_ids=_as_str_list(raw.get("source_element_ids")),
                )
            )

        relationships: list[Relationship] = []
        for raw in data.get("relationships") or []:
            if not isinstance(raw, dict):
                continue
            source = raw.get("source") or raw.get("source_concept_id")
            target = raw.get("target") or raw.get("target_concept_id")
            relation = raw.get("relation")
            if not (source and target and relation):
                continue
            relationships.append(
                Relationship(
                    source=str(source),
                    relation=str(relation),
                    target=str(target),
                    source_element_ids=_as_str_list(raw.get("source_element_ids")),
                )
            )

        quantities: list[Quantity] = []
        for raw in data.get("quantities") or []:
            if not isinstance(raw, dict) or raw.get("value") is None:
                continue
            quantities.append(
                Quantity(
                    value=str(raw["value"]),
                    unit=_optional_str(raw.get("unit")),
                    context=str(raw.get("context") or ""),
                    comparator=_optional_str(raw.get("comparator")),
                    source_element_ids=_as_str_list(
                        raw.get("source_element_ids") or raw.get("source_element_id")
                    ),
                )
            )

        negations: list[Negation] = []
        for raw in data.get("negations") or []:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            negations.append(
                Negation(
                    text=str(raw["text"]),
                    trigger_word=str(raw.get("trigger_word") or ""),
                    source_element_ids=_as_str_list(raw.get("source_element_ids")),
                )
            )

        procedure_steps: list[ProcedureStep] = []
        for raw in data.get("procedure_steps") or data.get("sequences") or []:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            try:
                order = int(raw.get("order", len(procedure_steps) + 1))
            except (TypeError, ValueError):
                order = len(procedure_steps) + 1
            procedure_steps.append(
                ProcedureStep(
                    order=order,
                    text=str(raw["text"]),
                    source_element_ids=_as_str_list(raw.get("source_element_ids")),
                )
            )

        visual_only: list[str] = []
        for item in data.get("visual_only_information") or []:
            if isinstance(item, str) and item.strip():
                visual_only.append(item.strip())
            elif isinstance(item, dict):
                desc = item.get("description") or item.get("text")
                if desc:
                    visual_only.append(str(desc).strip())

        main_topic = data.get("main_topic")
        if main_topic is not None:
            main_topic = str(main_topic) or None

        learning_objectives = [
            str(o) for o in (data.get("learning_objectives") or []) if o
        ]

        return SemanticRepresentation(
            document_id=document_id,
            main_topic=main_topic,
            learning_objectives=learning_objectives,
            concepts=concepts,
            relationships=relationships,
            quantities=quantities,
            negations=negations,
            procedure_steps=procedure_steps,
            visual_only_information=visual_only,
            warnings=warnings,
        )


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v)]
    return [str(value)]


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
