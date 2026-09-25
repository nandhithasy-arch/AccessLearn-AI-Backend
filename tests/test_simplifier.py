"""
Covers Simplifier.simplify():
    - happy path: builds a preservation checklist from the semantic
      representation, calls AIClient.complete, wraps the result
    - ReadingLevel.UNCHANGED short-circuits without calling the AI at all
    - empty document (no text-bearing sections) short-circuits too
    - AIClientError degrades to the original text + generation_confidence=0,
      instead of raising, per ai_client.py's "flag for human review" contract
    - non-text elements (table/image/equation) are excluded from the prompt
    - technical concepts become both prompt checklist entries and glossary
      entries on the returned TransformedVersion

Mocks AIClient itself (not the SDK) since Simplifier's contract is "call
self._ai.complete(...) correctly", not "the Anthropic SDK works" -- that's
already covered by test_ai_client.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.models.document import DocumentSection, ElementType, StructuredDocument
from app.models.semantic import Concept, Negation, Quantity, SemanticRepresentation
from app.models.transformation import ReadingLevel, TargetFormat
from app.services.ai_client import AIClientError
from app.services.transformation.simplifier import Simplifier


def _document(sections: list[DocumentSection]) -> StructuredDocument:
    return StructuredDocument(
        document_id="doc_test123",
        source_filename="photosynthesis.pdf",
        source_format="pdf",
        sections=sections,
    )


def _sample_document() -> StructuredDocument:
    return _document(
        [
            DocumentSection(type=ElementType.HEADING, order=0, heading_level=1, text="Photosynthesis"),
            DocumentSection(
                type=ElementType.PARAGRAPH,
                order=1,
                text="Photosynthesis occurs in chloroplasts and releases oxygen, but only under anaerobic conditions never at night.",
            ),
            DocumentSection(type=ElementType.LIST, order=2, list_items=["Light stage", "Dark stage"]),
            # Non-text elements: must be excluded from the prompt entirely.
            DocumentSection(type=ElementType.TABLE, order=3, cells=[]),
            DocumentSection(type=ElementType.IMAGE, order=4, image_ref="img_1"),
            DocumentSection(type=ElementType.EQUATION, order=5, equation_raw="6CO2 + 6H2O -> C6H12O6 + 6O2"),
        ]
    )


def _sample_semantics() -> SemanticRepresentation:
    return SemanticRepresentation(
        document_id="doc_test123",
        concepts=[
            Concept(name="chloroplast", definition="the part of a plant cell that captures light", is_technical_term=True),
            Concept(name="oxygen", definition="a gas plants release", is_technical_term=False),
        ],
        quantities=[Quantity(value="6", unit="molecules", context="carbon dioxide input")],
        negations=[Negation(text="only under anaerobic conditions", trigger_word="only")],
    )


def _mock_ai(return_value: str = "Plants make food using light.") -> MagicMock:
    ai = MagicMock()
    ai.complete.return_value = return_value
    return ai


class TestSimplifyHappyPath:
    def test_calls_ai_and_wraps_result(self):
        ai = _mock_ai("Plants use light to make food in a part called the chloroplast.")
        result = Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.ELEMENTARY)

        ai.complete.assert_called_once()
        assert result.format == TargetFormat.SIMPLIFIED_TEXT
        assert result.document_id == "doc_test123"
        assert result.content == "Plants use light to make food in a part called the chloroplast."
        assert result.generation_confidence == 0.85
        assert "elementary" in result.what_changed.lower()

    def test_prompt_excludes_non_text_elements(self):
        ai = _mock_ai()
        Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.MIDDLE_SCHOOL)

        user_prompt = ai.complete.call_args.args[1]
        assert "Photosynthesis" in user_prompt
        assert "Light stage" in user_prompt
        # Table cells / image ref / raw equation text should never appear --
        # those go through their own transformation targets, not this prompt.
        assert "img_1" not in user_prompt
        assert "6CO2" not in user_prompt

    def test_prompt_includes_preservation_checklist(self):
        ai = _mock_ai()
        Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.HIGH_SCHOOL)

        user_prompt = ai.complete.call_args.args[1]
        assert "chloroplast" in user_prompt  # technical term
        assert "carbon dioxide input" in user_prompt  # quantity context
        assert "only under anaerobic conditions" in user_prompt  # negation

    def test_reading_level_guidance_in_system_prompt(self):
        ai = _mock_ai()
        Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.ELEMENTARY)

        system_prompt = ai.complete.call_args.args[0]
        assert "grades 3-5" in system_prompt

    def test_glossary_built_from_technical_concepts_only(self):
        ai = _mock_ai()
        result = Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.MIDDLE_SCHOOL)

        assert len(result.glossary) == 1
        assert result.glossary[0].term == "chloroplast"


class TestSimplifyShortCircuits:
    def test_unchanged_reading_level_skips_ai_call(self):
        ai = _mock_ai()
        result = Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.UNCHANGED)

        ai.complete.assert_not_called()
        assert "Photosynthesis" in result.content
        assert result.generation_confidence == 1.0

    def test_empty_document_skips_ai_call(self):
        ai = _mock_ai()
        empty_doc = _document([DocumentSection(type=ElementType.TABLE, order=0, cells=[])])
        result = Simplifier(ai_client=ai).simplify(empty_doc, _sample_semantics(), ReadingLevel.ELEMENTARY)

        ai.complete.assert_not_called()
        assert result.content == ""
        assert "no paragraph" in result.what_changed.lower()


class TestSimplifyDegradesOnAIFailure:
    def test_ai_client_error_returns_original_text_with_zero_confidence(self):
        ai = MagicMock()
        ai.complete.side_effect = AIClientError("model unavailable")

        result = Simplifier(ai_client=ai).simplify(_sample_document(), _sample_semantics(), ReadingLevel.ELEMENTARY)

        assert result.generation_confidence == 0.0
        assert "Photosynthesis" in result.content  # original text, not lost
        assert "model unavailable" in result.what_changed
