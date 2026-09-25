"""
Compares a TransformedVersion against the document's SemanticRepresentation.

Per spec section 9's rule ("AI should interpret language and code should
validate exact values"), checks here are split:

    Deterministic (this file delegates to number_validator / plain string
    matching, NO model call, run first, always trustworthy):
        - numerical values, units, comparators
        - negation trigger words present/absent
        - procedure step ordering

    Semantic / lexical (fuzzier, run second, still NO second LLM call -- a
    validator must not share failure modes with the model that produced the
    transformed text):
        - concept preservation via name match + token-overlap of the
          definition against sliding windows of the transformed content
        - relationship preservation via co-occurrence of endpoint names
          near a relation keyword

Overall score is a weighted combination; critical categories (numerical,
negation) weigh more heavily than terminology/wording, per the severity
levels in spec section 8.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from app.models.document import Severity
from app.models.semantic import (
    Concept,
    Negation,
    ProcedureStep,
    Relationship,
    SemanticRepresentation,
)
from app.models.transformation import TransformedVersion
from app.models.validation import (
    ConceptPreservationCheck,
    NegationPreservationCheck,
    NumericalPreservationCheck,
    PreservationWarning,
    RelationshipPreservationCheck,
    SequencePreservationCheck,
    ValidationResult,
)
from app.services.validation.number_validator import NumberValidator

# Weights sum to 100; numerical + negation are the critical buckets.
SCORE_WEIGHTS = {
    "concepts": 30,
    "relationships": 20,
    "numerical": 25,
    "negation": 20,
    "sequence": 5,
}

# Below this overall score, always flag for human review even if no CRITICAL.
_REVIEW_SCORE_THRESHOLD = 85.0

# Concept counts as preserved if its name appears, or definition token
# overlap with any content window exceeds this.
_CONCEPT_OVERLAP_THRESHOLD = 0.45

# Relation keywords the model (and our simplifier) tend to use.
_RELATION_KEYWORDS: dict[str, list[str]] = {
    "causes": ["cause", "causes", "causing", "leads to", "results in", "produces"],
    "is_part_of": ["part of", "component of", "within", "inside"],
    "precedes": ["before", "precedes", "prior to", "then", "followed by"],
    "enables": ["enables", "enable", "allows", "makes possible", "helps"],
    "helps_enable": ["helps", "enables", "enable", "allows"],
    "contrasts_with": ["unlike", "whereas", "in contrast", "however", "but"],
    "requires": ["requires", "needs", "depends on", "necessary"],
    "produces": ["produces", "produce", "yields", "releases", "creates"],
    "converts": ["converts", "convert", "transforms", "turns into"],
}

_NEGATION_SYNONYMS: dict[str, list[str]] = {
    "not": ["not", "no", "never", "neither", "nor"],
    "never": ["never", "not", "no"],
    "except": ["except", "excepting", "aside from", "other than"],
    "unless": ["unless", "except when", "except if"],
    "cannot": ["cannot", "can't", "can not", "unable"],
    "only": ["only", "solely", "exclusively"],
    "without": ["without", "lacking", "absent"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "being", "that", "this", "with", "as", "by",
    "from", "at", "it", "its", "into", "than", "then", "also", "which", "who",
}


class SemanticValidator:
    def __init__(self) -> None:
        self._numbers = NumberValidator()

    def validate(
        self,
        semantics: SemanticRepresentation,
        transformed: TransformedVersion,
    ) -> ValidationResult:
        """Compare transformed content against the canonical semantic ground truth.

        Returns a ValidationResult with per-category checks, itemized
        warnings (severity-tagged), an overall weighted consistency score,
        and requires_human_review when anything critical is missing or the
        score falls below threshold.
        """
        content = transformed.content or ""
        warnings: list[PreservationWarning] = []

        concepts_check = self._check_concepts(semantics.concepts, content, warnings)
        relationships_check = self._check_relationships(
            semantics.relationships, content, warnings
        )
        numerical_check = self._check_numerical(semantics, content, warnings)
        negation_check = self._check_negations(semantics.negations, content, warnings)
        sequence_check = self._check_sequence(semantics.procedure_steps, content, warnings)

        self._check_visual_only(semantics.visual_only_information, content, warnings)

        overall = self._score(
            concepts_check,
            relationships_check,
            numerical_check,
            negation_check,
            sequence_check,
        )

        has_critical = any(w.severity == Severity.CRITICAL for w in warnings)
        requires_review = has_critical or overall < _REVIEW_SCORE_THRESHOLD

        # Surface pipeline warnings from the semantic extraction step itself.
        for note in semantics.warnings or []:
            warnings.append(
                PreservationWarning(
                    category="terminology",
                    severity=Severity.INFORMATIONAL,
                    message=f"Semantic extraction note: {note}",
                )
            )

        return ValidationResult(
            document_id=semantics.document_id,
            transformed_version_id=transformed.id,
            overall_consistency_score=round(overall, 1),
            concepts=concepts_check,
            relationships=relationships_check,
            numerical=numerical_check,
            negation=negation_check,
            sequence=sequence_check,
            warnings=warnings,
            requires_human_review=requires_review,
        )

    # -- category checks ---------------------------------------------

    def _check_concepts(
        self,
        concepts: list[Concept],
        content: str,
        warnings: list[PreservationWarning],
    ) -> ConceptPreservationCheck:
        if not concepts:
            return ConceptPreservationCheck(total=0, preserved=0)

        lowered = content.lower()
        windows = self._content_windows(content)
        missing: list[str] = []
        preserved = 0

        for concept in concepts:
            name = (concept.name or "").strip()
            if not name:
                continue
            name_hit = name.lower() in lowered
            def_overlap = self._best_definition_overlap(concept.definition or "", windows)
            ok = name_hit or def_overlap >= _CONCEPT_OVERLAP_THRESHOLD

            # Technical terms must keep the term itself -- definition paraphrase
            # alone is not enough (spec: preserve technical/scientific terms).
            if concept.is_technical_term and not name_hit:
                ok = False

            if ok:
                preserved += 1
            else:
                missing.append(concept.id)
                severity = Severity.HIGH if concept.is_technical_term else Severity.MEDIUM
                warnings.append(
                    PreservationWarning(
                        category="concept",
                        severity=severity,
                        message=(
                            f"Concept '{concept.name}' appears to be missing from the "
                            f"transformed text"
                            + (
                                f" (definition overlap {def_overlap:.2f})"
                                if concept.definition
                                else ""
                            )
                            + "."
                        ),
                        original_reference=concept.id,
                    )
                )

        return ConceptPreservationCheck(
            total=len(concepts),
            preserved=preserved,
            missing_concept_ids=missing,
        )

    def _check_relationships(
        self,
        relationships: list[Relationship],
        content: str,
        warnings: list[PreservationWarning],
    ) -> RelationshipPreservationCheck:
        if not relationships:
            return RelationshipPreservationCheck(total=0, preserved=0)

        lowered = content.lower()
        missing: list[str] = []
        preserved = 0

        for rel in relationships:
            source = (rel.source or "").strip().lower()
            target = (rel.target or "").strip().lower()
            if not source or not target:
                continue

            source_present = source in lowered
            target_present = target in lowered
            relation_ok = self._relation_keyword_present(rel.relation, lowered)

            # Both endpoints must appear; relation keyword is a soft signal
            # (paraphrase may drop the exact verb but keep the claim).
            ok = source_present and target_present
            if ok and rel.relation and not relation_ok:
                # Soft miss: endpoints present but relation phrasing gone --
                # still count as preserved, emit a low warning.
                warnings.append(
                    PreservationWarning(
                        category="relationship",
                        severity=Severity.LOW,
                        message=(
                            f"Relationship '{rel.source} {rel.relation} {rel.target}' "
                            "endpoints are present but the relation phrasing was not found."
                        ),
                        original_reference=rel.id,
                    )
                )

            if ok:
                preserved += 1
            else:
                missing.append(rel.id)
                warnings.append(
                    PreservationWarning(
                        category="relationship",
                        severity=Severity.MEDIUM,
                        message=(
                            f"Relationship '{rel.source} —{rel.relation}→ {rel.target}' "
                            "is not clearly preserved in the transformed text."
                        ),
                        original_reference=rel.id,
                    )
                )

        return RelationshipPreservationCheck(
            total=len(relationships),
            preserved=preserved,
            missing_relationship_ids=missing,
        )

    def _check_numerical(
        self,
        semantics: SemanticRepresentation,
        content: str,
        warnings: list[PreservationWarning],
    ) -> NumericalPreservationCheck:
        total, preserved, mismatched_ids = self._numbers.compare(
            semantics.quantities, content
        )
        by_id = {q.id: q for q in semantics.quantities}
        for qid in mismatched_ids:
            q = by_id.get(qid)
            label = (
                f"{q.value}{f' {q.unit}' if q and q.unit else ''} ({q.context})"
                if q
                else qid
            )
            warnings.append(
                PreservationWarning(
                    category="numerical",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Quantity '{label}' is missing or its comparator appears "
                        "to have flipped in the transformed text."
                    ),
                    original_reference=qid,
                )
            )
        return NumericalPreservationCheck(
            total=total,
            preserved=preserved,
            mismatched_quantity_ids=mismatched_ids,
        )

    def _check_negations(
        self,
        negations: list[Negation],
        content: str,
        warnings: list[PreservationWarning],
    ) -> NegationPreservationCheck:
        if not negations:
            return NegationPreservationCheck(total=0, preserved=0)

        lowered = content.lower()
        missing: list[str] = []
        preserved = 0

        for neg in negations:
            trigger = (neg.trigger_word or "").strip().lower()
            text = (neg.text or "").strip().lower()
            trigger_ok = self._negation_trigger_present(trigger, lowered)
            # Also accept a substantial fragment of the original negation clause.
            fragment_ok = False
            if text:
                tokens = [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]
                if tokens:
                    hits = sum(1 for t in tokens if t in lowered)
                    fragment_ok = hits / len(tokens) >= 0.6

            if trigger_ok or fragment_ok:
                preserved += 1
            else:
                missing.append(neg.id)
                warnings.append(
                    PreservationWarning(
                        category="negation",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Negation/exception '{neg.text or neg.trigger_word}' "
                            "is missing from the transformed text."
                        ),
                        original_reference=neg.id,
                    )
                )

        return NegationPreservationCheck(
            total=len(negations),
            preserved=preserved,
            missing_negation_ids=missing,
        )

    def _check_sequence(
        self,
        steps: list[ProcedureStep],
        content: str,
        warnings: list[PreservationWarning],
    ) -> Optional[SequencePreservationCheck]:
        if not steps:
            return None

        ordered = sorted(steps, key=lambda s: s.order)
        lowered = content.lower()
        positions: list[tuple[ProcedureStep, int]] = []
        out_of_order: list[str] = []

        for step in ordered:
            pos = self._step_position(step, lowered)
            if pos is None:
                out_of_order.append(step.id)
                warnings.append(
                    PreservationWarning(
                        category="sequence",
                        severity=Severity.MEDIUM,
                        message=f"Procedure step {step.order} ('{step.text[:80]}') was not found.",
                        original_reference=step.id,
                    )
                )
            else:
                positions.append((step, pos))

        # Relative order among steps that *were* found.
        order_preserved = True
        for i in range(1, len(positions)):
            if positions[i][1] < positions[i - 1][1]:
                order_preserved = False
                out_of_order.append(positions[i][0].id)
                warnings.append(
                    PreservationWarning(
                        category="sequence",
                        severity=Severity.HIGH,
                        message=(
                            f"Procedure step {positions[i][0].order} appears before "
                            f"step {positions[i - 1][0].order} in the transformed text."
                        ),
                        original_reference=positions[i][0].id,
                    )
                )

        if out_of_order and order_preserved:
            # Missing steps still mean order is not fully preserved.
            order_preserved = False

        return SequencePreservationCheck(
            total_steps=len(steps),
            order_preserved=order_preserved and len(out_of_order) == 0,
            out_of_order_step_ids=list(dict.fromkeys(out_of_order)),
        )

    def _check_visual_only(
        self,
        visual_items: list[str],
        content: str,
        warnings: list[PreservationWarning],
    ) -> None:
        """Informational: visual-only facts often cannot appear in pure-text
        transforms; surface them so a reviewer knows what was diagram-only."""
        if not visual_items:
            return
        lowered = content.lower()
        for item in visual_items:
            tokens = [t for t in _TOKEN_RE.findall(item.lower()) if t not in _STOPWORDS]
            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in lowered)
            if hits / len(tokens) < 0.4:
                warnings.append(
                    PreservationWarning(
                        category="visual",
                        severity=Severity.MEDIUM,
                        message=(
                            "Visual-only information may be missing from this text "
                            f"output: {item[:160]}"
                            + ("…" if len(item) > 160 else "")
                        ),
                        original_reference=None,
                    )
                )

    # -- scoring -----------------------------------------------------

    def _score(
        self,
        concepts: ConceptPreservationCheck,
        relationships: RelationshipPreservationCheck,
        numerical: NumericalPreservationCheck,
        negation: NegationPreservationCheck,
        sequence: Optional[SequencePreservationCheck],
    ) -> float:
        def ratio(preserved: int, total: int) -> float:
            return 1.0 if total == 0 else preserved / total

        parts = {
            "concepts": ratio(concepts.preserved, concepts.total),
            "relationships": ratio(relationships.preserved, relationships.total),
            "numerical": ratio(numerical.preserved, numerical.total),
            "negation": ratio(negation.preserved, negation.total),
            "sequence": (
                1.0
                if sequence is None or sequence.total_steps == 0
                else (1.0 if sequence.order_preserved else 0.0)
            ),
        }
        # Only average over categories that had something to check; if a
        # category is empty, redistribute its weight so an empty-semantics
        # doc doesn't drag the score to a meaningless 100 via zero-totals
        # alone. Empty categories still contribute full weight (ratio=1).
        total_weight = sum(SCORE_WEIGHTS.values())
        score = sum(parts[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS) / total_weight * 100.0
        return max(0.0, min(100.0, score))

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            t.lower()
            for t in _TOKEN_RE.findall(text or "")
            if t.lower() not in _STOPWORDS and len(t) > 1
        ]

    def _content_windows(self, content: str, window_size: int = 40) -> list[list[str]]:
        tokens = self._tokenize(content)
        if not tokens:
            return []
        if len(tokens) <= window_size:
            return [tokens]
        step = max(1, window_size // 2)
        return [
            tokens[i : i + window_size]
            for i in range(0, len(tokens) - window_size + 1, step)
        ] or [tokens]

    def _best_definition_overlap(self, definition: str, windows: list[list[str]]) -> float:
        def_tokens = set(self._tokenize(definition))
        if not def_tokens or not windows:
            return 0.0
        best = 0.0
        for window in windows:
            wset = set(window)
            overlap = len(def_tokens & wset) / len(def_tokens)
            if overlap > best:
                best = overlap
        return best

    @staticmethod
    def _relation_keyword_present(relation: Optional[str], lowered: str) -> bool:
        if not relation:
            return True
        key = relation.strip().lower().replace(" ", "_")
        phrases = _RELATION_KEYWORDS.get(key, [relation.lower()])
        return any(p in lowered for p in phrases)

    @staticmethod
    def _negation_trigger_present(trigger: str, lowered: str) -> bool:
        if not trigger:
            return False
        synonyms = _NEGATION_SYNONYMS.get(trigger, [trigger])
        # Word-boundary-ish check to avoid matching "not" inside "another".
        for syn in synonyms:
            if re.search(rf"(?<![a-z]){re.escape(syn)}(?![a-z])", lowered):
                return True
        return False

    def _step_position(self, step: ProcedureStep, lowered: str) -> Optional[int]:
        text = (step.text or "").strip().lower()
        if not text:
            return None
        # Prefer longest distinctive fragment so short common words don't
        # produce false early positions.
        tokens = [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]
        if not tokens:
            idx = lowered.find(text)
            return idx if idx >= 0 else None
        # Try progressively shorter prefixes of the token sequence.
        for length in range(min(6, len(tokens)), 0, -1):
            phrase = " ".join(tokens[:length])
            idx = lowered.find(phrase)
            if idx >= 0:
                return idx
        # Fallback: first token that appears.
        for t in tokens:
            idx = lowered.find(t)
            if idx >= 0:
                return idx
        return None
