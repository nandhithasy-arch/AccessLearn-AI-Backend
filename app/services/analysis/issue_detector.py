"""
Generates the accessibility issue report (spec section 3) shown to the user
BEFORE any transformation happens, plus the four accessibility sub-scores.

Deterministic checks (no AI needed):
    - Missing alt_text / description on IMAGE, DIAGRAM, CHART sections.
    - Missing table headers (no cells with is_header=True).
    - Missing equation_mathml / equation_spoken on EQUATION sections.
    - Low OCR confidence on text extracted from scanned pages (Stage 3).
    - Missing captions on audio/video sections.

AI-assisted checks:
    - Reading complexity scoring (can start with a deterministic formula
      like Flesch-Kincaid, then have AI explain *why* a passage scored high).
    - "Color-only" information detection (needs visual/semantic judgment).
"""
from __future__ import annotations

from app.core.config import get_settings
from app.models.document import (
    AccessibilityIssue,
    AccessibilityScore,
    DocumentSection,
    ElementType,
    Severity,
)


class IssueDetector:
    def __init__(self, ocr_confidence_threshold: float | None = None) -> None:
        settings = get_settings()
        self._ocr_threshold = (
            ocr_confidence_threshold
            if ocr_confidence_threshold is not None
            else settings.ocr_confidence_threshold
        )

    def detect(self, sections: list[DocumentSection]) -> list[AccessibilityIssue]:
        issues: list[AccessibilityIssue] = []

        for section in sections:
            if section.type in (ElementType.IMAGE, ElementType.DIAGRAM, ElementType.CHART):
                if not section.alt_text and not section.description:
                    issues.append(
                        AccessibilityIssue(
                            element_id=section.id,
                            issue="Missing alternative description",
                            severity=Severity.HIGH,
                            suggested_action="Generate detailed description",
                        )
                    )
            if section.type == ElementType.TABLE:
                headers = [c for c in (section.cells or []) if c.is_header]
                if not headers:
                    issues.append(
                        AccessibilityIssue(
                            element_id=section.id,
                            issue="Poor structural labeling",
                            severity=Severity.MEDIUM,
                            suggested_action="Convert to accessible table with headers",
                        )
                    )
            if section.type == ElementType.EQUATION:
                if not section.equation_mathml and not section.equation_spoken:
                    issues.append(
                        AccessibilityIssue(
                            element_id=section.id,
                            issue="Not screen-reader friendly",
                            severity=Severity.HIGH,
                            suggested_action="Convert to MathML/spoken form",
                        )
                    )

            # Stage 3: surface low-OCR-confidence spans as review warnings so
            # they appear in the Analysis issue table, not only as a silent
            # needs_human_review flag on the section model.
            if (
                section.ocr_confidence is not None
                and section.ocr_confidence < self._ocr_threshold
            ):
                pct = round(section.ocr_confidence * 100)
                threshold_pct = round(self._ocr_threshold * 100)
                severity = (
                    Severity.HIGH if section.ocr_confidence < 0.5 else Severity.MEDIUM
                )
                preview = (section.text or "")[:60].strip()
                suffix = f' ("{preview}...")' if preview else ""
                issues.append(
                    AccessibilityIssue(
                        element_id=section.id,
                        issue=(
                            f"Low OCR confidence ({pct}% < {threshold_pct}% threshold)"
                            f"{suffix}"
                        ),
                        severity=severity,
                        suggested_action=(
                            "Review the extracted text against the original scanned page; "
                            "correct any misread characters before transforming."
                        ),
                    )
                )

            # TODO: complex-paragraph detection via reading_complexity score,
            # missing captions for audio/video sections, color-only info via AI.

        return issues

    # Point deduction per issue, by severity -- same scale for every category
    # and for `overall`.
    _DEDUCTIONS = {
        Severity.CRITICAL: 25,
        Severity.HIGH: 10,
        Severity.MEDIUM: 4,
        Severity.LOW: 1,
        Severity.INFORMATIONAL: 0,
    }

    # Which element types' issues count against which sub-score. An issue on
    # a type not listed here still counts against `overall` but not against
    # any category -- audio/video element types don't exist yet (Phase 7
    # TODO), so `audio` currently only ever reads 100.
    _CATEGORY_TYPES = {
        "visual": {ElementType.IMAGE, ElementType.DIAGRAM, ElementType.CHART},
        "structural": {ElementType.TABLE, ElementType.HEADING, ElementType.LIST},
        "text": {ElementType.PARAGRAPH, ElementType.EQUATION, ElementType.CODE},
        "audio": set(),
    }

    def score(self, sections: list[DocumentSection], issues: list[AccessibilityIssue]) -> AccessibilityScore:
        """
        overall/text/visual/audio/structural each start at 100 and lose
        points per issue whose severity maps to a deduction (critical=25,
        high=10, medium=4, low=1, informational=0), clamped to [0, 100].
        `overall` is deducted by every issue; a sub-score is deducted only
        by issues on elements whose type falls in that sub-score's category
        (see `_CATEGORY_TYPES`).
        """
        section_by_id = {s.id: s for s in sections}
        overall_total = 0
        category_totals = {category: 0 for category in self._CATEGORY_TYPES}

        for issue in issues:
            deduction = self._DEDUCTIONS.get(issue.severity, 0)
            overall_total += deduction
            section = section_by_id.get(issue.element_id)
            if section is None:
                continue
            for category, types in self._CATEGORY_TYPES.items():
                if section.type in types:
                    category_totals[category] += deduction

        def clamp(total: int) -> int:
            return max(0, min(100, 100 - total))

        return AccessibilityScore(
            overall=clamp(overall_total),
            text=clamp(category_totals["text"]),
            visual=clamp(category_totals["visual"]),
            audio=clamp(category_totals["audio"]),
            structural=clamp(category_totals["structural"]),
        )
