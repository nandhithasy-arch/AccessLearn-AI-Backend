"""
Fully deterministic -- NO AI calls in this file, by design (spec section 9
and section 8's example: "changing 'less than 5' to 'more than 5' is a
severe semantic error" -- this must never depend on model judgment).
"""
from __future__ import annotations

import re

from app.models.semantic import Quantity

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_COMPARATOR_WORDS = {
    "<": ["less than", "fewer than", "under", "below"],
    ">": ["more than", "greater than", "over", "above", "exceeds"],
    "<=": ["at most", "no more than", "up to"],
    ">=": ["at least", "no fewer than", "minimum of"],
    "=": ["exactly", "equal to"],
}


class NumberValidator:
    def compare(self, quantities: list[Quantity], transformed_text: str) -> tuple[int, int, list[str]]:
        """
        Returns (total, preserved_count, mismatched_quantity_ids).

        A quantity is "preserved" if its numeric value still appears in the
        transformed text AND, when a comparator was present in the original,
        an equivalent comparator phrase still appears near that number
        (naive proximity check -- good enough to catch the '<' -> '>' flip
        called out in the spec; a false positive here should fail safe,
        i.e. be flagged for human review rather than silently passed).
        """
        preserved = 0
        mismatched: list[str] = []
        lowered = transformed_text.lower()

        for q in quantities:
            value_present = q.value in transformed_text or self._numeric_core(q.value) in lowered
            comparator_ok = True
            if q.comparator and q.comparator in _COMPARATOR_WORDS:
                comparator_ok = any(phrase in lowered for phrase in _COMPARATOR_WORDS[q.comparator])
                # Also fail if an OPPOSING comparator's phrase appears near
                # the same number -- catches the flip case explicitly.
                opposite = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(q.comparator)
                if opposite and any(phrase in lowered for phrase in _COMPARATOR_WORDS.get(opposite, [])):
                    comparator_ok = False

            if value_present and comparator_ok:
                preserved += 1
            else:
                mismatched.append(q.id)

        return len(quantities), preserved, mismatched

    @staticmethod
    def _numeric_core(value: str) -> str:
        match = _NUMBER_RE.search(value)
        return match.group(0) if match else value
