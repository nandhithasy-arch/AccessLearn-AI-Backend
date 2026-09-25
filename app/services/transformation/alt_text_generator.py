"""
Generates short alt text AND long descriptions separately (spec section 7):
short alt text is not sufficient for diagrams/charts/scientific figures.

Uses AIClient vision support (ImageInput) so the model actually sees the
image bytes, not just a path. Decorative images get alt="" instead of a
forced description; charts may include an approximate data table only when
values are clearly readable -- low confidence is preferred over invention.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.models.document import DocumentSection, ElementType
from app.services.ai_client import AIClient, AIClientError, ImageInput, get_ai_client

logger = logging.getLogger(__name__)

ALT_TEXT_SYSTEM_PROMPT = """You describe educational diagrams/images for \
blind and low-vision learners. Return JSON: {"is_decorative": bool, \
"short_alt_text": str, "long_description": str, "confidence": float, \
"chart_data_table": null or [{"label": str, "value": str}]}. \
short_alt_text is under 125 characters and states what the image IS. \
long_description explains structure, labels, and relationships shown -- \
e.g. for a process diagram, describe the steps and arrows/flow direction, \
not just "a diagram". Never invent labels or data you cannot actually see; \
note uncertainty instead. Set is_decorative true only for purely \
ornamental images with no educational content. confidence is 0.0-1.0. \
For charts/graphs, chart_data_table may list approximate readable values; \
omit or leave null when values cannot be read reliably."""

# Media types Anthropic accepts; sniff from magic bytes when the caller
# doesn't pass one. Default to PNG (most common from PDF extraction).
_MAGIC_TO_MEDIA = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # refined below
)


def _sniff_media_type(image_bytes: bytes) -> str:
    for magic, media in _MAGIC_TO_MEDIA:
        if image_bytes.startswith(magic):
            if media == "image/webp" and b"WEBP" not in image_bytes[:16]:
                continue
            return media
    return "image/png"


class AltTextGenerator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def generate(
        self,
        section: DocumentSection,
        image_bytes: bytes,
        surrounding_text: str,
        media_type: Optional[str] = None,
    ) -> DocumentSection:
        """Fill `section.alt_text` and `section.description` from a vision call.

        Returns the same section instance (mutated) so callers can keep their
        existing id/order/image_ref. On AI failure, leaves existing values
        alone, sets needs_human_review, and does not raise -- matching the
        pipeline's "flag for review, don't crash" contract.
        """
        if not image_bytes:
            section.needs_human_review = True
            if not section.alt_text:
                section.alt_text = ""
            return section

        media = media_type or _sniff_media_type(image_bytes)
        element_kind = (
            section.type.value
            if isinstance(section.type, ElementType)
            else str(section.type or "image")
        )
        context = (surrounding_text or "").strip()
        if len(context) > 2000:
            context = context[:2000] + "…"

        user = (
            f"Describe this educational {element_kind} for accessibility.\n\n"
            f"Surrounding document text (for context only -- describe what is "
            f"IN the image, do not merely restate this text):\n"
            f"{context or '(none provided)'}\n\n"
            "Return the JSON object specified in the system prompt."
        )

        try:
            data = self._ai.complete_json(
                ALT_TEXT_SYSTEM_PROMPT,
                user,
                max_tokens=1500,
                temperature=0.2,
                images=[ImageInput(data=image_bytes, media_type=media)],
            )
        except AIClientError as exc:
            logger.warning("Alt-text generation failed for section %s: %s", section.id, exc)
            section.needs_human_review = True
            if section.alt_text is None:
                section.alt_text = ""
            return section

        if not isinstance(data, dict):
            logger.warning(
                "Alt-text response was not a JSON object for section %s: %r",
                section.id,
                type(data).__name__,
            )
            section.needs_human_review = True
            if section.alt_text is None:
                section.alt_text = ""
            return section

        return self._apply(section, data)

    # -- internals ---------------------------------------------------

    @staticmethod
    def _apply(section: DocumentSection, data: dict[str, Any]) -> DocumentSection:
        is_decorative = bool(data.get("is_decorative", False))
        short = data.get("short_alt_text")
        long_desc = data.get("long_description")
        confidence = data.get("confidence")

        if is_decorative:
            # Spec / WCAG: decorative images get empty alt so screen readers
            # skip them rather than announcing a forced description.
            section.alt_text = ""
            section.description = None
        else:
            if isinstance(short, str):
                # Enforce the 125-char screen-reader convention client-side
                # in case the model overruns.
                section.alt_text = short.strip()[:125]
            elif section.alt_text is None:
                section.alt_text = ""

            if isinstance(long_desc, str) and long_desc.strip():
                section.description = long_desc.strip()
                # If a chart data table was returned, append it so downstream
                # screen-reader HTML / export can surface the numbers.
                table = data.get("chart_data_table")
                if isinstance(table, list) and table:
                    rows = []
                    for row in table:
                        if not isinstance(row, dict):
                            continue
                        label = str(row.get("label") or "").strip()
                        value = str(row.get("value") or "").strip()
                        if label or value:
                            rows.append(f"{label}: {value}" if label else value)
                    if rows:
                        section.description = (
                            section.description
                            + "\n\nApproximate data values:\n"
                            + "\n".join(f"- {r}" for r in rows)
                        )

        try:
            conf = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf = None
        if conf is not None and conf < 0.5:
            section.needs_human_review = True
        if not section.alt_text and not is_decorative:
            section.needs_human_review = True

        return section
