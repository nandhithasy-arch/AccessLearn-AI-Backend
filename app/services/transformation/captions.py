"""
Captions / transcript generator: produces WebVTT suitable for media players
or as a timed reading support track for the document content.
"""
from __future__ import annotations

from app.models.document import ElementType, StructuredDocument
from app.models.semantic import SemanticRepresentation
from app.models.transformation import TargetFormat, TransformedVersion
from app.services.ai_client import AIClient, AIClientError, get_ai_client

CAPTIONS_SYSTEM_PROMPT = """You convert educational document text into WebVTT \
captions. Rules:
1. Output valid WebVTT: start with WEBVTT, then cue blocks with timestamps \
   and cue text.
2. Keep each cue under ~80 characters when possible; break long sentences \
   across cues.
3. NEVER drop concepts, numbers, units, or negations from the preservation \
   checklist.
4. Use sequential timestamps starting at 00:00:00.000; estimate ~2.5 seconds \
   per short cue (adjust for length). Do not invent speaker labels unless \
   dialogue is explicit in the source.
Return WebVTT only."""


class CaptionsGenerator:
    def __init__(self, ai_client: AIClient | None = None) -> None:
        self._ai = ai_client or get_ai_client()

    def generate(
        self,
        document: StructuredDocument,
        semantics: SemanticRepresentation,
    ) -> TransformedVersion:
        source = _extract_caption_source(document)
        if not source.strip():
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.CAPTIONS,
                content="WEBVTT\n\n",
                generation_confidence=1.0,
                what_changed="No captionable text found; empty WEBVTT returned.",
            )

        checklist = _preservation_checklist(semantics)
        user = (
            "Produce WebVTT captions for the following educational content.\n\n"
            f"--- MUST BE PRESERVED ---\n{checklist}\n\n"
            f"--- DOCUMENT ---\n{source}"
        )
        try:
            vtt = self._ai.complete(CAPTIONS_SYSTEM_PROMPT, user, max_tokens=4000, temperature=0.2)
        except AIClientError as exc:
            # Deterministic fallback: one cue per paragraph.
            vtt = _fallback_vtt(source)
            return TransformedVersion(
                document_id=document.document_id,
                format=TargetFormat.CAPTIONS,
                content=vtt,
                generation_confidence=0.0,
                what_changed=f"Caption generation failed ({exc}); deterministic WEBVTT fallback used.",
            )

        content = vtt.strip()
        if not content.upper().startswith("WEBVTT"):
            content = "WEBVTT\n\n" + content

        return TransformedVersion(
            document_id=document.document_id,
            format=TargetFormat.CAPTIONS,
            content=content,
            generation_confidence=0.85,
            what_changed="Generated WebVTT caption track from document prose.",
        )


def _extract_caption_source(document: StructuredDocument) -> str:
    parts: list[str] = []
    for section in sorted(document.sections, key=lambda s: s.order):
        if section.type == ElementType.HEADING and section.text:
            parts.append(section.text)
        elif section.type == ElementType.PARAGRAPH and section.text:
            parts.append(section.text)
        elif section.type == ElementType.LIST and section.list_items:
            parts.extend(section.list_items)
        elif section.type == ElementType.CAPTION and section.text:
            parts.append(section.text)
    return "\n\n".join(parts)


def _fallback_vtt(source: str) -> str:
    cues: list[str] = ["WEBVTT", ""]
    t = 0.0
    for para in [p.strip() for p in source.split("\n\n") if p.strip()]:
        # ~150 words/min ≈ 2.5 chars/sec rough estimate
        duration = max(2.0, min(12.0, len(para) / 18.0))
        start = _fmt(t)
        end = _fmt(t + duration)
        cues.append(f"{start} --> {end}")
        cues.append(para)
        cues.append("")
        t += duration + 0.3
    return "\n".join(cues)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _preservation_checklist(semantics: SemanticRepresentation) -> str:
    lines: list[str] = []
    for c in semantics.concepts:
        lines.append(f"- concept: {c.name}")
    for q in semantics.quantities:
        lines.append(f"- quantity: {q.value} {q.unit or ''} ({q.context})")
    for n in semantics.negations:
        lines.append(f"- negation: {n.text}")
    return "\n".join(lines) if lines else "(none)"
