"""
Single chokepoint for all AI calls — Google Gemini via the google-genai SDK.

Design intent (project spec section 9):
    AI should interpret language and images; code should enforce structure
    and validate exact values.

Every service that needs the model (simplification, alt-text, translation,
semantic extraction, etc.) goes through AIClient, not the SDK directly.
Public methods match the previous Anthropic-based client so callers do not
need changes:

    complete() / complete_json() / ping() / ImageInput / get_ai_client()

What's real here:
    - complete() and complete_json() call client.models.generate_content
    - Fails fast with AIConfigurationError if GEMINI_API_KEY is missing
    - Transient errors (rate limit, 5xx) are retried with exponential backoff
    - Non-retryable errors are wrapped in AIClientError
    - complete_json() retries once on parse failure, then returns None
    - Vision: pass images=[ImageInput(...)] (bytes sent as inline parts)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class AIClientError(RuntimeError):
    """Raised when a call to the AI provider fails (after retries for
    transient errors; immediately for non-retryable ones).

    Callers should catch this and degrade gracefully
    (e.g. requires_human_review = True).
    """


class AIConfigurationError(AIClientError):
    """Raised at construction time when no API key is configured."""


_SUPPORTED_IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


@dataclass(frozen=True)
class ImageInput:
    """One image to attach to a vision-capable prompt (e.g. alt-text)."""

    data: bytes
    media_type: str = "image/png"

    def to_gemini_part(self) -> types.Part:
        if self.media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"Unsupported image media_type {self.media_type!r}; "
                f"must be one of {sorted(_SUPPORTED_IMAGE_MEDIA_TYPES)}"
            )
        return types.Part.from_bytes(data=self.data, mime_type=self.media_type)


class AIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        settings = get_settings()
        self._model = model or settings.ai_model or "gemini-3.8-flash"
        key = api_key if api_key is not None else (
            settings.gemini_api_key or settings.anthropic_api_key
        )
        if not key:
            raise AIConfigurationError(
                "GEMINI_API_KEY is not set. Every AI-backed feature "
                "(simplification, alt-text generation, semantic extraction, "
                "semantic validation) needs a real key in the environment "
                "or .env file before it can run. Get a key at "
                "https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        images: Optional[Iterable[ImageInput]] = None,
        max_retries: int = 3,
    ) -> str:
        """Plain-text completion. Pass images for vision-capable prompts."""
        parts: list[Any] = [img.to_gemini_part() for img in (images or [])]
        parts.append(user)

        return self._create_with_retries(
            system=system,
            contents=parts,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        images: Optional[Iterable[ImageInput]] = None,
        max_retries: int = 3,
        retry_on_parse_failure: bool = True,
    ) -> Optional[Any]:
        """
        JSON-mode completion. Returns parsed object/list, or None if the
        response still cannot be parsed after one stricter retry.
        """
        strict_system = (
            f"{system}\n\n"
            "Respond with ONLY raw JSON (an object or array, whichever the "
            "instructions above call for). No markdown code fences, no "
            "explanation, no preamble or trailing text."
        )
        raw = self.complete(
            strict_system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
            max_retries=max_retries,
        )
        parsed = self._parse_json(raw)
        if parsed is not None or not retry_on_parse_failure:
            return parsed

        logger.warning("AIClient.complete_json: response wasn't valid JSON, retrying once")
        firmer_user = (
            f"{user}\n\n"
            "Your previous response could not be parsed as JSON. Respond "
            "again with ONLY valid, complete JSON and nothing else -- no "
            "code fences, no commentary."
        )
        raw_retry = self.complete(
            strict_system,
            firmer_user,
            max_tokens=max_tokens,
            temperature=0.0,
            images=images,
            max_retries=max_retries,
        )
        return self._parse_json(raw_retry)

    def ping(self) -> bool:
        """Cheap round-trip to confirm the API key and model work."""
        try:
            reply = self.complete(
                "Reply with exactly one word.",
                "Reply with the single word: pong",
                max_tokens=16,
                temperature=0.0,
                max_retries=1,
            )
            return "pong" in reply.lower()
        except AIClientError:
            return False

    def _create_with_retries(
        self,
        *,
        system: str,
        contents: list[Any],
        max_tokens: int,
        temperature: float,
        max_retries: int,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(max(1, max_retries)):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                text = getattr(response, "text", None) or ""
                if not text:
                    for cand in getattr(response, "candidates", None) or []:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        for part in getattr(content, "parts", None) or []:
                            t = getattr(part, "text", None)
                            if t:
                                text += t
                return text
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                retryable = (
                    code in (429, 500, 502, 503, 504)
                    or "429" in msg
                    or "resource exhausted" in msg
                    or "unavailable" in msg
                    or "timeout" in msg
                    or "connection" in msg
                )
                if retryable and attempt < max_retries - 1:
                    delay = min(2 ** attempt, 8)
                    logger.warning(
                        "AIClient: retryable error %s on attempt %d/%d, backing off %ss",
                        type(exc).__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise AIClientError(f"Gemini API error: {exc}") from exc

        raise AIClientError(
            f"Gemini API call failed after {max_retries} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _parse_json(raw: str) -> Optional[Any]:
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return None


_ai_client: Optional[AIClient] = None


def get_ai_client() -> AIClient:
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client


def reset_ai_client() -> None:
    """Clear the cached singleton (tests / after env changes)."""
    global _ai_client
    _ai_client = None
