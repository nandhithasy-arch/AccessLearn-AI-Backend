"""
Proves `AIClient` itself is wired correctly, with the real `anthropic`
SDK objects mocked out at the `messages.create` boundary -- this is the one
place in the test suite that should NOT hit the network even with a real
key, so every test controls exactly what the "model" returns/raises.

Covers what Task 1 (wire AIClient to a real model) actually needed to add
on top of the original bare `messages.create` call:
    - fails fast at construction if there's no API key
    - a successful call returns the concatenated text blocks
    - a transient error (rate limit) is retried and can still succeed
    - a non-retryable error (bad request) raises immediately, no retries
    - complete_json parses clean JSON first try
    - complete_json retries once on a malformed first response, then
      returns None if the retry is also unparseable (never raises)
    - images are attached to the API call as base64 content blocks

**Testing note:** this sandbox has no network access, so `anthropic`,
`pytest`, etc. couldn't actually be installed/run here (same limitation
noted in STATUS.md for the extraction tests). The exception class names
(`RateLimitError`, `APIConnectionError`, `APITimeoutError`,
`InternalServerError`, `APIStatusError`) and their constructor shapes match
the `anthropic` Python SDK's public API as of the pinned `anthropic==0.34.2`
in requirements.txt. Please run `pytest` for real in an environment with
dependencies installed before merging.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from app.services.ai_client import (
    AIClient,
    AIClientError,
    AIConfigurationError,
    ImageInput,
)


def _fake_response(text: str):
    """Builds an object shaped like `anthropic.types.Message` enough for
    `AIClient.complete` to pull text back out of it."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit_error() -> anthropic.RateLimitError:
    response = httpx.Response(429, request=_fake_request(), json={"error": {"message": "rate limited"}})
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _bad_request_error() -> anthropic.BadRequestError:
    response = httpx.Response(400, request=_fake_request(), json={"error": {"message": "bad request"}})
    return anthropic.BadRequestError("bad request", response=response, body=None)


def _client_with_mocked_sdk(monkeypatch) -> tuple[AIClient, MagicMock]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AIClient(api_key="test-key", model="claude-sonnet-4-5")
    mock_create = MagicMock()
    client._client.messages.create = mock_create  # patch at the SDK boundary
    return client, mock_create


def test_missing_api_key_fails_fast():
    with pytest.raises(AIConfigurationError):
        AIClient(api_key="")


def test_complete_returns_text_from_a_successful_call(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response("hello learner")

    result = client.complete("system prompt", "user prompt")

    assert result == "hello learner"
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "claude-sonnet-4-5"
    assert kwargs["system"] == "system prompt"
    assert kwargs["messages"][0]["content"][-1] == {"type": "text", "text": "user prompt"}


def test_transient_error_is_retried_then_succeeds(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)  # don't actually wait in tests
    mock_create.side_effect = [_rate_limit_error(), _fake_response("worked on attempt 2")]

    result = client.complete("sys", "user", max_retries=3)

    assert result == "worked on attempt 2"
    assert mock_create.call_count == 2


def test_non_retryable_error_raises_immediately(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.side_effect = _bad_request_error()

    with pytest.raises(AIClientError):
        client.complete("sys", "user", max_retries=3)

    mock_create.assert_called_once()  # no retries burned on a non-retryable error


def test_exhausting_retries_raises_ai_client_error(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    mock_create.side_effect = _rate_limit_error()

    with pytest.raises(AIClientError):
        client.complete("sys", "user", max_retries=2)

    assert mock_create.call_count == 2


def test_complete_json_parses_clean_response(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response('{"is_decorative": false, "short_alt_text": "a graph"}')

    result = client.complete_json("sys", "user")

    assert result == {"is_decorative": False, "short_alt_text": "a graph"}
    assert mock_create.call_count == 1


def test_complete_json_strips_markdown_fences(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response('```json\n{"ok": true}\n```')

    assert client.complete_json("sys", "user") == {"ok": True}


def test_complete_json_retries_once_on_malformed_response_then_recovers(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.side_effect = [
        _fake_response("sure, here's some JSON: {not valid"),
        _fake_response('{"concepts": []}'),
    ]

    result = client.complete_json("sys", "user")

    assert result == {"concepts": []}
    assert mock_create.call_count == 2


def test_complete_json_returns_none_if_still_unparseable_after_retry(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response("not json at all")

    result = client.complete_json("sys", "user")

    assert result is None
    assert mock_create.call_count == 2  # first try + the one stricter retry


def test_images_are_attached_as_base64_content_blocks(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response('{"short_alt_text": "a diagram"}')
    image = ImageInput(data=b"\x89PNG fake bytes", media_type="image/png")

    client.complete_json("describe this image", "context: photosynthesis diagram", images=[image])

    _, kwargs = mock_create.call_args
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[-1]["type"] == "text"


def test_ping_returns_true_on_a_working_call(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.return_value = _fake_response("pong")

    assert client.ping() is True


def test_ping_returns_false_instead_of_raising_when_the_call_fails(monkeypatch):
    client, mock_create = _client_with_mocked_sdk(monkeypatch)
    mock_create.side_effect = _bad_request_error()

    assert client.ping() is False
