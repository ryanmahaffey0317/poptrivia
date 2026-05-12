from __future__ import annotations

import json

import httpx
import pytest

from poptrivia.prep.llm.client import (
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
)


def _make_client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return OllamaClient("http://ollama:11434", "qwen3:32b", client=http)


async def test_generate_string_no_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen3:32b"
        assert body["prompt"] == "hello"
        # No schema given → no format param sent
        assert "format" not in body
        # Thinking explicitly disabled (qwen3.6:27b would otherwise return
        # empty when format=json is set; harmless on non-thinking models).
        assert body["think"] is False
        return httpx.Response(200, json={"response": "hi there"})

    c = _make_client(handler)
    try:
        out = await c.generate("hello")
    finally:
        await c.aclose()
    assert out == "hi there"


async def test_generate_with_schema_uses_json_format_mode() -> None:
    """When a schema is provided we request format='json' from Ollama
    (NOT format=<schema>, which qwen3.6 doesn't honor). The schema is
    enforced downstream via pydantic, not by Ollama."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == "json"  # NOT a schema dict
        assert body["think"] is False    # thinking suppressed for JSON outputs
        return httpx.Response(200, json={"response": '{"x": 42}'})

    c = _make_client(handler)
    try:
        out = await c.generate("foo", schema=schema)
    finally:
        await c.aclose()
    assert out == {"x": 42}


async def test_generate_raises_on_invalid_json_when_schema_given() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "not json at all"})

    c = _make_client(handler)
    try:
        with pytest.raises(OllamaError):
            await c.generate("foo", schema={"type": "object"})
    finally:
        await c.aclose()


async def test_connection_error_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    c = _make_client(handler)
    try:
        with pytest.raises(OllamaUnavailable):
            await c.generate("foo")
    finally:
        await c.aclose()


async def test_health_probe_success_and_failure() -> None:
    state = {"ok": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["ok"]:
            return httpx.Response(200, json={"models": []})
        raise httpx.ConnectError("nope", request=request)

    c = _make_client(handler)
    try:
        assert await c.health() is True
        state["ok"] = False
        assert await c.health() is False
    finally:
        await c.aclose()


async def test_generate_no_schema_does_not_send_format() -> None:
    """Plain text completions don't constrain the response format."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"})

    c = _make_client(handler)
    try:
        await c.generate("just text", schema=None)
    finally:
        await c.aclose()
    assert "format" not in captured[0]
