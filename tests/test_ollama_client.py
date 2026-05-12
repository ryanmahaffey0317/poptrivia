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
    return OllamaClient("http://ollama:11434", "qwen2.5:32b", client=http)


async def test_generate_string_no_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5:32b"
        assert body["prompt"] == "hello"
        return httpx.Response(200, json={"response": "hi there"})

    c = _make_client(handler)
    try:
        out = await c.generate("hello")
    finally:
        await c.aclose()
    assert out == "hi there"


async def test_generate_with_schema_returns_parsed_json() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == schema
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


async def test_schema_empty_response_falls_back_to_json_mode() -> None:
    """qwen3.6:27b returns empty on format=<schema>. The client should
    automatically retry with format='json' and parse the result."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        # First call: schema mode → return empty (simulating qwen3.6's bug)
        # Second call: json mode → return real JSON
        if isinstance(body.get("format"), dict):
            return httpx.Response(200, json={"response": ""})
        if body.get("format") == "json":
            return httpx.Response(200, json={"response": '{"x": 42}'})
        return httpx.Response(500, text="unexpected format")

    c = _make_client(handler)
    try:
        out = await c.generate("foo", schema=schema)
    finally:
        await c.aclose()

    assert out == {"x": 42}
    # Verify both attempts actually happened in the right order.
    assert len(calls) == 2
    assert isinstance(calls[0]["format"], dict)  # first attempt: schema
    assert calls[1]["format"] == "json"          # fallback: json mode


async def test_schema_mode_success_does_not_retry() -> None:
    """When schema mode works, we should NOT issue a second request."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"response": '{"x": 7}'})

    c = _make_client(handler)
    try:
        out = await c.generate("foo", schema=schema)
    finally:
        await c.aclose()

    assert out == {"x": 7}
    assert calls["n"] == 1  # exactly one round-trip — no fallback fired


async def test_no_schema_does_not_trigger_fallback() -> None:
    """When the caller didn't pass a schema, an empty response is a valid
    answer; we should NOT retry as json mode."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"response": ""})

    c = _make_client(handler)
    try:
        out = await c.generate("foo")  # no schema
    finally:
        await c.aclose()

    assert out == ""
    assert calls["n"] == 1
