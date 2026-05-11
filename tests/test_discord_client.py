from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from poptrivia.discord_client import (
    CATEGORY_COLORS,
    DiscordClient,
    color_for,
    pretty_category,
)
from poptrivia.models import Movie, MovieStatus, TriviaCard


def _card(category: str = "production") -> TriviaCard:
    return TriviaCard(
        id="c1",
        timestamp_ms=100,
        text="Sample fact.",
        category=category,  # type: ignore[arg-type]
        interest_level=4,
        source_fact_id="f1",
    )


def _movie() -> Movie:
    return Movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        year=1980,
        status=MovieStatus.READY,
    )


def _make_client(handler) -> DiscordClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return DiscordClient(
        "https://discord/card",
        "https://discord/system",
        client=http,
    )


async def test_post_card_sends_expected_embed() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(204)

    client = _make_client(handler)
    try:
        await client.post_card(_card(), _movie())
    finally:
        await client.aclose()

    assert len(captured) == 1
    body = captured[0]
    assert "embeds" in body and len(body["embeds"]) == 1
    embed = body["embeds"][0]
    assert embed["description"].startswith("💭 ")
    assert embed["color"] == CATEGORY_COLORS["production"]
    assert embed["footer"]["text"] == "Production · The Shining (1980)"


async def test_post_track_active_format() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(204)

    client = _make_client(handler)
    try:
        await client.post_track_active(_movie(), card_count=42)
    finally:
        await client.aclose()

    assert (
        captured[0]["content"]
        == "🎬 **The Shining** (1980) — Trivia track active · 42 cards"
    )


async def test_post_system_uses_system_webhook() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(204)

    client = _make_client(handler)
    try:
        await client.post_system("hello")
    finally:
        await client.aclose()

    assert seen_urls == ["https://discord/system"]


async def test_retry_on_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={"retry_after": 0.01},
                headers={"Retry-After": "0"},
            )
        return httpx.Response(204)

    client = _make_client(handler)
    try:
        await client.post_card(_card(), _movie())
    finally:
        await client.aclose()

    assert calls["n"] == 2


def test_pretty_category_known() -> None:
    assert pretty_category("easter_egg") == "Easter Egg"


def test_pretty_category_unknown_fallback() -> None:
    assert pretty_category("unknown_bucket") == "Unknown Bucket"


def test_color_for_known_and_unknown() -> None:
    assert color_for("production") == 0x4A7C7C
    assert color_for("not_a_real_cat") == 0x808080
