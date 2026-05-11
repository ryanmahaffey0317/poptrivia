from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from poptrivia.models import Category, Movie, TriviaCard

log = logging.getLogger("poptrivia.discord")


# Hand-picked palette per category. Hex values are integers because that's
# the format Discord wants in embed bodies.
CATEGORY_COLORS: dict[str, int] = {
    "production": 0x4A7C7C,
    "casting": 0xC65D7B,
    "cinematography": 0x6B5B95,
    "historical_context": 0x8B6F47,
    "easter_egg": 0xF7B538,
    "cultural_impact": 0xE63946,
    "goof": 0xF77F00,
    "cut_content": 0x577590,
    "cast_biography": 0x9B5DE5,
    "score_music": 0x06A77D,
}

CATEGORY_PRETTY: dict[str, str] = {
    "production": "Production",
    "casting": "Casting",
    "cinematography": "Cinematography",
    "historical_context": "Historical Context",
    "easter_egg": "Easter Egg",
    "cultural_impact": "Cultural Impact",
    "goof": "Goof",
    "cut_content": "Cut Content",
    "cast_biography": "Cast Biography",
    "score_music": "Score & Music",
}


def pretty_category(category: Category | str) -> str:
    return CATEGORY_PRETTY.get(str(category), str(category).replace("_", " ").title())


def color_for(category: Category | str) -> int:
    return CATEGORY_COLORS.get(str(category), 0x808080)


class DiscordClient:
    """Thin Discord webhook client.

    Two posting methods:
      - post_card(card, movie): trivia card embed (the live UX)
      - post_system(text):      plain text to the system channel
                                (separate webhook if configured)

    Rate-limit handling: we read the X-RateLimit-Remaining / Reset-After
    headers and proactively pause when we're near zero. On a 429, we honour
    Retry-After and resend once.
    """

    def __init__(
        self,
        card_webhook: str,
        system_webhook: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        if not card_webhook:
            raise ValueError("DiscordClient requires a card webhook URL")
        self.card_webhook = card_webhook
        self.system_webhook = system_webhook or card_webhook
        self._client = client or httpx.AsyncClient(timeout=15.0)
        self._owns_client = client is None
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ─── public API ─────────────────────────────────────────────────

    async def post_card(self, card: TriviaCard, movie: Movie) -> None:
        year = f" ({movie.year})" if movie.year else ""
        footer_text = f"{pretty_category(card.category)} · {movie.title}{year}"
        embed = {
            "description": f"💭 {card.text}",
            "color": color_for(card.category),
            "footer": {"text": footer_text},
        }
        await self._post(self.card_webhook, {"embeds": [embed]})

    async def post_track_active(self, movie: Movie, card_count: int) -> None:
        year = f" ({movie.year})" if movie.year else ""
        text = (
            f"🎬 **{movie.title}**{year} — Trivia track active · "
            f"{card_count} card{'s' if card_count != 1 else ''}"
        )
        await self._post(self.card_webhook, {"content": text})

    async def post_session_summary(
        self, movie: Movie, fired: int, total: int
    ) -> None:
        year = f" ({movie.year})" if movie.year else ""
        text = (
            f"🏁 **{movie.title}**{year} — session ended · "
            f"{fired} of {total} cards fired"
        )
        await self._post(self.card_webhook, {"content": text})

    async def post_system(self, text: str) -> None:
        await self._post(self.system_webhook, {"content": text})

    # ─── internals ──────────────────────────────────────────────────

    async def _post(self, url: str, body: dict[str, Any]) -> None:
        async with self._lock:
            response = await self._post_with_retry(url, body)
            self._maybe_pause_for_ratelimit(response)

    async def _post_with_retry(self, url: str, body: dict[str, Any]) -> httpx.Response:
        for attempt in range(2):
            try:
                response = await self._client.post(url, json=body)
            except httpx.HTTPError as e:
                log.warning("Discord POST failed (attempt %d): %s", attempt + 1, e)
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                raise

            if response.status_code == 429:
                retry_after = _retry_after_seconds(response)
                log.warning(
                    "Discord rate limited, sleeping %.2fs before retry", retry_after
                )
                await asyncio.sleep(retry_after)
                continue
            if response.status_code >= 400:
                log.error(
                    "Discord POST failed with %d: %s",
                    response.status_code,
                    response.text[:500],
                )
                response.raise_for_status()
            return response
        raise RuntimeError("Discord POST exhausted retries")

    def _maybe_pause_for_ratelimit(self, response: httpx.Response) -> None:
        """If Discord says we have 0 requests remaining, the next call
        should sleep through the bucket reset. We just stash the deadline
        on the client and consult it on the next call."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_after = response.headers.get("X-RateLimit-Reset-After")
        if remaining == "0" and reset_after is not None:
            try:
                delay = float(reset_after)
            except ValueError:
                return
            log.debug("Discord bucket exhausted, will pause %.2fs", delay)
            # Cheapest possible implementation: hold the lock for the
            # remaining bucket duration. Caller already released it; we
            # schedule a brief sleep so the next call observes it.
            asyncio.create_task(self._hold_lock(delay))

    async def _hold_lock(self, seconds: float) -> None:
        async with self._lock:
            await asyncio.sleep(seconds)


def _retry_after_seconds(response: httpx.Response) -> float:
    # Discord puts retry_after in both the JSON body (seconds, float) and
    # the Retry-After header (seconds, integer). Body is more precise.
    try:
        body = response.json()
        if isinstance(body, dict) and "retry_after" in body:
            return float(body["retry_after"])
    except ValueError:
        pass
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return 1.0
