#!/usr/bin/env python3
"""Fire a sample trivia card + system message to verify the Discord webhook.

Reads webhook URLs from environment (or .env). Run from inside the container
or any environment with poptrivia installed:

    python scripts/test_discord.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running directly from the repo without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poptrivia.config import get_settings  # noqa: E402
from poptrivia.discord_client import DiscordClient  # noqa: E402
from poptrivia.models import Movie, MovieStatus, TriviaCard  # noqa: E402


async def main() -> None:
    settings = get_settings()
    client = DiscordClient(settings.discord_webhook_url, settings.system_webhook_url)

    movie = Movie(
        plex_guid="plex://movie/test",
        title="The Shining",
        year=1980,
        status=MovieStatus.READY,
    )
    card = TriviaCard(
        id="card_test",
        timestamp_ms=600000,
        text=(
            "The blood pouring out of the elevators required 10,000 gallons of fake "
            "blood and took a full year to film across multiple takes."
        ),
        category="production",
        interest_level=5,
        source_fact_id="fact_test",
        anchor_evidence="elevator scene",
    )

    try:
        await client.post_track_active(movie, card_count=42)
        await client.post_card(card, movie)
        await client.post_system("✅ poptrivia Discord test message")
        print("Sent: track-active header, sample card, system message")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
