from __future__ import annotations

from pathlib import Path

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database


def _make_settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return Settings(  # type: ignore[call-arg]
        plex_url="x",
        plex_token="x",
        tautulli_url="http://x",
        tautulli_api_key="x",
        monitored_users={"alice"},
        ollama_url="http://x",
        discord_webhook_url="http://x",
        config_dir=config_dir,
        tracks_dir=config_dir / "tracks",
        cache_dir=config_dir / "cache",
    )


@pytest.fixture
async def db(tmp_path: Path):
    settings = _make_settings(tmp_path)
    db = Database(settings.db_path)
    await db.connect()
    yield db
    await db.close()


async def test_movie_dismissed_at_starts_null(db) -> None:
    movie = await db.upsert_movie(plex_guid="plex://movie/abc", title="X")
    assert movie.dismissed_at is None


async def test_set_dismissed_populates_timestamp(db) -> None:
    await db.upsert_movie(plex_guid="plex://movie/abc", title="X")
    await db.set_dismissed("plex://movie/abc")
    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None
    assert movie.dismissed_at is not None
