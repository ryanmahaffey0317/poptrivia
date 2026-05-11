from __future__ import annotations

import json
from pathlib import Path

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import MovieStatus
from poptrivia.prep import queue as queue_mod
from poptrivia.prep.llm.client import OllamaUnavailable
from poptrivia.prep.queue import PrepWorker


class FakeDiscord:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def post_system(self, text: str) -> None:
        self.messages.append(text)


def _make_settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    tracks_dir = config_dir / "tracks"
    cache_dir = config_dir / "cache"
    for d in (config_dir, tracks_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return Settings(  # type: ignore[call-arg]
        plex_url="x",
        plex_token="x",
        tautulli_url="http://x",
        tautulli_api_key="x",
        monitored_users={"alice"},
        ollama_url="http://x",
        discord_webhook_url="http://x",
        config_dir=config_dir,
        tracks_dir=tracks_dir,
        cache_dir=cache_dir,
    )


@pytest.fixture
async def db_and_settings(tmp_path: Path):
    settings = _make_settings(tmp_path)
    db = Database(settings.db_path)
    await db.connect()
    yield db, settings
    await db.close()


async def test_empty_queue_returns_false(db_and_settings, monkeypatch) -> None:
    db, settings = db_and_settings
    worker = PrepWorker(settings=settings, db=db, discord=FakeDiscord())  # type: ignore[arg-type]
    handled = await worker._tick()
    assert handled is False


async def test_successful_prep_marks_ready_and_notifies(
    db_and_settings, monkeypatch
) -> None:
    db, settings = db_and_settings
    discord = FakeDiscord()
    worker = PrepWorker(settings=settings, db=db, discord=discord)  # type: ignore[arg-type]

    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        year=1980,
        imdb_id="tt0081505",
    )
    await db.enqueue_job("plex://movie/abc")

    async def fake_prepare(*, movie, settings, db):
        track_path = settings.tracks_dir / "plex___movie_abc.json"
        track_path.write_text(json.dumps({"cards": [{"id": "x"}] * 17}))
        await db.set_movie_status(
            movie.plex_guid, MovieStatus.READY, track_path=str(track_path)
        )
        return track_path

    monkeypatch.setattr(queue_mod, "prepare_movie", fake_prepare)

    handled = await worker._tick()
    assert handled is True

    job = await db.next_pending_job()
    assert job is None  # nothing pending anymore
    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None and movie.status == MovieStatus.READY
    assert len(discord.messages) == 1
    assert "✅" in discord.messages[0] and "17 cards" in discord.messages[0]


async def test_ollama_unavailable_requeues(
    db_and_settings, monkeypatch
) -> None:
    db, settings = db_and_settings
    discord = FakeDiscord()
    worker = PrepWorker(settings=settings, db=db, discord=discord)  # type: ignore[arg-type]
    # Don't actually sleep for the backoff.
    monkeypatch.setattr(queue_mod, "OLLAMA_BACKOFF_SECONDS", 0)

    await db.upsert_movie(plex_guid="plex://movie/a", title="A", year=2000)
    await db.enqueue_job("plex://movie/a")

    async def raises(*_, **__):
        raise OllamaUnavailable("connection refused")

    monkeypatch.setattr(queue_mod, "prepare_movie", raises)

    handled = await worker._tick()
    assert handled is True

    # Job should be back to PENDING.
    job = await db.next_pending_job()
    assert job is not None
    assert job.plex_guid == "plex://movie/a"
    assert "Ollama unavailable" in (job.last_error or "")
    assert discord.messages == []  # no failure notification for transient errors


async def test_other_exception_marks_failed_and_notifies(
    db_and_settings, monkeypatch
) -> None:
    db, settings = db_and_settings
    discord = FakeDiscord()
    worker = PrepWorker(settings=settings, db=db, discord=discord)  # type: ignore[arg-type]

    await db.upsert_movie(plex_guid="plex://movie/b", title="B", year=2001)
    await db.enqueue_job("plex://movie/b")

    async def raises(*_, **__):
        raise RuntimeError("no subtitles")

    monkeypatch.setattr(queue_mod, "prepare_movie", raises)

    handled = await worker._tick()
    assert handled is True

    movie = await db.get_movie("plex://movie/b")
    assert movie is not None and movie.status == MovieStatus.FAILED
    assert "no subtitles" in (movie.error_message or "")
    assert len(discord.messages) == 1
    assert "❌" in discord.messages[0]
