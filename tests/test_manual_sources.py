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


async def test_enqueue_job_stores_manual_sources_as_json(db) -> None:
    await db.upsert_movie(plex_guid="plex://movie/abc", title="X")
    job_id = await db.enqueue_job(
        "plex://movie/abc", manual_sources=["/a.txt", "/b.txt"]
    )

    job = await db.next_pending_job()
    assert job is not None
    assert job.id == job_id
    assert job.manual_sources == ["/a.txt", "/b.txt"]


async def test_enqueue_job_without_manual_sources_is_empty_list(db) -> None:
    await db.upsert_movie(plex_guid="plex://movie/abc", title="X")
    await db.enqueue_job("plex://movie/abc")
    job = await db.next_pending_job()
    assert job is not None
    assert job.manual_sources == []


async def test_pipeline_gather_sources_skips_imdb_when_manual_supplied(
    tmp_path: Path, monkeypatch
) -> None:
    """When manual_sources is non-empty, IMDB network scrape is bypassed."""
    from poptrivia.models import Movie, MovieStatus
    from poptrivia.prep import pipeline
    from poptrivia.prep.sources import imdb as imdb_source

    src = tmp_path / "trivia.txt"
    src.write_text("Kubrick demanded 127 takes of the baseball-bat scene.\n",
                   encoding="utf-8")

    imdb_called = {"n": 0}

    async def fake_imdb_trivia(*_args, **_kwargs):
        imdb_called["n"] += 1
        return []

    monkeypatch.setattr(imdb_source, "fetch_trivia", fake_imdb_trivia)
    monkeypatch.setattr(imdb_source, "fetch_goofs", fake_imdb_trivia)

    # Wikipedia / TMDB fail closed in this test — we only care about IMDB
    # being skipped vs called.
    async def fake_wiki(*_args, **_kwargs):
        raise RuntimeError("wikipedia not under test")

    monkeypatch.setattr(
        pipeline.wiki_source, "fetch_article", fake_wiki
    )

    movie = Movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        imdb_id="tt0081505",
        status=MovieStatus.NOT_STARTED,
    )
    settings = _make_settings(tmp_path)

    items = await pipeline._gather_sources(
        movie=movie, settings=settings, manual_sources=[src]
    )
    assert imdb_called["n"] == 0  # IMDB skipped
    assert any(
        i.source == "imdb_trivia" and "Kubrick" in i.text for i in items
    ), items
