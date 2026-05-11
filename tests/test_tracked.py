from __future__ import annotations

from pathlib import Path

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import MovieStatus
from poptrivia.tracked import ensure_template, is_tracked, read_tracked
from poptrivia.tracked_poll import TrackedPoller


def test_read_tracked_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_tracked(tmp_path / "nope.txt") == set()


def test_read_tracked_handles_comments_and_inline_comments(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    p.write_text(
        """
# A full-line comment.

tt0081505     # The Shining (1980)
tt1478338# inline comment no space
   plex://movie/abc

# tt9999999  ← commented out, should not be tracked
""",
        encoding="utf-8",
    )
    out = read_tracked(p)
    assert out == {"tt0081505", "tt1478338", "plex://movie/abc"}


def test_is_tracked_matches_imdb_or_plex_guid_case_insensitive() -> None:
    tracked = {"tt0081505", "plex://movie/abc"}
    assert is_tracked(tracked=tracked, imdb_id="tt0081505", plex_guid="plex://movie/zzz")
    assert is_tracked(tracked=tracked, imdb_id="TT0081505", plex_guid=None)
    assert is_tracked(tracked=tracked, imdb_id=None, plex_guid="plex://movie/abc")
    assert is_tracked(tracked=tracked, imdb_id=None, plex_guid="PLEX://MOVIE/ABC")
    assert not is_tracked(tracked=tracked, imdb_id="tt9999999", plex_guid="plex://movie/zzz")
    assert not is_tracked(tracked=tracked, imdb_id=None, plex_guid=None)


def test_ensure_template_creates_file_with_instructions(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    ensure_template(p)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "tracked-movies list" in text
    assert "tt0081505" in text  # example reference present


def test_ensure_template_does_not_clobber_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "tracked.txt"
    p.write_text("tt0081505\n", encoding="utf-8")
    ensure_template(p)
    assert p.read_text(encoding="utf-8") == "tt0081505\n"


# ─── TrackedPoller integration ───────────────────────────────────────────


def _make_settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "tracks").mkdir(exist_ok=True)
    (config_dir / "cache").mkdir(exist_ok=True)
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
        tracked_list_path=config_dir / "tracked.txt",
    )


@pytest.fixture
async def db_and_settings(tmp_path: Path):
    settings = _make_settings(tmp_path)
    db = Database(settings.db_path)
    await db.connect()
    yield db, settings
    await db.close()


async def test_poller_queues_tracked_movie_with_captured_metadata(db_and_settings) -> None:
    db, settings = db_and_settings
    # Captured metadata from a previous play:
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        year=1980,
        imdb_id="tt0081505",
    )
    settings.tracked_list_path.write_text("tt0081505\n", encoding="utf-8")

    queued = await TrackedPoller(settings=settings, db=db).poll_once()
    assert queued == 1

    job = await db.next_pending_job()
    assert job is not None and job.plex_guid == "plex://movie/abc"
    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None and movie.status == MovieStatus.QUEUED


async def test_poller_skips_already_ready_or_in_progress(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/ready",
        title="A",
        imdb_id="tt000001",
        status=MovieStatus.READY,
    )
    await db.upsert_movie(
        plex_guid="plex://movie/gen",
        title="B",
        imdb_id="tt000002",
        status=MovieStatus.GENERATING,
    )
    settings.tracked_list_path.write_text("tt000001\ntt000002\n", encoding="utf-8")

    queued = await TrackedPoller(settings=settings, db=db).poll_once()
    assert queued == 0


async def test_poller_does_not_duplicate_when_active_job_exists(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        imdb_id="tt0081505",
    )
    await db.enqueue_job("plex://movie/abc")  # already pending
    settings.tracked_list_path.write_text("tt0081505\n", encoding="utf-8")

    queued = await TrackedPoller(settings=settings, db=db).poll_once()
    assert queued == 0


async def test_poller_logs_unmatched_entries_without_crashing(
    db_and_settings, caplog
) -> None:
    db, settings = db_and_settings
    # tracked entry has never been played → no DB row yet
    settings.tracked_list_path.write_text("tt9999999\n", encoding="utf-8")

    queued = await TrackedPoller(settings=settings, db=db).poll_once()
    assert queued == 0  # nothing to queue (no captured metadata)
    # No crash, just an info log; existing movie rows untouched.
