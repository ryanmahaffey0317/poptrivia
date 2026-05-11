from __future__ import annotations

from pathlib import Path

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import JobStatus


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
    d = Database(settings.db_path)
    await d.connect()
    yield d
    await d.close()


async def test_reset_stale_running_jobs_promotes_running_to_pending(db) -> None:
    await db.upsert_movie(plex_guid="plex://movie/a", title="A")
    await db.upsert_movie(plex_guid="plex://movie/b", title="B")

    # Two jobs: one stuck in 'running' (zombie), one already 'succeeded'.
    job_a = await db.enqueue_job("plex://movie/a")
    job_b = await db.enqueue_job("plex://movie/b")
    await db.mark_job_running(job_a)
    await db.mark_job_succeeded(job_b)

    n = await db.reset_stale_running_jobs()
    assert n == 1  # only the running job got reset

    # job_a should now be pending again
    pending = await db.next_pending_job()
    assert pending is not None and pending.id == job_a
    assert "reset on startup" in (pending.last_error or "")


async def test_reset_stale_running_jobs_noop_when_no_running(db) -> None:
    await db.upsert_movie(plex_guid="plex://movie/a", title="A")
    job_a = await db.enqueue_job("plex://movie/a")
    await db.mark_job_succeeded(job_a)

    n = await db.reset_stale_running_jobs()
    assert n == 0
