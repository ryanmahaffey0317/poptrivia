from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.discord_bot import GeneratePromptView
from poptrivia.models import MovieStatus


def _make_settings(tmp_path: Path, approved: set[int]) -> Settings:
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
        discord_bot_token="x",
        discord_approved_users=approved,
        config_dir=config_dir,
        tracks_dir=config_dir / "tracks",
        cache_dir=config_dir / "cache",
        tracked_list_path=config_dir / "tracked.txt",
    )


@pytest.fixture
async def db_and_settings(tmp_path: Path):
    settings = _make_settings(tmp_path, approved={123456})
    db = Database(settings.db_path)
    await db.connect()
    yield db, settings
    await db.close()


def _make_interaction(user_id: int) -> MagicMock:
    """Build a discord.Interaction-like mock with the right shape."""
    interaction = MagicMock()
    interaction.user.id = user_id
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    interaction.message = MagicMock()
    return interaction


async def test_generate_button_requires_approved_user(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(plex_guid="plex://movie/abc", title="The Shining", imdb_id="tt0081505")

    view = GeneratePromptView(
        plex_guid="plex://movie/abc",
        imdb_id="tt0081505",
        title="The Shining",
        year=1980,
        settings=settings,
        db=db,
    )
    interaction = _make_interaction(user_id=999999)  # NOT in approved set

    # discord.py wires the @button-decorated method into view.children at
    # View instantiation. Invoke via the bound Button.callback so the
    # decorator's signature handling stays in our path.
    await view.children[0].callback(interaction)

    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    assert "Not authorized" in args[0]
    assert kwargs.get("ephemeral") is True

    # tracked.txt should not have been touched
    assert not settings.tracked_list_path.exists() or \
        "tt0081505" not in settings.tracked_list_path.read_text(encoding="utf-8")


async def test_generate_button_appends_and_queues(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(plex_guid="plex://movie/abc", title="The Shining", imdb_id="tt0081505")

    view = GeneratePromptView(
        plex_guid="plex://movie/abc",
        imdb_id="tt0081505",
        title="The Shining",
        year=1980,
        settings=settings,
        db=db,
    )
    interaction = _make_interaction(user_id=123456)  # approved

    # discord.py wires the @button-decorated method into view.children at
    # View instantiation. Invoke via the bound Button.callback so the
    # decorator's signature handling stays in our path.
    await view.children[0].callback(interaction)

    # tracked.txt was created and contains the IMDB id
    text = settings.tracked_list_path.read_text(encoding="utf-8")
    assert "tt0081505" in text
    assert "The Shining (1980)" in text

    # Movie status moved to QUEUED + a job was enqueued
    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None
    assert movie.status == MovieStatus.QUEUED
    job = await db.next_pending_job()
    assert job is not None and job.plex_guid == "plex://movie/abc"

    # Message was edited (not sent fresh)
    interaction.response.edit_message.assert_awaited_once()


async def test_dismiss_button_sets_dismissed_at(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(plex_guid="plex://movie/abc", title="X", imdb_id="tt0001")

    view = GeneratePromptView(
        plex_guid="plex://movie/abc",
        imdb_id="tt0001",
        title="X",
        year=2000,
        settings=settings,
        db=db,
    )
    interaction = _make_interaction(user_id=123456)

    await view.children[1].callback(interaction)

    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None
    assert movie.dismissed_at is not None
    interaction.response.edit_message.assert_awaited_once()
    args, kwargs = interaction.response.edit_message.call_args
    assert "Skipped" in kwargs.get("content", "") or "Skipped" in (args[0] if args else "")


async def test_dismiss_button_unapproved_user_rejected(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(plex_guid="plex://movie/abc", title="X", imdb_id="tt0001")

    view = GeneratePromptView(
        plex_guid="plex://movie/abc",
        imdb_id="tt0001",
        title="X",
        year=2000,
        settings=settings,
        db=db,
    )
    interaction = _make_interaction(user_id=999999)  # not approved

    await view.children[1].callback(interaction)

    # dismissed_at must NOT have been set
    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None
    assert movie.dismissed_at is None
    interaction.response.send_message.assert_awaited_once()
