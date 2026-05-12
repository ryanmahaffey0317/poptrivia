from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.discord_bot import BotDiscordSender, GeneratePromptView, PoptriviaBot
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


# ─── slash command handler tests ─────────────────────────────────────────


def _make_bot(db_and_settings) -> PoptriviaBot:
    """Construct a bot for handler-level tests. We don't connect to Discord;
    we just exercise the _handle_* methods directly with a mocked interaction."""
    db, settings = db_and_settings
    return PoptriviaBot(settings=settings, db=db)


async def test_untrack_removes_existing_entry(db_and_settings) -> None:
    db, settings = db_and_settings
    settings.tracked_list_path.write_text(
        "tt0081505   # The Shining\ntt1478338   # Bridesmaids\n",
        encoding="utf-8",
    )
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_untrack_command(interaction, "tt0081505")

    content = settings.tracked_list_path.read_text(encoding="utf-8")
    assert "tt0081505" not in content
    assert "tt1478338" in content
    interaction.response.send_message.assert_awaited_once()
    args, _kwargs = interaction.response.send_message.call_args
    assert "Removed" in args[0] and "tt0081505" in args[0]


async def test_untrack_not_present_returns_ephemeral(db_and_settings) -> None:
    db, settings = db_and_settings
    settings.tracked_list_path.write_text("tt1478338\n", encoding="utf-8")
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_untrack_command(interaction, "tt0000000")

    interaction.response.send_message.assert_awaited_once()
    _args, kwargs = interaction.response.send_message.call_args
    assert kwargs.get("ephemeral") is True


async def test_untrack_unauthorized_rejected(db_and_settings) -> None:
    db, settings = db_and_settings
    settings.tracked_list_path.write_text("tt0081505\n", encoding="utf-8")
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(999999)  # not approved

    await bot._handle_untrack_command(interaction, "tt0081505")

    # File untouched
    assert "tt0081505" in settings.tracked_list_path.read_text(encoding="utf-8")
    args, kwargs = interaction.response.send_message.call_args
    assert "Not authorized" in args[0]
    assert kwargs.get("ephemeral") is True


async def test_list_command_empty_tracked_file(db_and_settings) -> None:
    db, settings = db_and_settings
    # Intentionally no tracked.txt
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_list_command(interaction)

    interaction.response.send_message.assert_awaited_once()
    args, _kwargs = interaction.response.send_message.call_args
    assert "empty" in args[0].lower()


async def test_list_command_renders_status_badges(db_and_settings) -> None:
    db, settings = db_and_settings
    settings.tracked_list_path.write_text(
        "tt0081505   # The Shining\ntt1478338   # Bridesmaids\n",
        encoding="utf-8",
    )
    await db.upsert_movie(
        plex_guid="plex://movie/shining",
        title="The Shining",
        imdb_id="tt0081505",
        status=MovieStatus.READY,
    )
    await db.upsert_movie(
        plex_guid="plex://movie/bridesmaids",
        title="Bridesmaids",
        imdb_id="tt1478338",
        status=MovieStatus.FAILED,
    )
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_list_command(interaction)

    interaction.response.send_message.assert_awaited_once()
    _args, kwargs = interaction.response.send_message.call_args
    embed = kwargs.get("embed")
    assert embed is not None
    body = embed.description
    assert "tt0081505" in body and "ready" in body
    assert "tt1478338" in body and "failed" in body


async def test_show_command_returns_top_cards(db_and_settings, tmp_path: Path) -> None:
    db, settings = db_and_settings
    track_path = tmp_path / "track.json"
    track_path.write_text(
        json.dumps({
            "plex_guid": "plex://movie/abc",
            "title": "The Shining",
            "year": 1980,
            "generated_at": "2026-05-01T00:00:00+00:00",
            "model": "qwen3:32b",
            "cards": [
                {
                    "id": "card_001",
                    "timestamp_ms": 60_000,
                    "text": "Kubrick demanded 127 takes.",
                    "category": "production",
                    "interest_level": 5,
                    "source_fact_id": "f001",
                    "anchor_evidence": "x",
                },
                {
                    "id": "card_002",
                    "timestamp_ms": 120_000,
                    "text": "Stephen King disliked the adaptation.",
                    "category": "cultural_impact",
                    "interest_level": 4,
                    "source_fact_id": "f002",
                    "anchor_evidence": "x",
                },
            ],
        }),
        encoding="utf-8",
    )
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        year=1980,
        imdb_id="tt0081505",
        status=MovieStatus.READY,
        track_path=str(track_path),
    )
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_show_command(interaction, "tt0081505")

    interaction.response.send_message.assert_awaited_once()
    _args, kwargs = interaction.response.send_message.call_args
    embed = kwargs.get("embed")
    assert embed is not None
    body = embed.description
    assert "Kubrick demanded 127 takes" in body
    assert "Stephen King" in body


async def test_show_command_movie_not_found(db_and_settings) -> None:
    db, settings = db_and_settings
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_show_command(interaction, "tt9999999")

    args, kwargs = interaction.response.send_message.call_args
    assert "No movie in the DB" in args[0]
    assert kwargs.get("ephemeral") is True


async def test_regenerate_command_enqueues_job(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="The Shining",
        imdb_id="tt0081505",
        status=MovieStatus.READY,
    )
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_regenerate_command(interaction, "tt0081505")

    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None and movie.status == MovieStatus.QUEUED
    job = await db.next_pending_job()
    assert job is not None and job.plex_guid == "plex://movie/abc"


async def test_regenerate_command_rejects_if_job_already_active(
    db_and_settings,
) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="X",
        imdb_id="tt0081505",
        status=MovieStatus.READY,
    )
    await db.enqueue_job("plex://movie/abc")  # already pending

    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_regenerate_command(interaction, "tt0081505")

    args, kwargs = interaction.response.send_message.call_args
    assert "already pending or running" in args[0]
    assert kwargs.get("ephemeral") is True


async def test_dismiss_clear_command_resets_flag(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="X",
        imdb_id="tt0081505",
    )
    await db.set_dismissed("plex://movie/abc")
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_dismiss_clear_command(interaction, "tt0081505")

    movie = await db.get_movie("plex://movie/abc")
    assert movie is not None
    assert movie.dismissed_at is None


# ─── BotDiscordSender channel routing ─────────────────────────────────────


async def test_bot_sender_uses_cards_channel_for_cards(db_and_settings) -> None:
    """post_card routes to discord_cards_channel_id."""
    db, settings = db_and_settings
    settings.discord_bot_channel_id = 100
    settings.discord_cards_channel_id = 200
    settings.discord_system_channel_id = 300

    bot = _make_bot(db_and_settings)
    sender = BotDiscordSender(bot=bot, settings=settings)

    # Mock channel
    import discord as _discord
    channel = MagicMock(spec=_discord.TextChannel)
    channel.send = AsyncMock()

    async def fake_fetch(cid):
        # Should be asked for the cards channel
        assert cid == 200
        return channel

    bot.get_channel = MagicMock(return_value=None)
    bot.fetch_channel = AsyncMock(side_effect=fake_fetch)

    card = MagicMock()
    card.text = "test card"
    card.category = "production"

    movie = MagicMock()
    movie.title = "X"
    movie.year = 2000

    await sender.post_card(card, movie)

    channel.send.assert_awaited_once()
    _args, kwargs = channel.send.call_args
    assert "embed" in kwargs


async def test_bot_sender_uses_system_channel_for_system(db_and_settings) -> None:
    """post_system routes to discord_system_channel_id."""
    db, settings = db_and_settings
    settings.discord_bot_channel_id = 100
    settings.discord_cards_channel_id = 200
    settings.discord_system_channel_id = 300

    bot = _make_bot(db_and_settings)
    sender = BotDiscordSender(bot=bot, settings=settings)

    import discord as _discord
    channel = MagicMock(spec=_discord.TextChannel)
    channel.send = AsyncMock()

    async def fake_fetch(cid):
        assert cid == 300  # system channel
        return channel

    bot.get_channel = MagicMock(return_value=None)
    bot.fetch_channel = AsyncMock(side_effect=fake_fetch)

    await sender.post_system("hello")

    channel.send.assert_awaited_once()
    _args, kwargs = channel.send.call_args
    assert kwargs.get("content") == "hello"


async def test_bot_sender_falls_back_to_bot_channel_when_specific_unset(
    db_and_settings,
) -> None:
    """If cards/system channel IDs aren't set, fall back to bot_channel_id."""
    db, settings = db_and_settings
    settings.discord_bot_channel_id = 100
    settings.discord_cards_channel_id = 0   # not set
    settings.discord_system_channel_id = 0  # not set

    bot = _make_bot(db_and_settings)
    sender = BotDiscordSender(bot=bot, settings=settings)

    import discord as _discord
    channel = MagicMock(spec=_discord.TextChannel)
    channel.send = AsyncMock()

    seen_channels: list[int] = []

    async def fake_fetch(cid):
        seen_channels.append(cid)
        return channel

    bot.get_channel = MagicMock(return_value=None)
    bot.fetch_channel = AsyncMock(side_effect=fake_fetch)

    await sender.post_system("system message")

    card = MagicMock()
    card.text = "x"
    card.category = "production"
    movie = MagicMock()
    movie.title = "X"
    movie.year = 2000
    await sender.post_card(card, movie)

    # Both calls fell back to the bot channel.
    assert seen_channels == [100, 100]


async def test_dismiss_clear_command_no_op_when_not_dismissed(db_and_settings) -> None:
    db, settings = db_and_settings
    await db.upsert_movie(
        plex_guid="plex://movie/abc",
        title="X",
        imdb_id="tt0081505",
    )
    bot = _make_bot(db_and_settings)
    interaction = _make_interaction(123456)

    await bot._handle_dismiss_clear_command(interaction, "tt0081505")

    args, kwargs = interaction.response.send_message.call_args
    assert "wasn't dismissed" in args[0]
    assert kwargs.get("ephemeral") is True
