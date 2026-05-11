from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from poptrivia.models import Movie, MovieStatus
from poptrivia.tracked import append_to_tracked

if TYPE_CHECKING:
    from poptrivia.config import Settings
    from poptrivia.db import Database

log = logging.getLogger("poptrivia.discord_bot")


def _intents() -> discord.Intents:
    """Minimum intents we need.

    We don't read message content; we only post messages and receive
    interaction (button click) events. Server Members intent helps when
    resolving user mentions / IDs but isn't strictly required.
    """
    intents = discord.Intents.default()
    intents.message_content = False
    intents.members = True
    return intents


class PoptriviaBot(commands.Bot):
    """discord.py bot for interactive opt-in prompts.

    Lifecycle:
      - Started in FastAPI lifespan as an asyncio task (start_task).
      - on_ready: log + sync slash commands.
      - Posts prompts via post_prompt(movie).
      - Handles button clicks via GeneratePromptView.
      - Stop via stop().
    """

    def __init__(self, *, settings: "Settings", db: "Database"):
        super().__init__(command_prefix="!unused", intents=_intents())
        self.settings = settings
        self.db = db
        self._channel: discord.abc.Messageable | None = None

    # ─── lifecycle ──────────────────────────────────────────────────

    async def on_ready(self) -> None:
        log.info(
            "Bot connected as %s (id=%s); syncing application commands",
            self.user,
            self.user.id if self.user else "?",
        )
        try:
            synced = await self.tree.sync()
            log.info("Synced %d application command(s)", len(synced))
        except Exception as e:
            log.warning("Slash command sync failed: %s", e)

    async def setup_hook(self) -> None:
        # Register the /track slash command.
        @self.tree.command(
            name="track",
            description="Add a movie to the trivia tracked list by IMDB id.",
        )
        async def track_cmd(  # noqa: D401
            interaction: discord.Interaction, imdb_id: str, comment: str | None = None
        ) -> None:
            await self._handle_track_command(interaction, imdb_id, comment)

    async def get_prompt_channel(self) -> discord.abc.Messageable | None:
        """Lazy-fetch the configured channel."""
        if self._channel is not None:
            return self._channel
        cid = self.settings.discord_bot_channel_id
        if not cid:
            log.warning("DISCORD_BOT_CHANNEL_ID not set; cannot post prompts")
            return None
        channel = self.get_channel(cid)
        if channel is None:
            try:
                channel = await self.fetch_channel(cid)
            except discord.HTTPException as e:
                log.warning("Could not fetch channel %s: %s", cid, e)
                return None
        if not isinstance(channel, discord.abc.Messageable):
            log.warning("Channel %s is not messageable: %r", cid, channel)
            return None
        self._channel = channel
        return channel

    # ─── prompt posting ─────────────────────────────────────────────

    async def post_prompt(self, movie: Movie) -> None:
        """Send the 'generate trivia for this?' button prompt to the
        configured channel. Idempotent at the channel level — callers
        upstream (PromptTimer) are responsible for not double-firing
        per session."""
        channel = await self.get_prompt_channel()
        if channel is None:
            return

        year_str = f" ({movie.year})" if movie.year else ""
        embed = discord.Embed(
            title=f"📽️ Now playing: {movie.title}{year_str}",
            description=(
                "This movie isn't on your trivia list. Want me to generate "
                "a track for next time?"
            ),
            color=discord.Color.blurple(),
        )
        if movie.imdb_id:
            embed.set_footer(text=f"IMDB {movie.imdb_id}")

        view = GeneratePromptView(
            plex_guid=movie.plex_guid,
            imdb_id=movie.imdb_id,
            title=movie.title,
            year=movie.year,
            settings=self.settings,
            db=self.db,
        )
        try:
            await channel.send(embed=embed, view=view)
        except discord.HTTPException as e:
            log.warning("Failed to send prompt for %s: %s", movie.plex_guid, e)

    # ─── /track slash command ──────────────────────────────────────

    async def _handle_track_command(
        self,
        interaction: discord.Interaction,
        imdb_id: str,
        comment: str | None,
    ) -> None:
        if interaction.user.id not in self.settings.discord_approved_users:
            await interaction.response.send_message(
                "Not authorized to modify the trivia list.", ephemeral=True
            )
            return
        imdb_id = imdb_id.strip()
        if not imdb_id.startswith("tt"):
            await interaction.response.send_message(
                f"`{imdb_id}` doesn't look like an IMDB id (expected `tt...`).",
                ephemeral=True,
            )
            return
        try:
            append_to_tracked(
                self.settings.tracked_list_path, imdb_id, comment=comment
            )
        except Exception as e:
            log.exception("Failed to append to tracked list: %s", e)
            await interaction.response.send_message(
                f"Couldn't update the tracked list: `{e}`", ephemeral=True
            )
            return
        await interaction.response.send_message(
            f"✅ Added `{imdb_id}` to the tracked list. "
            "Prep will queue on the next poll cycle (or immediately if the "
            "movie has been seen before).",
            ephemeral=False,
        )


class GeneratePromptView(discord.ui.View):
    """The two-button view attached to each 'now playing' prompt."""

    def __init__(
        self,
        *,
        plex_guid: str,
        imdb_id: str | None,
        title: str,
        year: int | None,
        settings: "Settings",
        db: "Database",
    ):
        # 24h timeout. After that, buttons become unresponsive; user can
        # re-trigger the prompt by replaying the movie or use /track.
        super().__init__(timeout=24 * 3600)
        self.plex_guid = plex_guid
        self.imdb_id = imdb_id
        self.title = title
        self.year = year
        self.settings = settings
        self.db = db

    async def _user_is_approved(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id in self.settings.discord_approved_users:
            return True
        await interaction.response.send_message(
            "Not authorized to approve trivia prep.", ephemeral=True
        )
        return False

    @discord.ui.button(label="✨ Generate", style=discord.ButtonStyle.primary)
    async def generate(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        if not await self._user_is_approved(interaction):
            return

        identifier = self.imdb_id or self.plex_guid
        year_str = f" ({self.year})" if self.year else ""
        comment = f"{self.title}{year_str}"

        try:
            append_to_tracked(
                self.settings.tracked_list_path, identifier, comment=comment
            )
        except Exception as e:
            log.exception("Failed to append to tracked list: %s", e)
            await interaction.response.send_message(
                f"Couldn't update the tracked list: `{e}`", ephemeral=True
            )
            return

        # Queue prep immediately (don't wait for the 30-min poll).
        try:
            movie = await self.db.get_movie(self.plex_guid)
            if movie is None:
                log.warning(
                    "Approved Generate but no movie row exists for %s",
                    self.plex_guid,
                )
            else:
                if not await self.db.has_active_job(self.plex_guid):
                    await self.db.enqueue_job(self.plex_guid)
                    await self.db.set_movie_status(
                        self.plex_guid, MovieStatus.QUEUED
                    )
        except Exception as e:
            log.exception("Could not enqueue prep job: %s", e)

        # Update the message to confirm + disable buttons.
        self._disable_all()
        try:
            await interaction.response.edit_message(
                content=(
                    f"✅ Queued **{self.title}**{year_str} — I'll let you "
                    "know when the trivia track is ready."
                ),
                embed=None,
                view=self,
            )
        except discord.HTTPException as e:
            log.warning("Could not edit prompt message: %s", e)
        self.stop()

    @discord.ui.button(label="❌ No thanks", style=discord.ButtonStyle.secondary)
    async def dismiss(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ) -> None:
        if not await self._user_is_approved(interaction):
            return

        try:
            await self.db.set_dismissed(self.plex_guid)
        except Exception as e:
            log.exception("Could not set dismissed_at: %s", e)

        year_str = f" ({self.year})" if self.year else ""
        self._disable_all()
        try:
            await interaction.response.edit_message(
                content=f"👋 Skipped **{self.title}**{year_str}.",
                embed=None,
                view=self,
            )
        except discord.HTTPException as e:
            log.warning("Could not edit prompt message: %s", e)
        self.stop()

    def _disable_all(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True


async def start_bot(bot: PoptriviaBot, token: str) -> asyncio.Task[None]:
    """Spawn the bot's gateway connection as a background task.

    Returns the task so the caller can cancel/await on shutdown.
    """
    return asyncio.create_task(bot.start(token), name="discord-bot")
