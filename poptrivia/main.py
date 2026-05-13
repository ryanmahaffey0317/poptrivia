from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from poptrivia.config import get_settings
from poptrivia.db import Database
from poptrivia.discord_client import DiscordClient
from poptrivia.prep.queue import PrepWorker
from poptrivia.prompt_timer import PromptTimerRegistry
from poptrivia.session_monitor import MonitorRegistry
from poptrivia.tracked import ensure_template
from poptrivia.tracked_poll import TrackedPoller
from poptrivia.util.logging import configure_logging
from poptrivia.webhook import router as webhook_router

log = logging.getLogger("poptrivia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    ensure_template(settings.tracked_list_path)
    log.info(
        "poptrivia starting | log_level=%s monitored_users=%s tracked_list=%s",
        settings.log_level,
        sorted(settings.monitored_users),
        settings.tracked_list_path,
    )

    db = Database(settings.db_path)
    await db.connect()

    # Optional: Discord bot for interactive opt-in prompts on untracked
    # monitored plays AND for posting cards / system notifications via
    # channel IDs (instead of webhooks). When DISCORD_BOT_TOKEN is set,
    # the bot becomes the unified Discord output path. Webhooks remain
    # as the fallback when the bot isn't configured.
    bot = None
    bot_task: asyncio.Task | None = None
    prompt_timers = PromptTimerRegistry()
    discord: object  # WebhookDiscordSender | BotDiscordSender — duck-typed
    if settings.discord_bot_token:
        try:
            from poptrivia.discord_bot import (
                BotDiscordSender,
                PoptriviaBot,
                start_bot,
            )

            bot = PoptriviaBot(settings=settings, db=db)
            bot_task = await start_bot(bot, settings.discord_bot_token)
            discord = BotDiscordSender(bot=bot, settings=settings)
            log.info(
                "Discord bot active (prompts=%s, cards=%s, system=%s, "
                "approved_users=%s, prompt_delay=%ds)",
                settings.discord_bot_channel_id,
                settings.discord_cards_channel_id or settings.discord_bot_channel_id,
                settings.discord_system_channel_id or settings.discord_bot_channel_id,
                sorted(settings.discord_approved_users),
                settings.discord_prompt_delay_seconds,
            )
        except Exception as e:
            log.exception("Discord bot startup failed; falling back to webhooks: %s", e)
            bot = None
            bot_task = None
            discord = DiscordClient(
                settings.discord_webhook_url, settings.system_webhook_url
            )
    else:
        discord = DiscordClient(
            settings.discord_webhook_url, settings.system_webhook_url
        )
        log.info("DISCORD_BOT_TOKEN not set — using webhook-based posting")

    # Shared registry of active SessionMonitor tasks. Used by:
    #   - webhook: starts a monitor on monitored-user playback of a ready movie
    #   - worker: auto-starts a monitor on prep success if the user is
    #             currently watching the movie (so cards begin firing mid-watch)
    monitor_registry = MonitorRegistry(settings=settings, discord=discord)

    worker = PrepWorker(
        settings=settings,
        db=db,
        discord=discord,
        monitor_registry=monitor_registry,
    )
    poller = TrackedPoller(settings=settings, db=db)
    worker.start()
    poller.start()

    app.state.db = db
    app.state.settings = settings
    app.state.discord = discord
    app.state.worker = worker
    app.state.poller = poller
    app.state.bot = bot
    app.state.prompt_timers = prompt_timers
    app.state.monitor_registry = monitor_registry
    # Back-compat alias for callers that referenced app.state.session_monitors.
    app.state.session_monitors = monitor_registry.active

    try:
        yield
    finally:
        await prompt_timers.cancel_all()
        if bot is not None:
            try:
                await bot.close()
            except Exception as e:
                log.warning("Bot close raised: %s", e)
        if bot_task is not None:
            bot_task.cancel()
            try:
                await bot_task
            except (asyncio.CancelledError, Exception):
                pass
        await poller.stop()
        await worker.stop()
        await discord.aclose()
        await db.close()
        log.info("poptrivia shutting down")


app = FastAPI(title="poptrivia", lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
