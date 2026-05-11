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
    discord = DiscordClient(settings.discord_webhook_url, settings.system_webhook_url)
    worker = PrepWorker(settings=settings, db=db, discord=discord)
    poller = TrackedPoller(settings=settings, db=db)
    worker.start()
    poller.start()

    # Optional: Discord bot for interactive opt-in prompts on untracked
    # monitored plays. Only spun up when DISCORD_BOT_TOKEN is configured.
    bot = None
    bot_task: asyncio.Task | None = None
    prompt_timers = PromptTimerRegistry()
    if settings.discord_bot_token:
        try:
            from poptrivia.discord_bot import PoptriviaBot, start_bot

            bot = PoptriviaBot(settings=settings, db=db)
            bot_task = await start_bot(bot, settings.discord_bot_token)
            log.info(
                "Discord bot starting (channel=%s, approved_users=%s, "
                "prompt_delay=%ds)",
                settings.discord_bot_channel_id,
                sorted(settings.discord_approved_users),
                settings.discord_prompt_delay_seconds,
            )
        except Exception as e:
            log.exception("Discord bot startup failed: %s", e)
            bot = None
            bot_task = None
    else:
        log.info("DISCORD_BOT_TOKEN not set — interactive prompts disabled")

    app.state.db = db
    app.state.settings = settings
    app.state.discord = discord
    app.state.worker = worker
    app.state.poller = poller
    app.state.bot = bot
    app.state.prompt_timers = prompt_timers
    app.state.session_monitors = {}

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
