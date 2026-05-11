from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from poptrivia.config import get_settings
from poptrivia.db import Database
from poptrivia.discord_client import DiscordClient
from poptrivia.prep.queue import PrepWorker
from poptrivia.util.logging import configure_logging
from poptrivia.webhook import router as webhook_router

log = logging.getLogger("poptrivia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    log.info(
        "poptrivia starting | log_level=%s monitored_users=%s",
        settings.log_level,
        sorted(settings.monitored_users),
    )

    db = Database(settings.db_path)
    await db.connect()
    discord = DiscordClient(settings.discord_webhook_url, settings.system_webhook_url)
    worker = PrepWorker(settings=settings, db=db, discord=discord)
    worker.start()

    # Active SessionMonitor tasks keyed by Tautulli session_key. Populated by
    # the webhook handler in Step 12.
    app.state.db = db
    app.state.settings = settings
    app.state.discord = discord
    app.state.worker = worker
    app.state.session_monitors = {}

    try:
        yield
    finally:
        await worker.stop()
        await discord.aclose()
        await db.close()
        log.info("poptrivia shutting down")


app = FastAPI(title="poptrivia", lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
