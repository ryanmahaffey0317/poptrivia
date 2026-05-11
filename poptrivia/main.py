from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from poptrivia.config import get_settings
from poptrivia.util.logging import configure_logging

log = logging.getLogger("poptrivia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    log.info("poptrivia starting | log_level=%s monitored_users=%s",
             settings.log_level, sorted(settings.monitored_users))
    yield
    log.info("poptrivia shutting down")


app = FastAPI(title="poptrivia", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
