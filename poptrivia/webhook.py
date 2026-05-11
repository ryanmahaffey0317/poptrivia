from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from poptrivia.models import MovieStatus, TautulliEvent

log = logging.getLogger("poptrivia.webhook")

router = APIRouter()


# Tautulli sends events for many actions; we only care about playback-start
# style events that imply "someone is now watching". The rest get logged and
# dropped (they'll matter once Step 12 wires SessionMonitor in).
_START_EVENTS = {"play", "start", "playback.start", "playback_start"}


@router.post("/tautulli-webhook")
async def tautulli_webhook(request: Request) -> dict[str, Any]:
    raw_bytes = await request.body()
    content_type = request.headers.get("content-type", "")

    try:
        raw: Any = json.loads(raw_bytes) if raw_bytes else None
    except json.JSONDecodeError:
        log.warning(
            "Tautulli webhook: non-JSON body (content-type=%s, %d bytes): %r",
            content_type,
            len(raw_bytes),
            raw_bytes[:1000],
        )
        return {"ok": False, "reason": "non-json body logged"}

    log.info(
        "Tautulli webhook received (content-type=%s)\n%s",
        content_type,
        json.dumps(raw, indent=2, sort_keys=True, default=str),
    )

    if not isinstance(raw, dict):
        return {"ok": False, "reason": "payload not an object"}

    try:
        event = TautulliEvent.from_raw(raw)
    except ValidationError as e:
        log.warning("Tautulli webhook: payload failed validation: %s", e)
        return {"ok": False, "reason": "validation failed"}

    settings = request.app.state.settings
    db = request.app.state.db

    # ─── filters ────────────────────────────────────────────────────
    if event.media_type != "movie":
        log.info("Ignoring non-movie event (media_type=%r)", event.media_type)
        return {"ok": True, "skipped": "not a movie"}

    if event.username not in settings.monitored_users:
        log.info(
            "Ignoring event for non-monitored user %r (monitored=%s)",
            event.username,
            sorted(settings.monitored_users),
        )
        return {"ok": True, "skipped": "user not monitored"}

    # ─── upsert movie row (always, so we can observe state) ─────────
    movie = await db.upsert_movie(
        plex_guid=event.plex_guid,
        title=event.title,
        year=event.year,
        imdb_id=event.imdb_id,
        tmdb_id=event.tmdb_id,
        file_path=event.file,
    )

    # ─── routing ────────────────────────────────────────────────────
    if event.event.lower() not in _START_EVENTS:
        log.info(
            "Movie event but not a start event (event=%r, status=%s) — no routing",
            event.event,
            movie.status.value,
        )
        return {"ok": True, "status": movie.status.value, "skipped": "non-start event"}

    if movie.status == MovieStatus.READY:
        # SessionMonitor wiring lands in Step 12; for now log the intent.
        log.info(
            "Movie ready, would start SessionMonitor | guid=%s session_key=%s track=%s",
            movie.plex_guid,
            event.session_key,
            movie.track_path,
        )
        return {"ok": True, "action": "would_start_monitor", "status": movie.status.value}

    if movie.status in (MovieStatus.NOT_STARTED, MovieStatus.FAILED):
        # Guard against accidentally double-queueing: only enqueue if no
        # pending/running job exists for this guid.
        if await db.has_active_job(event.plex_guid):
            log.info(
                "Prep job already pending/running for %s — no new enqueue",
                event.plex_guid,
            )
        else:
            job_id = await db.enqueue_job(event.plex_guid)
            await db.set_movie_status(event.plex_guid, MovieStatus.QUEUED)
            log.info(
                "Queued prep job %d for %s (%s)", job_id, event.plex_guid, event.title
            )
        return {"ok": True, "action": "queued", "status": MovieStatus.QUEUED.value}

    # status is queued or generating — just acknowledge.
    log.info(
        "Prep already in progress for %s (status=%s) — no action",
        event.plex_guid,
        movie.status.value,
    )
    return {"ok": True, "action": "in_progress", "status": movie.status.value}
