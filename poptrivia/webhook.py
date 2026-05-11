from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from poptrivia.models import Movie, MovieStatus, TautulliEvent
from poptrivia.session_monitor import SessionMonitor, load_track
from poptrivia.tautulli_client import TautulliClient
from poptrivia.tracked import is_tracked, read_tracked

log = logging.getLogger("poptrivia.webhook")

router = APIRouter()

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

    # ─── coarse filters: skip everything we don't care about ────────
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

    # ─── capture metadata even for untracked movies ─────────────────
    # That way, when the user adds an already-watched movie to tracked.txt,
    # the next poll cycle can queue prep without requiring another play.
    movie = await db.upsert_movie(
        plex_guid=event.plex_guid,
        title=event.title,
        year=event.year,
        imdb_id=event.imdb_id,
        tmdb_id=event.tmdb_id,
        file_path=event.file,
    )

    # ─── is this movie in the curated list? ─────────────────────────
    tracked = read_tracked(settings.tracked_list_path)
    if not is_tracked(
        tracked=tracked, imdb_id=event.imdb_id, plex_guid=event.plex_guid
    ):
        # Captured the metadata; consider scheduling a Discord opt-in
        # prompt so the user can add it to the list from the couch.
        prompted = await _maybe_schedule_prompt(request, event, movie)
        log.info(
            "Movie %r (%s) is not in the tracked list — captured metadata; "
            "prompt scheduled: %s",
            event.title,
            event.imdb_id or event.plex_guid,
            prompted,
        )
        return {
            "ok": True,
            "action": "metadata_captured",
            "tracked": False,
            "prompt_scheduled": prompted,
        }

    # ─── tracked: handle by event type + status ─────────────────────
    if event.event.lower() not in _START_EVENTS:
        log.info(
            "Tracked movie event but not a start event (event=%r, status=%s)",
            event.event,
            movie.status.value,
        )
        return {"ok": True, "status": movie.status.value, "skipped": "non-start event"}

    if movie.status == MovieStatus.READY:
        if not movie.track_path:
            log.error(
                "Movie %s status=ready but track_path is empty — skipping",
                movie.plex_guid,
            )
            return {"ok": False, "reason": "ready but no track_path"}
        await _start_monitor_if_needed(
            request=request,
            session_key=event.session_key,
            movie=movie,
            track_path=movie.track_path,
        )
        return {"ok": True, "action": "monitor_started", "status": movie.status.value}

    if movie.status in (MovieStatus.NOT_STARTED, MovieStatus.FAILED):
        if await db.has_active_job(event.plex_guid):
            log.info(
                "Prep job already pending/running for %s — no new enqueue",
                event.plex_guid,
            )
        else:
            job_id = await db.enqueue_job(event.plex_guid)
            await db.set_movie_status(event.plex_guid, MovieStatus.QUEUED)
            log.info(
                "Queued prep job %d for tracked movie %s (%s)",
                job_id,
                event.plex_guid,
                event.title,
            )
        return {"ok": True, "action": "queued", "status": MovieStatus.QUEUED.value}

    # queued / generating
    log.info(
        "Prep already in progress for %s (status=%s) — no action",
        event.plex_guid,
        movie.status.value,
    )
    return {"ok": True, "action": "in_progress", "status": movie.status.value}


async def _start_monitor_if_needed(
    *,
    request: Request,
    session_key: str,
    movie: Movie,
    track_path: str,
) -> None:
    app = request.app
    monitors: dict[str, SessionMonitor] = app.state.session_monitors
    if session_key in monitors:
        log.info(
            "SessionMonitor already running for session_key=%s — leaving alone",
            session_key,
        )
        return

    track = load_track(Path(track_path))
    settings = app.state.settings
    discord = app.state.discord
    tautulli = TautulliClient(settings.tautulli_url, settings.tautulli_api_key)

    monitor = SessionMonitor(
        session_key=session_key,
        movie=movie,
        track=track,
        tautulli=tautulli,
        discord=discord,
        settings=settings,
    )
    task = monitor.start()

    async def _cleanup() -> None:
        monitors.pop(session_key, None)
        await tautulli.aclose()

    task.add_done_callback(lambda _t: asyncio.create_task(_cleanup()))
    monitors[session_key] = monitor
    log.info(
        "SessionMonitor started | session_key=%s movie=%r cards=%d",
        session_key,
        movie.title,
        len(track),
    )


async def _maybe_schedule_prompt(
    request: Request, event: TautulliEvent, movie: Movie
) -> bool:
    """Schedule a delayed Discord prompt for an untracked monitored play.

    Returns True if a new timer was created (False if disabled, dismissed,
    a timer already exists, or this isn't a start event).
    """
    settings = request.app.state.settings
    if not settings.discord_prompt_enabled:
        return False
    if not settings.discord_bot_token:
        return False  # bot disabled at config level
    if event.event.lower() not in _START_EVENTS:
        return False
    if movie.dismissed_at is not None:
        return False

    bot = getattr(request.app.state, "bot", None)
    registry = getattr(request.app.state, "prompt_timers", None)
    if bot is None or registry is None:
        return False

    return registry.schedule(
        session_key=event.session_key,
        plex_guid=event.plex_guid,
        settings=settings,
        db=request.app.state.db,
        bot=bot,
    )
