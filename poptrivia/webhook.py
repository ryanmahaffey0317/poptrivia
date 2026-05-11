from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from poptrivia.models import TautulliEvent

log = logging.getLogger("poptrivia.webhook")

router = APIRouter()


@router.post("/tautulli-webhook")
async def tautulli_webhook(request: Request) -> dict[str, Any]:
    """Receive Tautulli playback events.

    Step 2 behavior (still in place): log the entire payload so we can confirm
    the schema in production.

    Step 3 behavior (added): if the payload parses, upsert a row in `movies`
    so we can observe state changes via the DB. Filtering/routing on
    monitored_users + media_type lives in Step 4.
    """
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
        log.warning("Tautulli webhook: JSON payload was not an object, ignoring")
        return {"ok": False, "reason": "payload not an object"}

    try:
        event = TautulliEvent.from_raw(raw)
    except ValidationError as e:
        log.warning("Tautulli webhook: payload failed validation: %s", e)
        return {"ok": False, "reason": "validation failed"}

    db = request.app.state.db
    movie = await db.upsert_movie(
        plex_guid=event.plex_guid,
        title=event.title,
        year=event.year,
        imdb_id=event.imdb_id,
        tmdb_id=event.tmdb_id,
        file_path=event.file,
    )
    log.info(
        "Upserted movie | guid=%s title=%r year=%s status=%s",
        movie.plex_guid,
        movie.title,
        movie.year,
        movie.status.value,
    )

    return {"ok": True, "status": movie.status.value}
