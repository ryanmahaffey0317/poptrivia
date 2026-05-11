from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request

log = logging.getLogger("poptrivia.webhook")

router = APIRouter()


@router.post("/tautulli-webhook")
async def tautulli_webhook(request: Request) -> dict[str, Any]:
    """Log the entire payload Tautulli sends so we can pin down the schema.

    During step 2 this is intentionally a sink: no filtering, no routing.
    """
    raw = await request.body()
    content_type = request.headers.get("content-type", "")

    payload: Any
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        # Tautulli has been known to send form-encoded bodies depending on the
        # notification agent config. Log the raw bytes so we can see what's up.
        log.warning(
            "Tautulli webhook: non-JSON body (content-type=%s, %d bytes): %r",
            content_type,
            len(raw),
            raw[:1000],
        )
        return {"ok": False, "reason": "non-json body logged"}

    log.info(
        "Tautulli webhook received (content-type=%s)\n%s",
        content_type,
        json.dumps(payload, indent=2, sort_keys=True, default=str),
    )
    return {"ok": True}
