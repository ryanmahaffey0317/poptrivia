from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("poptrivia.tautulli")


@dataclass
class TautulliSession:
    session_key: str
    state: str           # "playing", "paused", "buffering", "stopped"
    view_offset_ms: int
    duration_ms: int | None
    username: str
    plex_guid: str | None


class TautulliError(RuntimeError):
    pass


class TautulliClient:
    """Wrapper around Tautulli's /api/v2 get_activity endpoint."""

    def __init__(
        self, base_url: str, api_key: str, *, client: httpx.AsyncClient | None = None
    ):
        if not base_url or not api_key:
            raise ValueError("TautulliClient requires base_url and api_key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_session(self, session_key: str) -> TautulliSession | None:
        sessions = await self.get_activity()
        for s in sessions:
            if s.session_key == session_key:
                return s
        return None

    async def get_activity(self) -> list[TautulliSession]:
        params = {"apikey": self.api_key, "cmd": "get_activity"}
        try:
            r = await self._client.get(f"{self.base_url}/api/v2", params=params)
        except httpx.HTTPError as e:
            raise TautulliError(f"Tautulli transport error: {e}") from e
        if r.status_code != 200:
            raise TautulliError(f"Tautulli HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        response_block = data.get("response", {})
        if response_block.get("result") != "success":
            raise TautulliError(f"Tautulli error response: {response_block!r}")
        sessions_raw = response_block.get("data", {}).get("sessions", [])
        return [_parse_session(s) for s in sessions_raw]


def _parse_session(raw: dict[str, Any]) -> TautulliSession:
    return TautulliSession(
        session_key=str(raw.get("session_key") or ""),
        state=str(raw.get("state") or "stopped"),
        view_offset_ms=int(raw.get("view_offset") or 0),
        duration_ms=_maybe_int(raw.get("duration")),
        username=str(raw.get("user") or raw.get("username") or ""),
        plex_guid=_first_str(raw.get("guid"), raw.get("rating_key")),
    )


def _maybe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _first_str(*values: Any) -> str | None:
    for v in values:
        if v:
            return str(v)
    return None
