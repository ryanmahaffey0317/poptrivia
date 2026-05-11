from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from poptrivia.prep.sources._cache import cache_path, read_text, write_text

log = logging.getLogger("poptrivia.prep.tmdb")

_BASE = "https://api.themoviedb.org/3"


class TMDBError(RuntimeError):
    pass


async def fetch_metadata(
    *,
    tmdb_id: str | None = None,
    imdb_id: str | None = None,
    api_key: str,
    cache_dir: Path,
) -> dict[str, Any]:
    """Fetch movie metadata + cast/crew/keywords from TMDB.

    Cached as a single JSON blob per movie. Either tmdb_id or imdb_id must
    be supplied; if only imdb_id is given we resolve via /find first.
    """
    if not api_key:
        raise TMDBError("TMDB_API_KEY is not set")
    if not (tmdb_id or imdb_id):
        raise TMDBError("fetch_metadata requires tmdb_id or imdb_id")

    key = tmdb_id or imdb_id
    assert key is not None
    cache = cache_path(cache_dir, "tmdb", str(key), "json")
    cached = read_text(cache)
    if cached is not None:
        return json.loads(cached)

    async with httpx.AsyncClient(timeout=20.0) as c:
        resolved_id = tmdb_id or await _resolve_from_imdb(c, imdb_id, api_key)
        details = await _get(
            c,
            f"/movie/{resolved_id}",
            api_key,
            params={"append_to_response": "credits,keywords,release_dates"},
        )

    write_text(cache, json.dumps(details, indent=2))
    return details


# ─── internals ────────────────────────────────────────────────────────────


async def _resolve_from_imdb(
    client: httpx.AsyncClient, imdb_id: str | None, api_key: str
) -> str:
    if not imdb_id:
        raise TMDBError("Cannot resolve TMDB id without imdb_id")
    data = await _get(
        client,
        f"/find/{imdb_id}",
        api_key,
        params={"external_source": "imdb_id"},
    )
    results = data.get("movie_results") or []
    if not results:
        raise TMDBError(f"TMDB /find returned no movie for {imdb_id}")
    return str(results[0]["id"])


async def _get(
    client: httpx.AsyncClient,
    path: str,
    api_key: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    p = {"api_key": api_key, **(params or {})}
    r = await client.get(f"{_BASE}{path}", params=p)
    if r.status_code != 200:
        raise TMDBError(f"TMDB {path}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()
