from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from poptrivia.models import RawSourceItem
from poptrivia.prep.sources._cache import cache_path, read_text, write_text

log = logging.getLogger("poptrivia.prep.imdb")

_BASE = "https://www.imdb.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class IMDBScrapeError(RuntimeError):
    """Raised when an IMDB page cannot be fetched or parsed."""


async def fetch_trivia(imdb_id: str, cache_dir: Path) -> list[RawSourceItem]:
    html = await _fetch_html(imdb_id, "trivia", cache_dir)
    return _parse_items(html, "imdb_trivia")


async def fetch_goofs(imdb_id: str, cache_dir: Path) -> list[RawSourceItem]:
    html = await _fetch_html(imdb_id, "goofs", cache_dir)
    return _parse_items(html, "imdb_goofs")


# ─── internals ────────────────────────────────────────────────────────────


async def _fetch_html(imdb_id: str, section: str, cache_dir: Path) -> str:
    if not imdb_id.startswith("tt"):
        raise IMDBScrapeError(f"Bad IMDB ID: {imdb_id!r}")

    path = cache_path(cache_dir, "imdb", f"{imdb_id}_{section}", "html")
    cached = read_text(path)
    if cached is not None:
        return cached

    url = f"{_BASE}/title/{imdb_id}/{section}/"
    log.info("Fetching IMDB %s for %s", section, imdb_id)
    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS, follow_redirects=True) as c:
        r = await c.get(url)
    if r.status_code != 200:
        raise IMDBScrapeError(f"IMDB {section} for {imdb_id}: HTTP {r.status_code}")
    write_text(path, r.text)
    return r.text


def _parse_items(html: str, source: str) -> list[RawSourceItem]:
    """Parse IMDB's trivia/goofs list out of the page.

    IMDB's HTML is volatile across redesigns. We try the modern list-item
    structure first, then fall back to old-style `.sodatext` divs. If both
    fail we return nothing rather than crash; the prep pipeline can still
    proceed with the remaining sources.
    """
    soup = BeautifulSoup(html, "html.parser")

    candidates: list[str] = []

    # Modern layout: each item lives in an <li class="ipc-metadata-list__item">
    # with the body in <div class="ipc-html-content-inner-div"> or a
    # data-testid="sub-section-..." container.
    for li in soup.select("li.ipc-metadata-list__item"):
        inner = li.select_one("div.ipc-html-content-inner-div")
        if inner is None:
            continue
        text = _clean(inner.get_text(" ", strip=True))
        if text:
            candidates.append(text)

    # Old layout: <div class="sodatext"> per item.
    if not candidates:
        for div in soup.select("div.sodatext"):
            text = _clean(div.get_text(" ", strip=True))
            if text:
                candidates.append(text)

    # Last-resort: any element with data-testid containing 'item-id' (recent
    # IMDB A/B variants).
    if not candidates:
        for el in soup.select('[data-testid^="item-id"]'):
            text = _clean(el.get_text(" ", strip=True))
            if text:
                candidates.append(text)

    log.info("IMDB %s parsed %d items", source, len(candidates))
    return [RawSourceItem(source=source, text=t) for t in candidates]  # type: ignore[arg-type]


_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text).strip()
    # IMDB appends helpful but noisy "X of Y people found this interesting"
    # tails. Drop them.
    text = re.sub(
        r"\s*\d+\s+of\s+\d+\s+(?:found this interesting|users found this interesting)\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()
