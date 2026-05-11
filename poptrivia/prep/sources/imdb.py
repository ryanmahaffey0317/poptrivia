from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import cloudscraper
from bs4 import BeautifulSoup

from poptrivia.models import RawSourceItem
from poptrivia.prep.sources._cache import cache_path, read_text, write_text

log = logging.getLogger("poptrivia.prep.imdb")

_BASE = "https://www.imdb.com"

# A full browser fingerprint that pairs with cloudscraper's JS-challenge
# handling. cloudscraper inherits these and adds its own challenge cookies
# during the first GET.
_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_RETRY_DELAYS_SECONDS = (3, 8, 15)
_REQUEST_TIMEOUT_SECONDS = 30


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
    referer = f"{_BASE}/title/{imdb_id}/"
    log.info("Fetching IMDB %s for %s (cloudscraper)", section, imdb_id)

    # cloudscraper is synchronous; run the whole session on a worker thread
    # so the event loop stays free.
    last_status, html = await asyncio.to_thread(
        _scrape_with_retries, url=url, referer=referer
    )

    if html is None:
        raise IMDBScrapeError(
            f"IMDB {section} for {imdb_id}: HTTP {last_status} after retries. "
            "If this persists, fall back to manual sourcing with "
            "`scripts/prep_movie.py --sources-file PATH` "
            "(paste the IMDB trivia/goofs text into a .txt file)."
        )

    write_text(path, html)
    return html


def _scrape_with_retries(*, url: str, referer: str) -> tuple[int | None, str | None]:
    """Synchronous scrape with cookie warmup + retry on 202.

    Returns (last_status, html_or_None). Only retries on 202; any other
    failure short-circuits.
    """
    headers: dict[str, str] = {**_HEADERS, "Referer": referer}
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    scraper.headers.update(headers)

    # Cookie warmup — hit the title page so Cloudflare sees a "first-party"
    # session before we ask for trivia/goofs. Non-fatal.
    try:
        scraper.get(referer, timeout=_REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        log.debug("IMDB referer warmup failed (non-fatal): %s", e)

    last_status: int | None = None
    for attempt, delay in enumerate((0,) + _RETRY_DELAYS_SECONDS):
        if delay:
            log.info(
                "IMDB %s returned %s — retrying in %ds (attempt %d/%d)",
                url,
                last_status,
                delay,
                attempt,
                len(_RETRY_DELAYS_SECONDS),
            )
            _sleep(delay)
        try:
            r = scraper.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            log.warning("IMDB request raised %s: %s", type(e).__name__, e)
            last_status = None
            break
        last_status = r.status_code
        if r.status_code == 200 and r.text:
            return last_status, r.text
        # Any non-202 failure is unlikely to fix itself with a retry.
        if r.status_code != 202:
            break

    return last_status, None


def _sleep(seconds: int) -> None:
    """Blocking sleep (we're inside asyncio.to_thread)."""
    import time

    time.sleep(seconds)


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

    # Last-resort: any element with data-testid containing 'item-id'.
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
    text = re.sub(
        r"\s*\d+\s+of\s+\d+\s+(?:found this interesting|users found this interesting)\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()
