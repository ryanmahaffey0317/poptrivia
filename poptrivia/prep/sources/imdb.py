from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import (
    Browser,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from poptrivia.models import RawSourceItem
from poptrivia.prep.sources._cache import cache_path, read_text, write_text

log = logging.getLogger("poptrivia.prep.imdb")

_BASE = "https://www.imdb.com"

# Realistic UA + locale to pair with Chromium. IMDB sits behind Cloudflare;
# a real headless browser passes the JS challenge that cloudscraper couldn't.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Minimal "stealth" — strips the headless flags Cloudflare looks for. Runs
# before every page load on a context.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});
window.chrome = { runtime: {} };
"""

_NAV_TIMEOUT_MS = 30_000
_POST_LOAD_SETTLE_MS = 2_000  # Let Cloudflare's JS check resolve.


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
    log.info("Fetching IMDB %s for %s (playwright/chromium)", section, imdb_id)

    html = await _scrape_via_playwright(url=url, referer=referer)
    if html is None:
        raise IMDBScrapeError(
            f"IMDB {section} for {imdb_id}: Playwright could not retrieve "
            "page content (Cloudflare challenge unresolved or timeout)."
        )

    write_text(path, html)
    return html


async def _scrape_via_playwright(*, url: str, referer: str) -> str | None:
    """Open a headless Chromium, warm the session via the title page,
    then fetch the trivia/goofs page. Returns rendered HTML or None.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        try:
            return await _scrape_with_browser(browser, url=url, referer=referer)
        finally:
            await browser.close()


async def _scrape_with_browser(
    browser: Browser, *, url: str, referer: str
) -> str | None:
    context = await browser.new_context(
        user_agent=_USER_AGENT,
        locale="en-US",
        viewport={"width": 1280, "height": 720},
        java_script_enabled=True,
    )
    await context.add_init_script(_STEALTH_INIT_SCRIPT)
    try:
        page = await context.new_page()

        # 1. Cookie warmup via the title page.
        try:
            await page.goto(
                referer, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS
            )
            await page.wait_for_timeout(_POST_LOAD_SETTLE_MS)
        except PlaywrightTimeoutError as e:
            log.debug("IMDB referer warmup timed out: %s", e)

        # 2. The actual target.
        try:
            await page.goto(
                url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS
            )
        except PlaywrightTimeoutError as e:
            log.warning("IMDB navigation timed out for %s: %s", url, e)
            return None
        # Let Cloudflare's JS check resolve.
        await page.wait_for_timeout(_POST_LOAD_SETTLE_MS)

        # If we landed on a Cloudflare challenge page, the content will be
        # very short and have no trivia/goofs markup. Heuristic check:
        html = await page.content()
        if _looks_like_challenge(html):
            log.warning("Playwright landed on a Cloudflare challenge page for %s", url)
            # Give it one more beat in case the JS challenge is still solving.
            await page.wait_for_timeout(5_000)
            html = await page.content()
            if _looks_like_challenge(html):
                return None

        return html
    finally:
        await context.close()


def _looks_like_challenge(html: str) -> bool:
    # Cloudflare's interstitials all share these markers.
    return (
        "Just a moment" in html
        or "Checking your browser" in html
        or "cf-browser-verification" in html
    )


# ─── parsing (unchanged) ──────────────────────────────────────────────────


def _parse_items(html: str, source: str) -> list[RawSourceItem]:
    soup = BeautifulSoup(html, "html.parser")

    candidates: list[str] = []

    for li in soup.select("li.ipc-metadata-list__item"):
        inner = li.select_one("div.ipc-html-content-inner-div")
        if inner is None:
            continue
        text = _clean(inner.get_text(" ", strip=True))
        if text:
            candidates.append(text)

    if not candidates:
        for div in soup.select("div.sodatext"):
            text = _clean(div.get_text(" ", strip=True))
            if text:
                candidates.append(text)

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
