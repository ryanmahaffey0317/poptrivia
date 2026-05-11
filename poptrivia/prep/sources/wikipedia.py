from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx

from poptrivia.models import RawSourceItem
from poptrivia.prep.sources._cache import cache_path, read_text, write_text

log = logging.getLogger("poptrivia.prep.wikipedia")

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "poptrivia/0.1 (Plex pop-up video; personal)",
}

# Order of preference; earlier sections weighted higher when stage 1 picks
# which chunks to feed the LLM.
PREFERRED_SECTIONS = [
    "Production",
    "Filming",
    "Development",
    "Pre-production",
    "Post-production",
    "Music",
    "Score",
    "Soundtrack",
    "Reception",
    "Release",
    "Legacy",
    "Themes",
    "Cast",
    "Writing",
]


class WikipediaError(RuntimeError):
    pass


async def fetch_article(
    title: str,
    year: int | None,
    cache_dir: Path,
) -> list[RawSourceItem]:
    """Pull the film's wikipedia article and slice it into named sections.

    We disambiguate the search by appending '(film)' or '(<year> film)' when
    we have a year — Wikipedia's search collapses film/book/etc otherwise.
    """
    key = _cache_key(title, year)
    cached_json = read_text(cache_path(cache_dir, "wikipedia", key, "json"))
    if cached_json is not None:
        return _items_from_json(cached_json)

    page_title = await _resolve_page_title(title, year)
    log.info("Wikipedia page resolved: %r -> %r", title, page_title)
    raw = await _fetch_extract(page_title)

    sections = _split_sections(raw)
    relevant = _filter_preferred(sections)
    payload = {"page": page_title, "sections": relevant}

    write_text(cache_path(cache_dir, "wikipedia", key, "json"), json.dumps(payload, indent=2))
    return _items_from_json(json.dumps(payload))


# ─── internals ────────────────────────────────────────────────────────────


def _cache_key(title: str, year: int | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    return f"{slug}_{year}" if year else slug


async def _resolve_page_title(title: str, year: int | None) -> str:
    queries: list[str] = []
    if year is not None:
        queries.append(f"{title} ({year} film)")
    queries.append(f"{title} (film)")
    queries.append(title)

    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as c:
        for q in queries:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": q,
                "srlimit": "1",
            }
            r = await c.get(_API, params=params)
            if r.status_code != 200:
                continue
            data = r.json()
            hits = data.get("query", {}).get("search", [])
            if hits:
                return hits[0]["title"]
    raise WikipediaError(f"No Wikipedia page found for {title!r} ({year})")


async def _fetch_extract(page_title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "titles": page_title,
        "redirects": "1",
    }
    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as c:
        r = await c.get(_API, params=params)
    if r.status_code != 200:
        raise WikipediaError(f"Wikipedia extract HTTP {r.status_code}")
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        raise WikipediaError("Wikipedia returned no pages")
    page = next(iter(pages.values()))
    extract = page.get("extract")
    if not extract:
        raise WikipediaError(f"Wikipedia page {page_title!r} has no extract")
    return extract


# Plaintext Wikipedia extracts use "== Section ==" markers; "=== Sub ===" for
# subsections. We collapse subsections into their parent for simplicity.
_HEADING_RE = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.MULTILINE)


def _split_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pieces = _HEADING_RE.split(text)
    # _HEADING_RE.split returns: [lead_text, eq1, title1, body1, eq2, title2, body2, ...]
    if pieces:
        lead = pieces[0].strip()
        if lead:
            out["(lead)"] = lead
    for i in range(1, len(pieces), 3):
        if i + 2 >= len(pieces):
            break
        section = pieces[i + 1].strip()
        body = pieces[i + 2].strip()
        if not section or not body:
            continue
        # Merge if duplicate (e.g. nested headings).
        prior = out.get(section, "")
        out[section] = (prior + "\n\n" + body).strip() if prior else body
    return out


def _filter_preferred(sections: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    # Always keep the lead — it's biographical/contextual gold.
    if "(lead)" in sections:
        out["(lead)"] = sections["(lead)"]
    for name in PREFERRED_SECTIONS:
        for candidate, body in sections.items():
            if candidate.lower() == name.lower() and candidate not in out:
                out[candidate] = body
    return out


def _items_from_json(blob: str) -> list[RawSourceItem]:
    payload = json.loads(blob)
    return [
        RawSourceItem(source="wikipedia", section=section, text=body)
        for section, body in payload["sections"].items()
        if body
    ]
