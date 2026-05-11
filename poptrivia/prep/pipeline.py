from __future__ import annotations

import logging
from pathlib import Path

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import Movie, MovieStatus, RawSourceItem
from poptrivia.prep.llm.client import OllamaClient
from poptrivia.prep.llm.stage1_extract import extract_facts
from poptrivia.prep.llm.stage2_align import align_facts, write_track_file
from poptrivia.prep.sources import imdb as imdb_source
from poptrivia.prep.sources import tmdb as tmdb_source
from poptrivia.prep.sources import wikipedia as wiki_source
from poptrivia.prep.subtitles import SubtitleEntry, extract as extract_subs

log = logging.getLogger("poptrivia.prep.pipeline")


class PipelineError(RuntimeError):
    """Raised when prep can't proceed (e.g. file missing, no source material)."""


async def prepare_movie(
    *,
    movie: Movie,
    settings: Settings,
    db: Database,
) -> Path:
    """Run the full prep pipeline for a single movie.

    Returns the path to the written track file. Raises PipelineError on
    fatal issues. OllamaUnavailable bubbles up to the caller so the queue
    worker can back off.
    """
    log.info("Prep starting for %s (%s)", movie.title, movie.plex_guid)
    await db.set_movie_status(movie.plex_guid, MovieStatus.GENERATING)

    # 1. Scrape source material -----------------------------------------
    source_items = await _gather_sources(movie=movie, settings=settings)
    if not source_items:
        raise PipelineError("No source material found from IMDB or Wikipedia")
    log.info("Sources: %d items", len(source_items))

    # 2. Subtitles ------------------------------------------------------
    if not movie.file_path:
        raise PipelineError("movie.file_path is empty — cannot extract subtitles")
    subs = await extract_subs(Path(movie.file_path))
    if not subs:
        raise PipelineError("Subtitle extraction returned 0 entries")

    # 3. LLM stages -----------------------------------------------------
    llm = OllamaClient(
        settings.ollama_url, settings.ollama_model, timeout=settings.ollama_timeout
    )
    try:
        facts = await extract_facts(
            movie_title=movie.title,
            movie_year=movie.year,
            source_items=source_items,
            llm=llm,
            target_count=settings.target_cards_per_movie,
        )
        if not facts:
            raise PipelineError("Stage 1 produced 0 facts")

        cards = await align_facts(
            movie_title=movie.title,
            movie_year=movie.year,
            movie_duration_ms=_estimate_duration_ms(subs),
            facts=facts,
            subtitles=subs,
            llm=llm,
            max_cards=settings.max_cards_per_movie,
        )
        if not cards:
            raise PipelineError("Stage 2 produced 0 placed cards")
    finally:
        await llm.aclose()

    # 4. Persist --------------------------------------------------------
    path = write_track_file(
        tracks_dir=settings.tracks_dir,
        plex_guid=movie.plex_guid,
        imdb_id=movie.imdb_id,
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        year=movie.year,
        model=settings.ollama_model,
        cards=cards,
    )
    await db.set_movie_status(movie.plex_guid, MovieStatus.READY, track_path=str(path))
    log.info(
        "Prep complete for %s: %d cards -> %s", movie.title, len(cards), path
    )
    return path


# ─── helpers ──────────────────────────────────────────────────────────────


async def _gather_sources(
    *, movie: Movie, settings: Settings
) -> list[RawSourceItem]:
    items: list[RawSourceItem] = []

    if movie.imdb_id:
        try:
            items.extend(await imdb_source.fetch_trivia(movie.imdb_id, settings.cache_dir))
        except Exception as e:
            log.warning("IMDB trivia fetch failed for %s: %s", movie.imdb_id, e)
        try:
            items.extend(await imdb_source.fetch_goofs(movie.imdb_id, settings.cache_dir))
        except Exception as e:
            log.warning("IMDB goofs fetch failed for %s: %s", movie.imdb_id, e)
    else:
        log.warning("No imdb_id for %s — skipping IMDB scrapers", movie.plex_guid)

    try:
        items.extend(
            await wiki_source.fetch_article(movie.title, movie.year, settings.cache_dir)
        )
    except Exception as e:
        log.warning("Wikipedia fetch failed for %r: %s", movie.title, e)

    # TMDB is metadata-only; we attach it as context inside (lead) for now.
    if settings.tmdb_api_key and (movie.tmdb_id or movie.imdb_id):
        try:
            meta = await tmdb_source.fetch_metadata(
                tmdb_id=movie.tmdb_id,
                imdb_id=movie.imdb_id,
                api_key=settings.tmdb_api_key,
                cache_dir=settings.cache_dir,
            )
            text = _tmdb_to_text(meta)
            if text:
                items.append(RawSourceItem(source="wikipedia", section="TMDB", text=text))
        except Exception as e:
            log.warning("TMDB fetch failed: %s", e)

    return items


def _tmdb_to_text(meta: dict) -> str:
    """Render the bits of TMDB metadata that are actually useful as fact source.

    Cast top-billing and crew (director, DP, composer, editor) — the rest is
    largely duplicate of what's already in Wikipedia/IMDB.
    """
    lines: list[str] = []
    title = meta.get("title")
    year = (meta.get("release_date") or "")[:4]
    if title:
        lines.append(f"TMDB metadata for {title} ({year})")
    tagline = meta.get("tagline")
    if tagline:
        lines.append(f"Tagline: {tagline}")
    runtime = meta.get("runtime")
    if runtime:
        lines.append(f"Runtime: {runtime} minutes")
    overview = meta.get("overview")
    if overview:
        lines.append(f"Overview: {overview}")

    credits = meta.get("credits") or {}
    cast = credits.get("cast") or []
    if cast:
        top = ", ".join(
            f"{c.get('name')} as {c.get('character')}"
            for c in cast[:8]
            if c.get("name")
        )
        lines.append(f"Top-billed cast: {top}")

    crew_by_job: dict[str, list[str]] = {}
    for member in credits.get("crew") or []:
        job = member.get("job")
        name = member.get("name")
        if not job or not name:
            continue
        if job in {"Director", "Director of Photography", "Original Music Composer",
                    "Editor", "Production Design", "Costume Design"}:
            crew_by_job.setdefault(job, []).append(name)
    for job, names in crew_by_job.items():
        lines.append(f"{job}: {', '.join(names)}")

    return "\n".join(lines)


def _estimate_duration_ms(subs: list[SubtitleEntry]) -> int | None:
    if not subs:
        return None
    return subs[-1].end_ms
