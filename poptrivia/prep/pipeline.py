from __future__ import annotations

import logging
from pathlib import Path

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import Movie, MovieStatus, RawSourceItem
from poptrivia.prep.chapters import ChapterProbeError, probe_metadata
from poptrivia.prep.llm.client import OllamaClient
from poptrivia.prep.llm.stage1_extract import extract_facts
from poptrivia.prep.placement import place_facts
from poptrivia.prep.sources import imdb as imdb_source
from poptrivia.prep.sources import tmdb as tmdb_source
from poptrivia.prep.sources import wikipedia as wiki_source

log = logging.getLogger("poptrivia.prep.pipeline")


class PipelineError(RuntimeError):
    """Raised when prep can't proceed (e.g. file missing, no source material)."""


async def prepare_movie(
    *,
    movie: Movie,
    settings: Settings,
    db: Database,
    manual_sources: list[Path] | None = None,
) -> Path:
    """Run the full prep pipeline for a single movie.

    Steps:
      1. Scrape source material (IMDB + Wikipedia + TMDB) or read manual files.
      2. Probe the movie file for duration + chapter list (ffprobe metadata
         only — no stream decoding).
      3. Stage 1 LLM: extract facts from source material.
      4. Place facts on the timeline using chapters (or even-spacing).
      5. Write the JSON track file.

    No subtitle extraction. The previous Stage 2 LLM call is replaced with
    algorithmic placement — chapters when present, even spacing otherwise.
    Trade-off: cards no longer fire 5 s before a specific line, but the
    pipeline works reliably on every movie regardless of subtitle quality.
    """
    log.info("Prep starting for %s (%s)", movie.title, movie.plex_guid)
    await db.set_movie_status(movie.plex_guid, MovieStatus.GENERATING)

    # 1. Source material -----------------------------------------------
    source_items = await _gather_sources(
        movie=movie, settings=settings, manual_sources=manual_sources or []
    )
    if not source_items:
        raise PipelineError("No source material found from IMDB or Wikipedia")
    log.info("Sources: %d items", len(source_items))

    # 2. Movie metadata (duration + chapters) ---------------------------
    if not movie.file_path:
        raise PipelineError("movie.file_path is empty — cannot probe metadata")
    try:
        metadata = await probe_metadata(Path(movie.file_path))
    except ChapterProbeError as e:
        raise PipelineError(f"Could not read movie metadata: {e}") from e

    # 3. Stage 1 LLM ---------------------------------------------------
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
            num_ctx=settings.ollama_num_ctx,
        )
    finally:
        await llm.aclose()

    if not facts:
        raise PipelineError("Stage 1 produced 0 facts")

    # 4. Place facts on the timeline -----------------------------------
    cards = place_facts(
        facts=facts,
        metadata=metadata,
        max_cards=settings.max_cards_per_movie,
    )
    if not cards:
        raise PipelineError("Placement produced 0 cards")

    # 5. Persist --------------------------------------------------------
    from poptrivia.prep.llm.stage2_align import write_track_file  # leftover utility

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
    *,
    movie: Movie,
    settings: Settings,
    manual_sources: list[Path],
) -> list[RawSourceItem]:
    items: list[RawSourceItem] = []

    if manual_sources:
        for src_path in manual_sources:
            try:
                text = src_path.read_text(encoding="utf-8")
            except OSError as e:
                log.warning("Could not read manual source %s: %s", src_path, e)
                continue
            text = text.strip()
            if not text:
                log.warning("Manual source %s is empty — skipping", src_path)
                continue
            items.append(RawSourceItem(source="imdb_trivia", text=text))
            log.info(
                "Loaded manual source %s (%d chars) as imdb_trivia",
                src_path,
                len(text),
            )
        log.info(
            "Skipping IMDB network scrape — %d manual source(s) supplied",
            len(manual_sources),
        )
    elif movie.imdb_id:
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
        wiki_items = await wiki_source.fetch_article(
            movie.title, movie.year, settings.cache_dir
        )
        log.info("Wikipedia parsed %d section(s) for %r", len(wiki_items), movie.title)
        items.extend(wiki_items)
    except Exception as e:
        log.warning("Wikipedia fetch failed for %r: %s", movie.title, e)

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
                log.info("TMDB metadata attached as 1 source chunk")
        except Exception as e:
            log.warning("TMDB fetch failed: %s", e)
    elif not settings.tmdb_api_key:
        log.info("TMDB skipped: TMDB_API_KEY not configured")

    return items


def _tmdb_to_text(meta: dict) -> str:
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
