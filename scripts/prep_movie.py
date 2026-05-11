#!/usr/bin/env python3
"""Manually queue a prep job.

Usage:
    python scripts/prep_movie.py --guid plex://movie/abc
    python scripts/prep_movie.py --title "The Shining" --year 1980 \
                                  --imdb tt0081505 \
                                  --file "/media/movies/The Shining (1980)/The.Shining.mkv"

    # Bypass IMDB scraping with manually-captured source text:
    python scripts/prep_movie.py --guid plex://movie/abc \
                                  --sources-file /config/manual_sources/the_shining_trivia.txt \
                                  --sources-file /config/manual_sources/the_shining_goofs.txt

The --guid form requires the movie to already exist in the DB (you'll usually
have triggered it once via a Tautulli playback). The --title form creates a
new row from scratch and is useful for prepping ahead of a planned movie
night.

--sources-file (repeatable) bypasses the IMDB network scrape entirely. Use it
when Cloudflare keeps blocking the trivia/goofs pages: open IMDB in your
browser, select the visible trivia or goofs list text, paste it into a
plain `.txt` file, and pass the path here. Wikipedia and TMDB still run.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poptrivia.config import get_settings  # noqa: E402
from poptrivia.db import Database  # noqa: E402
from poptrivia.models import MovieStatus  # noqa: E402
from poptrivia.util.logging import configure_logging  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Queue a prep job for a movie.")
    parser.add_argument("--guid", help="Plex GUID (movie must already exist in DB)")
    parser.add_argument("--title", help="Movie title (creates row)")
    parser.add_argument("--year", type=int, help="Release year")
    parser.add_argument("--imdb", help="IMDB id, e.g. tt0081505")
    parser.add_argument("--tmdb", help="TMDB id")
    parser.add_argument("--file", help="Path to media file (required if creating a row)")
    parser.add_argument(
        "--sources-file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Path to a .txt file containing IMDB trivia or goofs text "
            "manually pasted from the browser. Repeatable. When supplied, "
            "the IMDB network scrape is skipped entirely for this job."
        ),
    )
    args = parser.parse_args()

    if not args.guid and not args.title:
        parser.error("Provide --guid OR --title")

    # Validate sources-file paths up front.
    for src in args.sources_file:
        if not Path(src).is_file():
            parser.error(f"--sources-file path does not exist or is not a file: {src}")

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    db = Database(settings.db_path)
    await db.connect()
    try:
        if args.guid:
            movie = await db.get_movie(args.guid)
            if movie is None:
                print(f"No movie found with guid={args.guid}", file=sys.stderr)
                return 1
            plex_guid = args.guid
        else:
            if not args.file:
                parser.error("--file is required when creating a row via --title")
            slug = re.sub(r"[^A-Za-z0-9]+", "_", args.title).strip("_").lower()
            year_part = f"_{args.year}" if args.year else ""
            plex_guid = f"manual://{slug}{year_part}"
            await db.upsert_movie(
                plex_guid=plex_guid,
                title=args.title,
                year=args.year,
                imdb_id=args.imdb,
                tmdb_id=args.tmdb,
                file_path=args.file,
            )
            print(f"Created movie row with plex_guid={plex_guid}")

        if await db.has_active_job(plex_guid):
            print(f"A prep job is already pending/running for {plex_guid}")
            return 0

        job_id = await db.enqueue_job(
            plex_guid,
            manual_sources=list(args.sources_file) or None,
        )
        await db.set_movie_status(plex_guid, MovieStatus.QUEUED)
        print(f"Queued prep job {job_id} for {plex_guid}")
        if args.sources_file:
            print(
                f"Manual sources attached ({len(args.sources_file)}): "
                f"IMDB network scrape will be skipped for this job."
            )
        print(
            "The running poptrivia container's worker will pick it up "
            "within a few seconds (or whenever Ollama is reachable)."
        )
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
