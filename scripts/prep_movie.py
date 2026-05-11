#!/usr/bin/env python3
"""Manually queue a prep job.

Usage:
    python scripts/prep_movie.py --guid plex://movie/abc
    python scripts/prep_movie.py --title "The Shining" --year 1980 \
                                  --imdb tt0081505 \
                                  --file "/media/movies/The Shining (1980)/The.Shining.mkv"

The --guid form requires the movie to already exist in the DB (you'll usually
have triggered it once via a Tautulli playback). The --title form creates a
new row from scratch and is useful for prepping ahead of a planned movie
night.
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
    args = parser.parse_args()

    if not args.guid and not args.title:
        parser.error("Provide --guid OR --title")

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

        job_id = await db.enqueue_job(plex_guid)
        await db.set_movie_status(plex_guid, MovieStatus.QUEUED)
        print(f"Queued prep job {job_id} for {plex_guid}")
        print(
            "The running poptrivia container's worker will pick it up "
            "within a few seconds (or whenever Ollama is reachable)."
        )
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
