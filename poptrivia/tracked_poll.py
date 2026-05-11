from __future__ import annotations

import asyncio
import logging

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.models import MovieStatus
from poptrivia.tracked import is_tracked, read_tracked

log = logging.getLogger("poptrivia.tracked_poll")


class TrackedPoller:
    """Periodically scan tracked.txt × movies-in-DB and queue prep for any
    tracked movie whose metadata we have but whose track isn't ready.

    Two complementary halves of the curated model:
      - This task: covers "user added a movie they've watched before."
      - Webhook:   covers "user is watching a tracked movie right now."
    """

    def __init__(self, *, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="tracked-poller")

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        interval = self.settings.tracked_poll_interval_seconds
        log.info("Tracked-list poller started (every %ds)", interval)
        # Do one pass immediately at startup, then loop on the timer.
        while not self._stopped.is_set():
            try:
                await self.poll_once()
            except Exception as e:
                log.exception("Tracked poll crashed: %s", e)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
        log.info("Tracked-list poller stopped")

    async def poll_once(self) -> int:
        """Run one scan. Returns the number of jobs queued this pass."""
        tracked = read_tracked(self.settings.tracked_list_path)
        if not tracked:
            log.debug("Tracked list is empty (%s)", self.settings.tracked_list_path)
            return 0

        movies = await self.db.list_movies()
        queued = 0
        unmatched = set(tracked)

        for movie in movies:
            if not is_tracked(
                tracked=tracked,
                imdb_id=movie.imdb_id,
                plex_guid=movie.plex_guid,
            ):
                continue
            if movie.imdb_id:
                unmatched.discard(movie.imdb_id.lower())
            unmatched.discard(movie.plex_guid.lower())

            if movie.status not in (MovieStatus.NOT_STARTED, MovieStatus.FAILED):
                continue
            if await self.db.has_active_job(movie.plex_guid):
                continue

            job_id = await self.db.enqueue_job(movie.plex_guid)
            await self.db.set_movie_status(movie.plex_guid, MovieStatus.QUEUED)
            log.info(
                "Tracked poll: queued prep job %d for %s (%s)",
                job_id,
                movie.title,
                movie.imdb_id or movie.plex_guid,
            )
            queued += 1

        if unmatched:
            log.info(
                "Tracked entries with no captured metadata yet (will queue once played at least once): %s",
                sorted(unmatched),
            )
        if queued:
            log.info("Tracked poll: queued %d new prep job(s)", queued)
        return queued
