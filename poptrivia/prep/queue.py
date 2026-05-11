from __future__ import annotations

import asyncio
import logging
import traceback

from poptrivia.config import Settings
from poptrivia.db import Database
from poptrivia.discord_client import DiscordClient
from poptrivia.prep.llm.client import OllamaUnavailable
from poptrivia.prep.pipeline import PipelineError, prepare_movie

log = logging.getLogger("poptrivia.prep.queue")


IDLE_POLL_SECONDS = 5
OLLAMA_BACKOFF_SECONDS = 60


class PrepWorker:
    """Single-concurrency asyncio worker that drains the prep_jobs table.

    On OllamaUnavailable the job is put back to PENDING and we sleep
    OLLAMA_BACKOFF_SECONDS before looking again. On any other failure the
    job is marked FAILED and we notify Discord.
    """

    def __init__(self, *, settings: Settings, db: Database, discord: DiscordClient):
        self.settings = settings
        self.db = db
        self.discord = discord
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="prep-worker")

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ─── main loop ──────────────────────────────────────────────────

    async def _run(self) -> None:
        log.info("Prep worker started")
        try:
            while not self._stopped.is_set():
                try:
                    handled = await self._tick()
                except Exception as e:
                    log.exception("Prep worker tick crashed: %s", e)
                    handled = False
                if not handled:
                    try:
                        await asyncio.wait_for(
                            self._stopped.wait(), timeout=IDLE_POLL_SECONDS
                        )
                    except asyncio.TimeoutError:
                        pass
        finally:
            log.info("Prep worker stopped")

    async def _tick(self) -> bool:
        job = await self.db.next_pending_job()
        if job is None:
            return False

        movie = await self.db.get_movie(job.plex_guid)
        if movie is None:
            await self.db.mark_job_failed(job.id, "movie row not found")
            log.error("Prep job %d references unknown movie %s", job.id, job.plex_guid)
            return True

        log.info(
            "Prep worker picking up job %d for %s (%s)",
            job.id,
            movie.title,
            movie.plex_guid,
        )
        await self.db.mark_job_running(job.id)

        try:
            track_path = await prepare_movie(
                movie=movie, settings=self.settings, db=self.db
            )
        except OllamaUnavailable as e:
            log.warning(
                "Prep job %d: Ollama unavailable (%s) — requeueing and sleeping %ds",
                job.id,
                e,
                OLLAMA_BACKOFF_SECONDS,
            )
            await self.db.requeue_job(job.id, f"Ollama unavailable: {e}")
            # Don't reset the movie status — keep it at GENERATING; the next
            # attempt will set it back to GENERATING again no-op.
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=OLLAMA_BACKOFF_SECONDS
                )
            except asyncio.TimeoutError:
                pass
            return True
        except (PipelineError, Exception) as e:
            tb = traceback.format_exc()
            err = f"{type(e).__name__}: {e}"
            log.error("Prep job %d failed: %s\n%s", job.id, err, tb)
            await self.db.mark_job_failed(job.id, err)
            from poptrivia.models import MovieStatus  # local import to avoid cycle

            await self.db.set_movie_status(
                movie.plex_guid, MovieStatus.FAILED, error_message=err
            )
            try:
                await self.discord.post_system(
                    f"❌ Prep failed for **{movie.title}**"
                    f"{f' ({movie.year})' if movie.year else ''}: {err[:300]}"
                )
            except Exception as discord_err:
                log.warning("Could not post failure notification: %s", discord_err)
            return True

        # Success.
        await self.db.mark_job_succeeded(job.id)
        # The pipeline already set the movie row to READY. Notify.
        try:
            updated = await self.db.get_movie(movie.plex_guid)
            card_count = _count_cards(track_path)
            label_year = f" ({movie.year})" if movie.year else ""
            await self.discord.post_system(
                f"✅ Trivia track ready for **{movie.title}**{label_year} · "
                f"{card_count} cards"
            )
            assert updated is not None
        except Exception as discord_err:
            log.warning("Could not post success notification: %s", discord_err)
        return True


def _count_cards(track_path) -> int:
    import json

    try:
        data = json.loads(track_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    cards = data.get("cards")
    return len(cards) if isinstance(cards, list) else 0
