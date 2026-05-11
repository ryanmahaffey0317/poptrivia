from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from poptrivia.tautulli_client import TautulliClient, TautulliError
from poptrivia.tracked import is_tracked, read_tracked

if TYPE_CHECKING:
    from poptrivia.config import Settings
    from poptrivia.db import Database
    from poptrivia.discord_bot import PoptriviaBot

log = logging.getLogger("poptrivia.prompt_timer")


class PromptTimerRegistry:
    """Tracks one delayed-prompt task per Tautulli session_key.

    Prevents duplicate prompts when Tautulli sends multiple start/resume
    events for the same session.
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}

    def schedule(
        self,
        *,
        session_key: str,
        plex_guid: str,
        settings: "Settings",
        db: "Database",
        bot: "PoptriviaBot",
    ) -> bool:
        """Schedule a delayed prompt for this session, if not already
        scheduled. Returns True if a new timer was created."""
        if session_key in self._tasks and not self._tasks[session_key].done():
            return False
        task = asyncio.create_task(
            _wait_then_prompt(
                session_key=session_key,
                plex_guid=plex_guid,
                settings=settings,
                db=db,
                bot=bot,
            ),
            name=f"prompt-{session_key}",
        )
        task.add_done_callback(lambda _t: self._tasks.pop(session_key, None))
        self._tasks[session_key] = task
        return True

    async def cancel_all(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()


async def _wait_then_prompt(
    *,
    session_key: str,
    plex_guid: str,
    settings: "Settings",
    db: "Database",
    bot: "PoptriviaBot",
) -> None:
    """The actual delayed task body."""
    delay = settings.discord_prompt_delay_seconds
    log.info(
        "Prompt scheduled for session=%s movie=%s in %ds",
        session_key,
        plex_guid,
        delay,
    )
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        log.info("Prompt for session=%s cancelled before delay elapsed", session_key)
        raise

    # Re-check state before firing. Several things could have changed
    # during the delay:
    #  - User stopped the movie → abort
    #  - User added the movie to tracked.txt manually → abort
    #  - User dismissed via a previous prompt → abort
    #  - Prep was queued some other way → abort
    movie = await db.get_movie(plex_guid)
    if movie is None:
        log.info("Prompt skipped: movie row vanished for %s", plex_guid)
        return
    if movie.dismissed_at is not None:
        log.info("Prompt skipped: movie %s previously dismissed", plex_guid)
        return
    if movie.status not in ("not_started", "failed"):
        # Movie is already queued / generating / ready; no point prompting.
        log.info(
            "Prompt skipped: movie %s status=%s (not eligible)",
            plex_guid,
            movie.status,
        )
        return

    tracked = read_tracked(settings.tracked_list_path)
    if is_tracked(
        tracked=tracked, imdb_id=movie.imdb_id, plex_guid=movie.plex_guid
    ):
        log.info(
            "Prompt skipped: movie %s was added to tracked list during the wait",
            plex_guid,
        )
        return

    # Verify the user actually kept watching for the delay.
    tautulli = TautulliClient(settings.tautulli_url, settings.tautulli_api_key)
    try:
        try:
            session = await tautulli.get_session(session_key)
        except TautulliError as e:
            log.warning(
                "Tautulli poll failed during prompt verify for session=%s: %s",
                session_key,
                e,
            )
            session = None
    finally:
        await tautulli.aclose()

    if session is None:
        log.info(
            "Prompt skipped: session=%s no longer active in Tautulli",
            session_key,
        )
        return

    log.info(
        "Firing prompt for session=%s movie=%s (%s)",
        session_key,
        movie.title,
        movie.imdb_id or plex_guid,
    )
    await bot.post_prompt(movie)
