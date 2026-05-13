from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Protocol

from poptrivia.config import Settings
from poptrivia.models import Movie, TriviaCard
from poptrivia.tautulli_client import TautulliClient, TautulliError, TautulliSession

log = logging.getLogger("poptrivia.session")


class _Discord(Protocol):
    async def post_card(self, card: TriviaCard, movie: Movie) -> None: ...
    async def post_track_active(self, movie: Movie, card_count: int) -> None: ...
    async def post_session_summary(self, movie: Movie, fired: int, total: int) -> None: ...


# Seek detection: if (current_view_offset - last_view_offset) differs from
# wall-clock elapsed by more than this while the state is "playing", it's
# a seek.
_SEEK_THRESHOLD_MS = 10_000

# Number of consecutive "session disappeared from Tautulli" polls before we
# decide the user has stopped. One transient miss is normal during seeks.
_DISAPPEAR_LIMIT = 3


def load_track(path: Path) -> list[TriviaCard]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TriviaCard.model_validate(c) for c in data.get("cards", [])]


class MonitorRegistry:
    """Single owner of active SessionMonitor tasks.

    Two entry points:
      - `start_for_session(session_key, movie, track_path)`: idempotent
        per session_key. Called by the webhook on play events.
      - `auto_start_for_movie(movie, track_path)`: query Tautulli for
        any currently-active sessions playing this movie and start a
        monitor for each. Called by the prep worker on success so cards
        begin firing the moment a track becomes ready, without needing
        the user to stop + restart playback.
    """

    def __init__(self, *, settings: Settings, discord: "_Discord"):
        self._settings = settings
        self._discord = discord
        self._monitors: dict[str, SessionMonitor] = {}

    @property
    def active(self) -> dict[str, SessionMonitor]:
        return self._monitors

    async def stop_all(self) -> None:
        for mon in list(self._monitors.values()):
            try:
                await mon.stop()
            except Exception as e:
                log.warning("MonitorRegistry: stop_all on %s raised: %s",
                            mon.session_key, e)
        self._monitors.clear()

    async def start_for_session(
        self, session_key: str, movie: Movie, track_path: str
    ) -> bool:
        """Start a monitor for the given Tautulli session. Returns True
        if a new monitor was started, False if one already exists for
        this session_key or the track failed to load."""
        if session_key in self._monitors:
            log.info(
                "MonitorRegistry: already running for session_key=%s",
                session_key,
            )
            return False
        try:
            track = load_track(Path(track_path))
        except Exception as e:
            log.warning(
                "MonitorRegistry: failed to load track %s: %s", track_path, e
            )
            return False

        tautulli = TautulliClient(
            self._settings.tautulli_url, self._settings.tautulli_api_key
        )
        monitor = SessionMonitor(
            session_key=session_key,
            movie=movie,
            track=track,
            tautulli=tautulli,
            discord=self._discord,
            settings=self._settings,
        )
        task = monitor.start()

        def _cleanup_callback(_t: asyncio.Task) -> None:
            asyncio.create_task(self._cleanup(session_key, tautulli))

        task.add_done_callback(_cleanup_callback)
        self._monitors[session_key] = monitor
        log.info(
            "MonitorRegistry: started session=%s movie=%r cards=%d",
            session_key,
            movie.title,
            len(track),
        )
        return True

    async def _cleanup(
        self, session_key: str, tautulli: TautulliClient
    ) -> None:
        self._monitors.pop(session_key, None)
        await tautulli.aclose()

    async def auto_start_for_movie(
        self, movie: Movie, track_path: str
    ) -> int:
        """Query Tautulli for active sessions playing this movie and
        start a SessionMonitor for each one we don't already have.

        Returns the number of new monitors started.

        Called from the prep worker on success: when a track first
        becomes available WHILE the user is watching, this lets cards
        start firing immediately rather than requiring a play/replay.
        """
        tautulli = TautulliClient(
            self._settings.tautulli_url, self._settings.tautulli_api_key
        )
        try:
            sessions = await tautulli.get_activity()
        except TautulliError as e:
            log.warning(
                "MonitorRegistry: Tautulli poll for auto-start failed: %s", e
            )
            return 0
        finally:
            await tautulli.aclose()

        matching = [
            s
            for s in sessions
            if s.plex_guid == movie.plex_guid and s.state in ("playing", "paused")
        ]
        if not matching:
            log.info(
                "MonitorRegistry: no active session playing %s; not auto-starting",
                movie.plex_guid,
            )
            return 0

        started = 0
        for session in matching:
            if await self.start_for_session(
                session.session_key, movie, track_path
            ):
                started += 1
        if started:
            log.info(
                "MonitorRegistry: auto-started %d monitor(s) for newly-ready %r",
                started,
                movie.title,
            )
        return started


class SessionMonitor:
    """Drives one active playback session: polls Tautulli, fires due cards.

    Stop the task by calling stop() — the monitor will post a session
    summary on its way out.
    """

    def __init__(
        self,
        *,
        session_key: str,
        movie: Movie,
        track: list[TriviaCard],
        tautulli: TautulliClient,
        discord: _Discord,
        settings: Settings,
    ):
        self.session_key = session_key
        self.movie = movie
        self.track = sorted(track, key=lambda c: c.timestamp_ms)
        self.tautulli = tautulli
        self.discord = discord
        self.settings = settings

        self.fired_card_ids: set[str] = set()
        self._last_offset_ms: int | None = None
        self._last_poll_wall: float | None = None
        self._missing_polls = 0
        self._stopped = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # ─── lifecycle ──────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        self._task = asyncio.create_task(self._run(), name=f"monitor-{self.session_key}")
        return self._task

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ─── main loop ──────────────────────────────────────────────────

    async def _run(self) -> None:
        log.info(
            "SessionMonitor starting | session=%s movie=%r cards=%d",
            self.session_key,
            self.movie.title,
            len(self.track),
        )
        try:
            await self.discord.post_track_active(self.movie, card_count=len(self.track))
        except Exception as e:
            log.warning("Discord track-active post failed: %s", e)

        interval = self.settings.session_poll_interval_seconds

        try:
            while not self._stopped.is_set():
                if not await self._tick():
                    break
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            log.info("SessionMonitor %s cancelled", self.session_key)
            raise
        finally:
            try:
                await self.discord.post_session_summary(
                    self.movie, fired=len(self.fired_card_ids), total=len(self.track)
                )
            except Exception as e:
                log.warning("Discord summary post failed: %s", e)
            log.info(
                "SessionMonitor exiting | session=%s fired=%d/%d",
                self.session_key,
                len(self.fired_card_ids),
                len(self.track),
            )

    async def _tick(self) -> bool:
        """Poll once. Returns False to stop the loop."""
        try:
            session = await self.tautulli.get_session(self.session_key)
        except TautulliError as e:
            log.warning("Tautulli poll failed (%s) — will retry next tick", e)
            return True

        if session is None:
            self._missing_polls += 1
            if self._missing_polls >= _DISAPPEAR_LIMIT:
                log.info(
                    "Session %s vanished from Tautulli — stopping monitor",
                    self.session_key,
                )
                return False
            return True

        self._missing_polls = 0
        await self._process(session)
        return True

    async def _process(self, session: TautulliSession) -> None:
        now = time.monotonic()
        offset_ms = session.view_offset_ms

        # ─── seek detection ─────────────────────────────────────────
        is_seek = False
        if (
            session.state == "playing"
            and self._last_offset_ms is not None
            and self._last_poll_wall is not None
        ):
            wall_elapsed_ms = (now - self._last_poll_wall) * 1000
            playback_delta = offset_ms - self._last_offset_ms
            divergence = abs(playback_delta - wall_elapsed_ms)
            if divergence > _SEEK_THRESHOLD_MS:
                is_seek = True

        # First-poll mid-movie join: treat any card sufficiently before the
        # current offset as already fired so we don't flood the channel.
        if self._last_offset_ms is None and offset_ms > 0:
            threshold = self.settings.missed_card_threshold_seconds * 1000
            for card in self.track:
                if card.timestamp_ms < offset_ms - threshold:
                    self.fired_card_ids.add(card.id)
            log.info(
                "Mid-movie join at %d ms: marked %d cards as already fired",
                offset_ms,
                len(self.fired_card_ids),
            )

        if is_seek:
            # On any seek (forward or back) we don't fire cards this tick.
            # For backward seeks the cards we already fired stay in the set,
            # so they won't re-fire. For forward seeks any cards we jumped
            # past need to be marked fired silently (otherwise they'd flood
            # the channel as soon as the seek lands inside the lead window).
            lookahead_ms = self.settings.card_lead_time_seconds * 1000
            for card in self.track:
                if (
                    card.id not in self.fired_card_ids
                    and card.timestamp_ms < offset_ms - lookahead_ms
                ):
                    self.fired_card_ids.add(card.id)
            log.info(
                "Seek detected (state=%s, new offset=%d ms) — silenced cards before %d ms",
                session.state,
                offset_ms,
                offset_ms - lookahead_ms,
            )
            self._last_offset_ms = offset_ms
            self._last_poll_wall = now
            return

        # ─── pause: bookkeep but don't fire ──────────────────────────
        if session.state != "playing":
            log.debug(
                "Session %s state=%s at %d ms — not firing",
                self.session_key,
                session.state,
                offset_ms,
            )
            self._last_offset_ms = offset_ms
            self._last_poll_wall = now
            return

        # ─── fire any cards in the lookahead window ──────────────────
        lookahead_ms = self.settings.card_lead_time_seconds * 1000
        window_end = offset_ms + lookahead_ms
        fired_this_tick: list[TriviaCard] = []
        for card in self.track:
            if card.id in self.fired_card_ids:
                continue
            if offset_ms <= card.timestamp_ms <= window_end:
                fired_this_tick.append(card)

        for card in fired_this_tick:
            try:
                await self.discord.post_card(card, self.movie)
                self.fired_card_ids.add(card.id)
                log.info(
                    "Fired card %s at offset=%d ms (card ts=%d)",
                    card.id,
                    offset_ms,
                    card.timestamp_ms,
                )
            except Exception as e:
                log.warning("Discord post_card failed for %s: %s", card.id, e)
                # Don't mark fired — try again next tick.

        self._last_offset_ms = offset_ms
        self._last_poll_wall = now
