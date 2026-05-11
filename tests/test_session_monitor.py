from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from poptrivia.config import Settings
from poptrivia.models import Movie, MovieStatus, TriviaCard
from poptrivia.session_monitor import SessionMonitor
from poptrivia.tautulli_client import TautulliSession


@dataclass
class FakeTautulli:
    """Returns sessions from a scripted queue; one entry per get_session call."""

    sessions: list[TautulliSession | None]
    calls: int = 0

    async def get_session(self, session_key: str) -> TautulliSession | None:
        if self.calls < len(self.sessions):
            s = self.sessions[self.calls]
        else:
            s = self.sessions[-1] if self.sessions else None
        self.calls += 1
        return s

    async def aclose(self) -> None:  # SessionMonitor doesn't call this
        pass


@dataclass
class FakeDiscord:
    posted_cards: list[str] = field(default_factory=list)
    track_active: list[int] = field(default_factory=list)
    summary: list[tuple[int, int]] = field(default_factory=list)

    async def post_card(self, card: TriviaCard, movie: Movie) -> None:
        self.posted_cards.append(card.id)

    async def post_track_active(self, movie: Movie, card_count: int) -> None:
        self.track_active.append(card_count)

    async def post_session_summary(self, movie: Movie, fired: int, total: int) -> None:
        self.summary.append((fired, total))


def _settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        plex_url="x",
        plex_token="x",
        tautulli_url="http://x",
        tautulli_api_key="x",
        monitored_users={"alice"},
        ollama_url="http://x",
        discord_webhook_url="http://x",
        config_dir=tmp_path / "config",
        tracks_dir=tmp_path / "tracks",
        cache_dir=tmp_path / "cache",
        card_lead_time_seconds=5,
        session_poll_interval_seconds=1,
        missed_card_threshold_seconds=30,
    )


def _movie() -> Movie:
    return Movie(
        plex_guid="plex://movie/abc",
        title="Test",
        year=2000,
        status=MovieStatus.READY,
    )


def _card(cid: str, ts_ms: int) -> TriviaCard:
    return TriviaCard(
        id=cid,
        timestamp_ms=ts_ms,
        text="t",
        category="production",
        interest_level=3,
        source_fact_id="f1",
    )


def _session(state: str, offset_ms: int, *, key: str = "K1") -> TautulliSession:
    return TautulliSession(
        session_key=key,
        state=state,
        view_offset_ms=offset_ms,
        duration_ms=600_000,
        username="alice",
        plex_guid="plex://movie/abc",
    )


def _make_monitor(tmp_path, sessions, track) -> tuple[SessionMonitor, FakeDiscord]:
    fake_t = FakeTautulli(sessions=sessions)
    fake_d = FakeDiscord()
    mon = SessionMonitor(
        session_key="K1",
        movie=_movie(),
        track=track,
        tautulli=fake_t,  # type: ignore[arg-type]
        discord=fake_d,
        settings=_settings(tmp_path),
    )
    return mon, fake_d


async def test_card_fires_once_in_lookahead(tmp_path) -> None:
    sessions = [
        # First tick at 5s -> baseline.
        _session("playing", 5_000),
        # Second tick at 10s -> card at 12s is within [10000, 15000]; fires.
        _session("playing", 10_000),
        # Third tick at 13s -> already fired.
        _session("playing", 13_000),
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=[_card("c1", 12_000)])
    await mon._tick()
    await mon._tick()
    await mon._tick()
    assert discord.posted_cards == ["c1"]


async def test_pause_does_not_fire(tmp_path) -> None:
    sessions = [
        _session("playing", 5_000),
        _session("paused", 10_000),  # paused exactly when card is in window
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=[_card("c1", 12_000)])
    await mon._tick()
    await mon._tick()
    assert discord.posted_cards == []


async def test_forward_seek_silences_skipped_cards(tmp_path) -> None:
    cards = [_card("c1", 60_000), _card("c2", 120_000), _card("c3", 600_000)]
    sessions = [
        _session("playing", 5_000),      # baseline
        _session("playing", 300_000),    # 5 minutes later — only 1s wall — that's a seek
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=cards)
    await mon._tick()
    # Real elapsed wall time between ticks is near zero, so 295s playback
    # delta is way past the seek threshold -> seek detected.
    await mon._tick()
    # c1 and c2 should be silenced (skipped past), c3 still pending.
    assert "c1" not in discord.posted_cards
    assert "c2" not in discord.posted_cards
    assert mon.fired_card_ids == {"c1", "c2"}


async def test_backward_seek_does_not_refire(tmp_path) -> None:
    # Fire c1, then seek back, then continue past c1's window again — must
    # not re-fire.
    sessions = [
        _session("playing", 5_000),       # baseline
        _session("playing", 10_000),      # c1 (12s) in window -> fires
        _session("playing", 13_000),      # c1 already fired
        _session("playing", 0),           # seek back to start
        _session("playing", 10_000),      # window contains c1 again
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=[_card("c1", 12_000)])
    for _ in range(5):
        await mon._tick()
    # c1 fires exactly once.
    assert discord.posted_cards == ["c1"]


async def test_mid_movie_join_marks_past_cards_as_fired(tmp_path) -> None:
    # First poll lands at 5 minutes in; cards far behind should be silenced.
    cards = [_card("c1", 10_000), _card("c2", 60_000), _card("c3", 295_000)]
    sessions = [
        _session("playing", 300_000),   # joined at 5:00
        _session("playing", 305_000),
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=cards)
    await mon._tick()
    await mon._tick()
    # c1 and c2 are way behind, threshold 30s -> silenced.
    # c3 is just 5s back; within the missed threshold so NOT silenced, but
    # also not in the lookahead window so it doesn't fire either.
    assert "c1" in mon.fired_card_ids
    assert "c2" in mon.fired_card_ids
    assert "c3" not in mon.fired_card_ids
    assert discord.posted_cards == []


async def test_session_disappears_eventually_stops(tmp_path) -> None:
    sessions = [
        _session("playing", 5_000),
        None,
        None,
        None,  # 3 consecutive Nones -> stop
    ]
    mon, discord = _make_monitor(tmp_path, sessions, track=[_card("c1", 12_000)])
    assert (await mon._tick()) is True   # session present
    assert (await mon._tick()) is True   # missing 1
    assert (await mon._tick()) is True   # missing 2
    assert (await mon._tick()) is False  # missing 3 -> stop
